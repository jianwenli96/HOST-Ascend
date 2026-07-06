# coding=utf-8
"""Deterministic alignment between all pairs of sequences in a batch."""

import os
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from config import CONFIG
from .losses import classification_loss, regression_loss, causal_loss
from .smooth_dtw import smooth_dtw_probs
from utils import check_nan
try:
    from dtaidistance import dtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False


def _alignment_is_save_rank():
  """True for single-process or distributed global rank 0 only (no file I/O on other ranks)."""
  if not torch.distributed.is_available():
    return True
  if not torch.distributed.is_initialized():
    return True
  return torch.distributed.get_rank() == 0


def _maybe_save_raw_sim12(
      raw_sim_mr,
      raw_sim_rm,
      global_step,
      training,
  ):
  """Save scaled pre-softmax sim_12 tensors (MR and RM) when CONFIG flags and step match.
  Returns (abs_mr_path, abs_rm_path) or (None, None) if nothing written."""
  if not getattr(CONFIG.ALIGNMENT, "SAVE_RAW_SIM12", False):
    return None, None
  if not training:
    return None, None
  if global_step is None:
    return None, None
  every = getattr(CONFIG.ALIGNMENT, "SAVE_RAW_SIM12_EVERY", 1000)
  if every <= 0:
    return None, None
  if global_step % every != 0:
    return None, None
  save_dir = getattr(CONFIG.ALIGNMENT, "SAVE_RAW_SIM12_DIR", "") or ""
  if not save_dir:
    return None, None
  if not _alignment_is_save_rank():
    return None, None

  os.makedirs(save_dir, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  max_b = getattr(CONFIG.ALIGNMENT, "SAVE_RAW_SIM12_MAX_BATCH", None)

  def _to_cpu_slice(t):
    if t is None:
      return None
    x = t.detach().cpu()
    if max_b is not None:
      n = int(max_b)
      if n > 0:
        x = x[:n]
    return x

  stem_mr = os.path.join(save_dir, f"sim12_mr_step{global_step:08d}_{ts}.pt")
  stem_rm = os.path.join(save_dir, f"sim12_rm_step{global_step:08d}_{ts}.pt")

  payload_mr = {
      "direction": "mr",
      "global_step": int(global_step),
      "sim12": _to_cpu_slice(raw_sim_mr),
  }

  payload_rm = {
      "direction": "rm",
      "global_step": int(global_step),
      "sim12": _to_cpu_slice(raw_sim_rm),
  }

  torch.save(payload_mr, stem_mr)
  torch.save(payload_rm, stem_rm)
  print(
      f"[SAVE_RAW_SIM12] rank0 wrote {stem_mr} and {stem_rm}",
      flush=True,
  )
  return os.path.abspath(stem_mr), os.path.abspath(stem_rm)


def _compute_variance_loss(beta, steps, seq_lens, normalize_indices, variance_lambda):
  """Computes variance loss for a given similarity matrix."""
  check_nan(beta, "beta", "_compute_variance_loss input")
  if steps is None:
    return torch.tensor(0.0, device=beta.device), torch.zeros(beta.shape[:-1], device=beta.device), torch.tensor(0.0, device=beta.device)

  if normalize_indices and seq_lens is not None:
    float_seq_lens = seq_lens.float()
    if steps.dim() == 2:
      # Batched steps: (B, T)
      # float_seq_lens: (B,)
      # Use larger epsilon for BF16 stability
      tile_seq_lens = float_seq_lens.unsqueeze(1) # (B, 1)
      cur_steps = steps.float() / (tile_seq_lens + 1e-6)
    else:
      # Single seq steps: (T)
      # float_seq_lens: scalar or (1,)
      cur_steps = steps.float() / (float_seq_lens + 1e-6)
  else:
    cur_steps = steps.float()

  check_nan(cur_steps, "cur_steps", "_compute_variance_loss")

  if beta.dim() == 3:
    b, t1, t2 = beta.shape
    # cur_steps: (B, T2) -> (B, 1, T2)
    cur_steps_tiled = cur_steps.unsqueeze(1).repeat(1, t1, 1)
    pred_time = torch.sum(cur_steps_tiled * beta, dim=-1, keepdim=True) # (B, T1, 1)
    check_nan(pred_time, "pred_time", "_compute_variance_loss")
    variance = torch.sum(torch.square(cur_steps_tiled - pred_time) * beta, dim=-1) # (B, T1)
  else:
    t1, t2 = beta.shape
    # cur_steps: (T2,)
    # beta: (T1, T2)
    pred_time = torch.sum(cur_steps * beta, dim=-1, keepdim=True) # (T1, 1)
    check_nan(pred_time, "pred_time", "_compute_variance_loss")
    variance = torch.sum(torch.square(cur_steps - pred_time) * beta, dim=-1) # (T1)
  
  check_nan(variance, "variance", "_compute_variance_loss")

  var_mean = variance.mean()

  # Increase epsilon for numerical stability in gradients (especially for BF16)
  variance_eps = 1e-5
  log_var = torch.log(variance + variance_eps)
  check_nan(log_var, "log_var", "_compute_variance_loss")
  per_step_loss = variance_lambda * log_var
  return torch.mean(per_step_loss), per_step_loss, var_mean


def pairwise_l2_distance(embs1, embs2):
  """Computes pairwise distances (squared) between all rows of embs1 and embs2.
     Uses torch.cdist for numerical stability in BF16/FP16.
  """
  check_nan(embs1, "embs1", "pairwise_l2_distance input")
  check_nan(embs2, "embs2", "pairwise_l2_distance input")

  # Enforce float32 for distance calculation to prevent overflow in squared terms
  v1 = embs1.float() 
  v2 = embs2.float()
  
  if v1.dim() == 2:
      # cdist expects (B, P, M) and (B, R, M)
      # p=2 is L2 distance. We square it to get squared L2.
      dist = torch.cdist(v1.unsqueeze(0), v2.unsqueeze(0), p=2).squeeze(0)
  elif v1.dim() == 3:
      dist = torch.cdist(v1, v2, p=2)
  else:
      raise ValueError("Invalid dim for pairwise_l2_distance")
  
  check_nan(dist, "dist (cdist output)", "pairwise_l2_distance")

  return (dist ** 2).to(dtype=embs1.dtype)


def get_scaled_similarity(embs1, embs2, similarity_type, temperature):
  """Returns similarity between each all rows of embs1 and all rows of embs2."""
  check_nan(embs1, "embs1", "get_scaled_similarity input")
  check_nan(embs2, "embs2", "get_scaled_similarity input")
  channels = float(embs1.size(-1))
  # Go for embs1 to embs2.
  if similarity_type == 'cosine':
    if embs1.dim() == 2:
        similarity = torch.matmul(embs1, embs2.t())
    else:
        similarity = torch.bmm(embs1, embs2.transpose(1, 2))
  elif similarity_type == 'l2':
    similarity = -1.0 * pairwise_l2_distance(embs1, embs2)
  else:
    raise ValueError('similarity_type can either be l2 or cosine.')

  # Scale the distance  by number of channels. This normalization helps with
  # optimization.
  if not CONFIG.ALIGNMENT.NORMALIZE_EMBEDDINGS:
    similarity /= channels
  # Scale the distance by a temperature that helps with how soft/hard the
  # alignment should be.
  similarity /= temperature

  check_nan(similarity, "similarity", "get_scaled_similarity output")

  return similarity


def align_pair_of_sequences(embs1,
                            embs2,
                            similarity_type,
                            temperature,
                            steps2=None,
                            seq_lens2=None,
                            normalize_indices=False,
                            forward_variance_lambda=0.0):
  """Align a given pair embedding sequences."""
  max_num_steps = embs1.size(-2)

  # Find distances between embs1 and embs2.
  sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
  check_nan(sim_12, "sim_12", "align_pair_of_sequences before softmax")
  
  # Compute beta: matching probability for each source frame over target frames.
  # USE_SMOOTH_DTW=True: derive from soft-DTW cumulative cost (temporal monotonicity bias).
  # USE_SMOOTH_DTW=False: plain row-wise softmax (original TCC behaviour).
  _want_cost = CONFIG.ALIGNMENT.USE_SMOOTH_DTW and CONFIG.ALIGNMENT.USE_D2TW_LOSS
  path_cost_12 = None  # (B,) soft-DTW terminal cost; None when D2TW loss is disabled
  if CONFIG.ALIGNMENT.USE_SMOOTH_DTW:
      # smooth_dtw_probs requires [B, T1, T2]; handle the 2D case (single pair).
      if sim_12.dim() == 2:
          _result = smooth_dtw_probs(
              sim_12.unsqueeze(0),
              gamma_s=CONFIG.ALIGNMENT.SMOOTH_DTW_GAMMA_S,
              gamma_f=CONFIG.ALIGNMENT.SMOOTH_DTW_GAMMA_F,
              softning=CONFIG.ALIGNMENT.SMOOTH_DTW_SOFTNING,
              bidirectional=CONFIG.ALIGNMENT.SMOOTH_DTW_BIDIRECTIONAL,
              method=CONFIG.ALIGNMENT.SMOOTH_DTW_METHOD,
              return_cost=_want_cost,
          )
          if _want_cost:
              softmaxed_sim_12, path_cost_12 = _result[0].squeeze(0), _result[1]  # [T1,T2], (1,)
          else:
              softmaxed_sim_12 = _result.squeeze(0)   # [T1, T2]
      else:
          _result = smooth_dtw_probs(
              sim_12,
              gamma_s=CONFIG.ALIGNMENT.SMOOTH_DTW_GAMMA_S,
              gamma_f=CONFIG.ALIGNMENT.SMOOTH_DTW_GAMMA_F,
              softning=CONFIG.ALIGNMENT.SMOOTH_DTW_SOFTNING,
              bidirectional=CONFIG.ALIGNMENT.SMOOTH_DTW_BIDIRECTIONAL,
              method=CONFIG.ALIGNMENT.SMOOTH_DTW_METHOD,
              return_cost=_want_cost,
          )
          if _want_cost:
              softmaxed_sim_12, path_cost_12 = _result[0], _result[1]   # [B,T1,T2], (B,)
          else:
              softmaxed_sim_12 = _result                # [B, T1, T2]
  else:
      softmaxed_sim_12 = F.softmax(sim_12, dim=-1)
  check_nan(softmaxed_sim_12, "softmaxed_sim_12", "align_pair_of_sequences")

  forward_var_loss, _, var_mean = _compute_variance_loss(
      softmaxed_sim_12, steps2, seq_lens2, normalize_indices, forward_variance_lambda)
  check_nan(forward_var_loss, "forward_var_loss", "align_pair_of_sequences")

  max_probs = softmaxed_sim_12.max(dim=-1)[0]
  stats = {
      'max': max_probs.max(),
      'min': max_probs.min(),
      'mean': max_probs.mean(),
      'var_mean': var_mean
  }

  # Calculate soft-nearest neighbors.
  if embs1.dim() == 2:
      nn_embs = torch.matmul(softmaxed_sim_12, embs2)
  else:
      nn_embs = torch.bmm(softmaxed_sim_12, embs2)
  
  check_nan(nn_embs, "nn_embs", "align_pair_of_sequences")

  # Find distances between nn_embs and embs1.
  sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)
  check_nan(sim_21, "sim_21", "align_pair_of_sequences (backward logits)")

  logits = sim_21
  if embs1.dim() == 2:
      labels = torch.eye(max_num_steps, device=embs1.device)
  else:
      batch_size = embs1.size(0)
      labels = torch.eye(max_num_steps, device=embs1.device).unsqueeze(0).repeat(batch_size, 1, 1)

  # Also return forward logits (sim_12) for debugging/visualization/DTW
  # path_cost_12: (B,) soft-DTW terminal cost, or None when USE_D2TW_LOSS=False
  return logits, labels, softmaxed_sim_12, forward_var_loss, stats, sim_12, path_cost_12


def compute_deterministic_alignment_loss(embs,
                                         steps,
                                         seq_lens,
                                         num_steps,
                                         batch_size,
                                         loss_type,
                                         similarity_type,
                                         temperature,
                                         label_smoothing,
                                         variance_lambda,
                                         huber_delta,
                                         normalize_indices,
                                         normalize_embeddings=False,
                                         forward_variance_lambda=0.0,
                                         tcc_regression_margin=0.0):
  """Compute cycle-consistency loss for all steps in each sequence."""
  if normalize_embeddings:
    embs = F.normalize(embs, p=2, dim=-1)

  labels_list = []
  logits_list = []
  steps_list = []
  seq_lens_list = []
  forward_var_loss_list = []

  for i in range(batch_size):
    for j in range(batch_size):
      # We do not align the sequence with itself.
      if i != j:
        logits, labels, _, forward_var_loss, _, _, _ = align_pair_of_sequences(embs[i],
                                                 embs[j],
                                                 similarity_type,
                                                 temperature,
                                                 steps2=steps[j],
                                                 seq_lens2=seq_lens[j],
                                                 normalize_indices=normalize_indices,
                                                 forward_variance_lambda=forward_variance_lambda)
        logits_list.append(logits)
        labels_list.append(labels)
        steps_list.append(steps[i:i+1].repeat(num_steps, 1))
        seq_lens_list.append(seq_lens[i:i+1].repeat(num_steps))
        forward_var_loss_list.append(forward_var_loss)

  if not logits_list:
      return torch.tensor(0.0, device=embs.device, requires_grad=True), {}

  logits = torch.cat(logits_list, dim=0)
  labels = torch.cat(labels_list, dim=0)
  steps = torch.cat(steps_list, dim=0)
  seq_lens = torch.cat(seq_lens_list, dim=0)
  forward_var_loss = torch.stack(forward_var_loss_list).mean()

  loss_dict = {}
  if loss_type == 'classification':
    loss = classification_loss(logits, labels, label_smoothing)
  elif 'regression' in loss_type:
    loss, reg_loss, var_loss = regression_loss(logits, labels, num_steps, steps, seq_lens,
                           loss_type, normalize_indices, variance_lambda,
                           huber_delta, tcc_regression_margin=tcc_regression_margin)
    if reg_loss is not None:
        loss_dict['regression_loss'] = reg_loss
    if var_loss is not None:
        loss_dict['variance_loss'] = var_loss
  elif loss_type == 'both':
    classif_loss = classification_loss(logits, labels, label_smoothing)
    reg_loss_combined, reg_loss, var_loss = regression_loss(logits, labels, num_steps, steps, seq_lens,
                           'regression_mse_var', normalize_indices, variance_lambda,
                           huber_delta, tcc_regression_margin=tcc_regression_margin)
    loss = classif_loss + reg_loss_combined
    loss_dict['classification_loss'] = classif_loss
    loss_dict['regression_loss'] = reg_loss
    loss_dict['variance_loss'] = var_loss
  else:
    raise ValueError('Unidentified loss_type %s.' % loss_type)

  if forward_variance_lambda > 0:
    loss = loss + forward_var_loss
    loss_dict['forward_variance_loss'] = forward_var_loss

  return loss, loss_dict


def detect_sequence_direction(steps):
  """Detect if sequence is forward (1) or reverse (-1).
  
  Args:
      steps: Tensor (B, T), timestamps/indices for each frame
  
  Returns:
      direction: Tensor (B,), 1 for forward, -1 for reverse, 0 for constant
  """
  return torch.sign(steps[:, -1] - steps[:, 0])


def extract_alignment_indices_from_sim_matrix_dtw(sim_matrix, temperature, window=None, alignment_strategy="middle"):
  """
  Extract DTW alignment indices from a similarity matrix. No direction handling:
  matrix rows = source positions, columns = target positions.
  Returns (B, T1) with alignment_indices[b, i] = target column index j.

  Args:
      sim_matrix: torch.Tensor, shape (B, T1, T2) — similarity matrix (pre-softmax)
      temperature: float — scale used when converting sim to distance
      window: int or None — DTW band window
      alignment_strategy: str — "first", "middle", or "last" when multiple j map to same i

  Returns:
      alignment_indices: torch.Tensor, shape (B, T1), dtype long
  """
  batch_size, T1, T2 = sim_matrix.shape
  device = sim_matrix.device

  # sim = - (dist^2) / temp  =>  dist = sqrt(max(-sim * temp, 0))
  dist_matrices = torch.sqrt(torch.clamp(-sim_matrix * temperature, min=0.0)).detach().cpu().float().numpy().astype(np.double)
  alignment_indices = torch.zeros(batch_size, T1, dtype=torch.long, device=device)

  for b in range(batch_size):
      dist_matrix = dist_matrices[b].copy()
      r, c = T1, T2
      accumulated_cost = np.full((r + 1, c + 1), np.inf)
      accumulated_cost[0, 0] = 0

      for i in range(r):
          j_start = max(0, i - window) if window is not None else 0
          j_end = min(c, i + window + 1) if window is not None else c
          for j in range(j_start, j_end):
              cost = dist_matrix[i, j]
              accumulated_cost[i + 1, j + 1] = cost + min(
                  accumulated_cost[i, j + 1],
                  accumulated_cost[i + 1, j],
                  accumulated_cost[i, j],
              )

      path = []
      i, j = r - 1, c - 1
      while i >= 0 and j >= 0:
          path.append((i, j))
          if i == 0 and j == 0:
              break
          choices = [
              accumulated_cost[i, j],
              accumulated_cost[i, j + 1],
              accumulated_cost[i + 1, j],
          ]
          best = np.argmin(choices)
          if best == 0:
              i, j = i - 1, j - 1
          elif best == 1:
              i -= 1
          else:
              j -= 1
      path.reverse()

      if alignment_strategy == "last":
          path_dict = {i: j for (i, j) in path}
      elif alignment_strategy == "middle":
          from collections import defaultdict
          path_multidict = defaultdict(list)
          for (i, j) in path:
              path_multidict[i].append(j)
          path_dict = {i: j_list[len(j_list) // 2] for i, j_list in path_multidict.items()}
      else:
          path_dict = {}
          for (i, j) in path:
              if i not in path_dict:
                  path_dict[i] = j

      for i in range(T1):
          j = path_dict.get(i, 0)
          alignment_indices[b, i] = j

  return alignment_indices


def compute_deterministic_alignment_loss_paired(embs_main,
                                                embs_ref,
                                                steps_main,
                                                steps_ref,
                                                seq_lens_main,
                                                seq_lens_ref,
                                                num_steps,
                                                batch_size,
                                                loss_type,
                                                similarity_type,
                                                temperature,
                                                label_smoothing,
                                                variance_lambda,
                                                huber_delta,
                                                normalize_indices,
                                                normalize_embeddings=False,
                                                tcc_regression_margin=0.0,
                                                causal_lambda=0.0,
                                                causal_margin=0.05,
                                                forward_variance_lambda=0.0,
                                                global_step=None,
                                                training=True):
  """Compute cycle-consistency loss for paired sequences (Main <-> Ref)."""
  if normalize_embeddings:
      embs_main = F.normalize(embs_main, p=2, dim=-1)
      embs_ref = F.normalize(embs_ref, p=2, dim=-1)

  direction_main = None
  direction_ref  = None

  # --- 1. Main -> Ref -> Main (Backward Cycle) ---
  logits_mr, labels_mr, sim_mr, var_loss_mr, stats_mr, raw_sim_mr, path_cost_mr = align_pair_of_sequences(
      embs_main, embs_ref, similarity_type, temperature,
      steps2=steps_ref, seq_lens2=seq_lens_ref,
      normalize_indices=normalize_indices,
      forward_variance_lambda=forward_variance_lambda)

  # --- 2. Ref -> Main -> Ref (Forward Cycle) ---
  logits_rm, labels_rm, sim_rm, var_loss_rm, stats_rm, raw_sim_rm, path_cost_rm = align_pair_of_sequences(
      embs_ref, embs_main, similarity_type, temperature,
      steps2=steps_main, seq_lens2=seq_lens_main,
      normalize_indices=normalize_indices,
      forward_variance_lambda=forward_variance_lambda)

  # --- DTW Indices Extraction (Compute ONCE and reuse for loss/logging) ---
  # Always compute DTW indices for logging, regardless of config
  dtw_sim_mr_input = raw_sim_mr
  dtw_sim_rm_input = raw_sim_rm

  raw_sim12_mr_path, raw_sim12_rm_path = _maybe_save_raw_sim12(
      raw_sim_mr,
      raw_sim_rm,
      global_step,
      training,
  )

  _use_smooth_as_indices = (
      CONFIG.ALIGNMENT.SMOOTH_DTW_AS_DTW_INDICES
      and CONFIG.ALIGNMENT.USE_SMOOTH_DTW
  )

  if _use_smooth_as_indices:
      # Fast path: reuse already-computed student beta matrices.
      # argmax(beta[b, i, :]) gives ~90% agreement with hard DTW at any gamma_s.
      # The remaining 10% is at genuinely ambiguous path positions (gap ≈ 0) where
      # both choices are valid DTW paths; no meaningful regression signal is lost.
      dtw_indices_mr = sim_mr.detach().argmax(dim=-1)                  # [B, T_main]
      dtw_indices_rm = sim_rm.detach().argmax(dim=-1)                      # [B, T_ref]
  else:
      dtw_indices_mr = extract_alignment_indices_from_sim_matrix_dtw(
          dtw_sim_mr_input,
          temperature,
          window=CONFIG.ALIGNMENT.DTW_WINDOW,
          alignment_strategy=CONFIG.ALIGNMENT.DTW_ALIGNMENT_STRATEGY,
      ).detach()  # CRITICAL: detach to avoid gradient backprop

      dtw_indices_rm = extract_alignment_indices_from_sim_matrix_dtw(
          dtw_sim_rm_input,
          temperature,
          window=CONFIG.ALIGNMENT.DTW_WINDOW,
          alignment_strategy=CONFIG.ALIGNMENT.DTW_ALIGNMENT_STRATEGY,
      ).detach()

  # Note: dtw_indices_mr and dtw_indices_rm are in the range of real frames (0-indexed)

  # --- DTW Guidance Loss ---
  dtw_guidance_loss = torch.tensor(0.0, device=embs_main.device)
  dtw_guidance_lambda = CONFIG.ALIGNMENT.DTW_GUIDANCE_LAMBDA
  d2tw_loss = None  # initialised here; set below when USE_D2TW_LOSS=True

  if dtw_guidance_lambda > 0 and CONFIG.ALIGNMENT.USE_DTW:
      sim_mr_for_pred = sim_mr  # Already softmaxed
      steps_ref_for_dtw = steps_ref
      seq_lens_ref_for_dtw = seq_lens_ref

      # RM direction (Ref->Main)
      sim_rm_for_pred = sim_rm  # Already softmaxed

      # --- MR direction DTW loss ---
      # Normalize steps if needed
      if normalize_indices:
          steps_ref_norm = steps_ref_for_dtw.float() / (seq_lens_ref_for_dtw.float().unsqueeze(1) + 1e-8)
      else:
          steps_ref_norm = steps_ref_for_dtw.float()

      # pred_time: weighted sum of steps using softmax weights
      # sim_mr_for_pred: (B, T_main, T_ref)
      # steps_ref_norm: (B, T_ref)
      pred_time_mr = torch.sum(sim_mr_for_pred * steps_ref_norm.unsqueeze(1), dim=-1)  # (B, T_main)

      # true_time: use DTW indices to select steps
      # dtw_indices_mr: (B, T_main) - each main frame's corresponding ref frame index
      batch_size_local = dtw_indices_mr.shape[0]
      T_main = dtw_indices_mr.shape[1]

      # Gather true_time from steps using DTW indices
      batch_indices = torch.arange(batch_size_local, device=dtw_indices_mr.device).unsqueeze(1).expand(-1, T_main)
      true_time_mr = steps_ref_norm[batch_indices, dtw_indices_mr]  # (B, T_main)

      # MR loss: plain squared error (no precision weighting, no variance reg)
      squared_error_mr = torch.square(true_time_mr - pred_time_mr)  # (B, T_main)
      loss_dtw_mr = torch.mean(squared_error_mr)
      dtw_per_sample_mr = squared_error_mr.mean(dim=1)  # (B,)
      check_nan(loss_dtw_mr, "loss_dtw_mr", "DTW guidance MR")

      # --- RM direction DTW loss ---
      if normalize_indices:
          steps_main_norm = steps_main.float() / (seq_lens_main.float().unsqueeze(1) + 1e-8)
      else:
          steps_main_norm = steps_main.float()

      pred_time_rm = torch.sum(sim_rm_for_pred * steps_main_norm.unsqueeze(1), dim=-1)  # (B, T_ref)

      T_ref = dtw_indices_rm.shape[1]
      batch_indices_rm = torch.arange(batch_size_local, device=dtw_indices_rm.device).unsqueeze(1).expand(-1, T_ref)
      true_time_rm = steps_main_norm[batch_indices_rm, dtw_indices_rm]  # (B, T_ref)

      # RM loss: plain squared error (no precision weighting, no variance reg)
      squared_error_rm = torch.square(true_time_rm - pred_time_rm)  # (B, T_ref)
      loss_dtw_rm = torch.mean(squared_error_rm)
      dtw_per_sample_rm = squared_error_rm.mean(dim=1)  # (B,)
      check_nan(loss_dtw_rm, "loss_dtw_rm", "DTW guidance RM")

      # Average loss from both directions
      dtw_guidance_loss = (loss_dtw_mr + loss_dtw_rm) / 2.0
      dtw_guidance_loss_per_sample = (dtw_per_sample_mr + dtw_per_sample_rm) / 2.0  # (B,)

  # --- Loss Aggregation ---

  # Helper to compute loss component with optional scaling
  def compute_component_loss(logits, labels, steps_tgt, seq_lens_tgt, scale=None):
      b, n, _ = logits.shape
      logits_flat = logits.view(-1, n)
      labels_flat = labels.view(-1, n)

      # Steps prep
      steps_tiled = steps_tgt.unsqueeze(1).repeat(1, n, 1).view(-1, n)
      seq_lens_tiled = seq_lens_tgt.unsqueeze(1).repeat(1, n).view(-1)

      p_loss = None
      if loss_type == 'classification':
         p_loss = classification_loss(logits_flat, labels_flat, label_smoothing, reduction='none')
      elif 'regression' in loss_type:
         p_loss, _, _ = regression_loss(logits_flat, labels_flat, n, steps_tiled, seq_lens_tiled,
                           loss_type, normalize_indices, variance_lambda,
                           huber_delta, reduction='none',
                           tcc_regression_margin=tcc_regression_margin)
      elif loss_type == 'both':
         p_loss_c = classification_loss(logits_flat, labels_flat, label_smoothing, reduction='none')
         p_loss_r, _, _ = regression_loss(logits_flat, labels_flat, n, steps_tiled, seq_lens_tiled,
                           'regression_mse_var', normalize_indices, variance_lambda,
                           huber_delta, reduction='none',
                           tcc_regression_margin=tcc_regression_margin)
         p_loss = p_loss_c + p_loss_r
      else:
         return torch.tensor(0.0), None

      if p_loss is not None:
          check_nan(p_loss, f"p_loss (type={loss_type})", "compute_component_loss")

      if scale is not None:
             # Scale applies to Source steps.
             # Scale shape (B, N). p_loss shape (B*N).
             p_loss = p_loss * scale.view(-1)

      # Reshape back to (B, N) to get per-sample loss
      p_loss_b_n = p_loss.view(b, n)
      per_sample = p_loss_b_n.mean(dim=1) # (B,)

      return torch.mean(p_loss), per_sample

  # Loss MR (Main -> Ref)
  # Target steps are "Main" (Cycle: Main->Ref->Main)
  loss_mr, loss_mr_per_sample = compute_component_loss(logits_mr, labels_mr, steps_main, seq_lens_main, scale=None)

  # Loss RM (Ref -> Main)
  # Source is Ref. Target steps are "Ref".
  loss_rm, loss_rm_per_sample = compute_component_loss(logits_rm, labels_rm, steps_ref, seq_lens_ref, scale=None)

  loss = (loss_mr + loss_rm) / 2.0

  if forward_variance_lambda > 0:
      loss += (var_loss_mr + var_loss_rm) / 2.0

  # Add DTW guidance loss if enabled
  if dtw_guidance_lambda > 0 and CONFIG.ALIGNMENT.USE_DTW:
      loss = loss + dtw_guidance_loss * dtw_guidance_lambda

  # Add D2TW path-cost loss if enabled (reuses the DP table already computed by smooth_dtw_probs)
  # sdtw[-1,-1] is the cumulative cost of the optimal alignment path; minimising it encourages
  # globally coherent alignment, complementing the local TCC cycle-consistency objective.
  if CONFIG.ALIGNMENT.USE_D2TW_LOSS:
      assert CONFIG.ALIGNMENT.USE_SMOOTH_DTW, (
          "USE_D2TW_LOSS=True requires USE_SMOOTH_DTW=True "
          "(the path cost is read from the same DP table)"
      )
      if path_cost_mr is not None and path_cost_rm is not None:
          if CONFIG.ALIGNMENT.D2TW_NORMALIZE_LENGTH_SUM:
              # Grid axes match sim_mr / sim_rm: (B, T_src, T_tgt) -> denom = T_src + T_tgt
              t1_mr, t2_mr = sim_mr.shape[-2], sim_mr.shape[-1]
              t1_rm, t2_rm = sim_rm.shape[-2], sim_rm.shape[-1]
              denom_mr = path_cost_mr.new_tensor(float(t1_mr + t2_mr) + 1e-8)
              denom_rm = path_cost_rm.new_tensor(float(t1_rm + t2_rm) + 1e-8)
              d2tw_cost_mr = path_cost_mr / denom_mr
              d2tw_cost_rm = path_cost_rm / denom_rm
          else:
              d2tw_cost_mr = path_cost_mr
              d2tw_cost_rm = path_cost_rm
          # Optional hinge: linear excess ReLU(cost - m) to stay consistent with original D2TW
          # (minimize path cost, not squared error on cost).
          d2tw_m = float(CONFIG.ALIGNMENT.D2TW_COST_MARGIN)
          if d2tw_m > 0.0:
              d2tw_mr_for_loss = F.relu(d2tw_cost_mr - d2tw_m)
              d2tw_rm_for_loss = F.relu(d2tw_cost_rm - d2tw_m)
          else:
              d2tw_mr_for_loss = d2tw_cost_mr
              d2tw_rm_for_loss = d2tw_cost_rm
          d2tw_loss = (d2tw_mr_for_loss.mean() + d2tw_rm_for_loss.mean()) / 2.0
          loss = loss + d2tw_loss * CONFIG.ALIGNMENT.D2TW_LOSS_LAMBDA

  # Aggregate Per Sample Loss for Logging (Main + Ref)
  ps_main = loss_mr_per_sample
  ps_ref = loss_rm_per_sample

  full_per_sample_loss = torch.cat([ps_main, ps_ref], dim=0) # (2*B)

  # Compute top-5 alignment candidates and their probabilities
  # sim_mr: (B, T_main, T_ref) or (T_main, T_ref)
  # sim_rm: (B, T_ref, T_main) or (T_ref, T_main)
  top5_k = min(5, sim_mr.shape[-1])  # Handle cases where ref has < 5 frames

  # Forward alignment (Main -> Ref): get top-5 ref frames for each main frame
  forward_top5_values, forward_top5_indices = torch.topk(sim_mr, k=top5_k, dim=-1, largest=True, sorted=True)

  # Backward alignment (Ref -> Main): get top-5 main frames for each ref frame
  backward_top5_values, backward_top5_indices = torch.topk(sim_rm, k=top5_k, dim=-1, largest=True, sorted=True)

  # Compute both argmax and DTW indices for logging and comparison
  # Argmax indices (always computed)
  forward_argmax_indices = torch.argmax(sim_mr, dim=-1)
  backward_argmax_indices = torch.argmax(sim_rm, dim=-1)

  # DTW indices (reuse the ones computed above)
  forward_dtw_indices = dtw_indices_mr
  backward_dtw_indices = dtw_indices_rm

  # Determine which indices to use as primary based on config
  if CONFIG.ALIGNMENT.USE_DTW:
      forward_indices = forward_dtw_indices
      backward_indices = backward_dtw_indices
  else:
      forward_indices = forward_argmax_indices
      backward_indices = backward_argmax_indices

  # Construct Loss Dict
  loss_dict = {
      'loss_mr': loss_mr,
      'loss_rm': loss_rm,
      'forward_alignment_indices': forward_indices,  # Main -> Ref (primary based on config)
      'backward_alignment_indices': backward_indices,  # Ref -> Main (primary based on config)
      'forward_argmax_indices': forward_argmax_indices,  # Main -> Ref (argmax)
      'backward_argmax_indices': backward_argmax_indices,  # Ref -> Main (argmax)
      'forward_dtw_indices': forward_dtw_indices,  # Main -> Ref (DTW)
      'backward_dtw_indices': backward_dtw_indices,  # Ref -> Main (DTW)
      'forward_top5_indices': forward_top5_indices,  # (B, T_main, 5) or (T_main, 5)
      'forward_top5_probs': forward_top5_values,     # (B, T_main, 5) or (T_main, 5)
      'backward_top5_indices': backward_top5_indices,  # (B, T_ref, 5) or (T_ref, 5)
      'backward_top5_probs': backward_top5_values,     # (B, T_ref, 5) or (T_ref, 5)
      'per_sample_loss': full_per_sample_loss,
      'mr_softmax_max_max': stats_mr['max'],
      'mr_softmax_max_min': stats_mr['min'],
      'mr_softmax_max_mean': stats_mr['mean'],
      'mr_raw_variance': stats_mr['var_mean'],
      'rm_softmax_max_max': stats_rm['max'],
      'rm_softmax_max_min': stats_rm['min'],
      'rm_softmax_max_mean': stats_rm['mean'],
      'rm_raw_variance': stats_rm['var_mean'],
      'forward_variance_loss': (var_loss_mr + var_loss_rm) / 2.0
  }

  if raw_sim12_mr_path:
    loss_dict['raw_sim12_mr_path'] = raw_sim12_mr_path
    loss_dict['raw_sim12_rm_path'] = raw_sim12_rm_path

  # Add DTW guidance loss to loss_dict if computed
  if dtw_guidance_lambda > 0 and CONFIG.ALIGNMENT.USE_DTW:
      loss_dict['dtw_guidance_loss'] = dtw_guidance_loss
      loss_dict['dtw_guidance_loss_mr'] = loss_dtw_mr
      loss_dict['dtw_guidance_loss_rm'] = loss_dtw_rm
      loss_dict['dtw_guidance_loss_per_sample'] = dtw_guidance_loss_per_sample  # (B,)

  # Add D2TW path-cost loss to loss_dict if computed
  if CONFIG.ALIGNMENT.USE_D2TW_LOSS and path_cost_mr is not None and path_cost_rm is not None:
      loss_dict['d2tw_loss'] = d2tw_loss.detach()
      # Means of the term actually optimized (after optional hinge).
      loss_dict['d2tw_loss_mr'] = d2tw_mr_for_loss.mean().detach()
      loss_dict['d2tw_loss_rm'] = d2tw_rm_for_loss.mean().detach()
      loss_dict['d2tw_norm_cost_mr_mean'] = d2tw_cost_mr.mean().detach()
      loss_dict['d2tw_norm_cost_rm_mean'] = d2tw_cost_rm.mean().detach()
      loss_dict['d2tw_path_cost_mr_raw_mean'] = path_cost_mr.mean().detach()
      loss_dict['d2tw_path_cost_rm_raw_mean'] = path_cost_rm.mean().detach()

  # --- Causal Loss ---
  if causal_lambda > 0:
    if normalize_indices:
        float_seq_lens_main = seq_lens_main.float().unsqueeze(1)
        float_seq_lens_ref = seq_lens_ref.float().unsqueeze(1)
        norm_steps_main = steps_main.float() / float_seq_lens_main
        norm_steps_ref = steps_ref.float() / float_seq_lens_ref
    else:
        norm_steps_main = steps_main.float()
        norm_steps_ref = steps_ref.float()

    weights_mr = sim_mr # Already softmaxed and gated
    weights_rm = sim_rm # Already softmaxed and gated

    loss_causal_mr = causal_loss(weights_mr, norm_steps_main, norm_steps_ref,
                                 margin=causal_margin, direction=direction_main)
    loss_causal_rm = causal_loss(weights_rm, norm_steps_ref, norm_steps_main,
                                 margin=causal_margin, direction=direction_ref)
    causal_loss_val = (loss_causal_mr + loss_causal_rm) / 2.0

    loss = loss + causal_loss_val * causal_lambda
    loss_dict['causal_loss'] = causal_loss_val

  return loss, loss_dict
