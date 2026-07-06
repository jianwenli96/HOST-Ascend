# coding=utf-8
"""Deterministic alignment between all pairs of sequences in a batch."""

import torch
import torch.nn.functional as F
from config import CONFIG
from .losses import regression_loss
from .smooth_dtw import smooth_dtw_probs


def _compute_variance_loss(beta, steps, seq_lens, normalize_indices, variance_lambda):
  """Computes variance loss for a given similarity matrix."""
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

  if beta.dim() == 3:
    b, t1, t2 = beta.shape
    # cur_steps: (B, T2) -> (B, 1, T2)
    cur_steps_tiled = cur_steps.unsqueeze(1).repeat(1, t1, 1)
    pred_time = torch.sum(cur_steps_tiled * beta, dim=-1, keepdim=True) # (B, T1, 1)
    variance = torch.sum(torch.square(cur_steps_tiled - pred_time) * beta, dim=-1) # (B, T1)
  else:
    t1, t2 = beta.shape
    # cur_steps: (T2,)
    # beta: (T1, T2)
    pred_time = torch.sum(cur_steps * beta, dim=-1, keepdim=True) # (T1, 1)
    variance = torch.sum(torch.square(cur_steps - pred_time) * beta, dim=-1) # (T1)

  var_mean = variance.mean()

  # Increase epsilon for numerical stability in gradients (especially for BF16)
  variance_eps = 1e-5
  log_var = torch.log(variance + variance_eps)
  per_step_loss = variance_lambda * log_var
  return torch.mean(per_step_loss), per_step_loss, var_mean


def pairwise_l2_distance(embs1, embs2):
  """Computes pairwise distances (squared) between all rows of embs1 and embs2.
     Uses torch.cdist for numerical stability in BF16/FP16.
  """
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

  return (dist ** 2).to(dtype=embs1.dtype)


def get_scaled_similarity(embs1, embs2, similarity_type, temperature):
  """Returns similarity between each all rows of embs1 and all rows of embs2."""
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

  # Compute beta: matching probability for each source frame over target frames,
  # derived from soft-DTW cumulative cost (temporal monotonicity bias).
  _want_cost = CONFIG.ALIGNMENT.USE_D2TW_LOSS
  path_cost_12 = None  # (B,) soft-DTW terminal cost; None when D2TW loss is disabled
  # smooth_dtw_probs requires [B, T1, T2]; handle the 2D case (single pair).
  if sim_12.dim() == 2:
      _result = smooth_dtw_probs(
          sim_12.unsqueeze(0),
          gamma_s=CONFIG.ALIGNMENT.SMOOTH_DTW_GAMMA_S,
          gamma_f=CONFIG.ALIGNMENT.SMOOTH_DTW_GAMMA_F,
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
          bidirectional=CONFIG.ALIGNMENT.SMOOTH_DTW_BIDIRECTIONAL,
          method=CONFIG.ALIGNMENT.SMOOTH_DTW_METHOD,
          return_cost=_want_cost,
      )
      if _want_cost:
          softmaxed_sim_12, path_cost_12 = _result[0], _result[1]   # [B,T1,T2], (B,)
      else:
          softmaxed_sim_12 = _result                # [B, T1, T2]

  forward_var_loss, _, var_mean = _compute_variance_loss(
      softmaxed_sim_12, steps2, seq_lens2, normalize_indices, forward_variance_lambda)

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

  # Find distances between nn_embs and embs1.
  sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)

  logits = sim_21
  if embs1.dim() == 2:
      labels = torch.eye(max_num_steps, device=embs1.device)
  else:
      batch_size = embs1.size(0)
      labels = torch.eye(max_num_steps, device=embs1.device).unsqueeze(0).repeat(batch_size, 1, 1)

  # path_cost_12: (B,) soft-DTW terminal cost, or None when USE_D2TW_LOSS=False
  return logits, labels, softmaxed_sim_12, forward_var_loss, stats, path_cost_12


def compute_deterministic_alignment_loss_paired(embs_main,
                                                embs_ref,
                                                steps_main,
                                                steps_ref,
                                                seq_lens_main,
                                                seq_lens_ref,
                                                num_steps,
                                                batch_size,
                                                similarity_type,
                                                temperature,
                                                variance_lambda,
                                                normalize_indices,
                                                normalize_embeddings=False,
                                                tcc_regression_margin=0.0,
                                                forward_variance_lambda=0.0):
  """Compute cycle-consistency loss for paired sequences (Main <-> Ref)."""
  if normalize_embeddings:
      embs_main = F.normalize(embs_main, p=2, dim=-1)
      embs_ref = F.normalize(embs_ref, p=2, dim=-1)

  # --- 1. Main -> Ref -> Main (Backward Cycle) ---
  logits_mr, labels_mr, sim_mr, var_loss_mr, stats_mr, path_cost_mr = align_pair_of_sequences(
      embs_main, embs_ref, similarity_type, temperature,
      steps2=steps_ref, seq_lens2=seq_lens_ref,
      normalize_indices=normalize_indices,
      forward_variance_lambda=forward_variance_lambda)

  # --- 2. Ref -> Main -> Ref (Forward Cycle) ---
  logits_rm, labels_rm, sim_rm, var_loss_rm, stats_rm, path_cost_rm = align_pair_of_sequences(
      embs_ref, embs_main, similarity_type, temperature,
      steps2=steps_main, seq_lens2=seq_lens_main,
      normalize_indices=normalize_indices,
      forward_variance_lambda=forward_variance_lambda)

  # --- DTW Indices Extraction (Compute ONCE and reuse for loss/logging) ---
  # Always compute DTW indices for logging, regardless of config.
  # Reuse already-computed student beta matrices: argmax(beta[b, i, :]) gives ~90%
  # agreement with hard DTW at any gamma_s. The remaining 10% is at genuinely
  # ambiguous path positions (gap ≈ 0) where both choices are valid DTW paths; no
  # meaningful regression signal is lost.
  d2tw_loss = None  # initialised here; set below when USE_D2TW_LOSS=True

  dtw_indices_mr = sim_mr.detach().argmax(dim=-1)                  # [B, T_main]
  dtw_indices_rm = sim_rm.detach().argmax(dim=-1)                      # [B, T_ref]

  # Note: dtw_indices_mr and dtw_indices_rm are in the range of real frames (0-indexed)

  # --- Loss Aggregation ---

  # Helper to compute loss component with optional scaling
  def compute_component_loss(logits, labels, steps_tgt, seq_lens_tgt, scale=None):
      b, n, _ = logits.shape
      logits_flat = logits.view(-1, n)
      labels_flat = labels.view(-1, n)

      # Steps prep
      steps_tiled = steps_tgt.unsqueeze(1).repeat(1, n, 1).view(-1, n)
      seq_lens_tiled = seq_lens_tgt.unsqueeze(1).repeat(1, n).view(-1)

      p_loss, _, _ = regression_loss(logits_flat, labels_flat, n, steps_tiled, seq_lens_tiled,
                        normalize_indices, variance_lambda, reduction='none',
                        tcc_regression_margin=tcc_regression_margin)

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

  # Add D2TW path-cost loss if enabled (reuses the DP table already computed by smooth_dtw_probs)
  # sdtw[-1,-1] is the cumulative cost of the optimal alignment path; minimising it encourages
  # globally coherent alignment, complementing the local TCC cycle-consistency objective.
  if CONFIG.ALIGNMENT.USE_D2TW_LOSS:
      if path_cost_mr is not None and path_cost_rm is not None:
          # Grid axes match sim_mr / sim_rm: (B, T_src, T_tgt) -> denom = T_src + T_tgt
          t1_mr, t2_mr = sim_mr.shape[-2], sim_mr.shape[-1]
          t1_rm, t2_rm = sim_rm.shape[-2], sim_rm.shape[-1]
          denom_mr = path_cost_mr.new_tensor(float(t1_mr + t2_mr) + 1e-8)
          denom_rm = path_cost_rm.new_tensor(float(t1_rm + t2_rm) + 1e-8)
          d2tw_cost_mr = path_cost_mr / denom_mr
          d2tw_cost_rm = path_cost_rm / denom_rm
          # Hinge: linear excess ReLU(cost - m) to stay consistent with original D2TW
          # (minimize path cost, not squared error on cost). m=0 reduces to the raw cost
          # since d2tw_cost is always >= 0.
          d2tw_m = float(CONFIG.ALIGNMENT.D2TW_COST_MARGIN)
          d2tw_mr_for_loss = F.relu(d2tw_cost_mr - d2tw_m)
          d2tw_rm_for_loss = F.relu(d2tw_cost_rm - d2tw_m)
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

  forward_indices = forward_dtw_indices
  backward_indices = backward_dtw_indices

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

  return loss, loss_dict
