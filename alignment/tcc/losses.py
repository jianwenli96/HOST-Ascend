# coding=utf-8
"""Loss functions imposing the cycle-consistency constraints."""

import torch
import torch.nn.functional as F
from utils import check_nan


def hinge_squared_abs_error(abs_error, margin):
  """Squared hinge on absolute error: ReLU(|e| - margin)^2. margin=0 recovers e^2."""
  return torch.square(F.relu(abs_error - margin))


def classification_loss(logits, labels, label_smoothing, reduction='mean'):
  """Loss function based on classifying the correct indices.

  Args:
    logits: Tensor, Pre-softmax scores used for classification loss.
    labels: Tensor, One hot labels containing the ground truth.
    label_smoothing: Float, label smoothing factor.
    reduction: String, 'mean', 'sum' or 'none'.
  Returns:
    loss: Tensor, A scalar classification loss.
  """
  labels = labels.detach()
  
  check_nan(logits, "logits", "classification_loss input")

  log_probs = F.log_softmax(logits, dim=-1)
  check_nan(log_probs, "log_probs", "classification_loss input")
  
  num_classes = logits.size(-1)
  smoothed_labels = labels * (1.0 - label_smoothing) + label_smoothing / num_classes
  
  loss = -torch.sum(smoothed_labels * log_probs, dim=-1)
  
  if reduction == 'mean':
      return torch.mean(loss)
  elif reduction == 'sum':
      return torch.sum(loss)
  else:
      return loss


def regression_loss(logits, labels, num_steps, steps, seq_lens, loss_type,
                    normalize_indices, variance_lambda, huber_delta, reduction='mean',
                    tcc_regression_margin=0.0):
  """Loss function based on regressing to the correct indices.

  Args:
    logits: Tensor, Pre-softmax similarity scores.
    labels: Tensor, One hot labels containing the ground truth.
    num_steps: Integer, Number of steps in the sequence embeddings.
    steps: Tensor, step indices/frame indices of the embeddings.
    seq_lens: Tensor, Lengths of the sequences.
    loss_type: String, This specifies the kind of regression loss function.
    normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
    variance_lambda: Float, Weight of the variance of the similarity predictions.
    huber_delta: float, Huber delta.
    reduction: String, 'mean', 'sum' or 'none'.
    tcc_regression_margin: Float, For regression_mse_var only: hinge on absolute time error
        (ReLU(|e|-m)^2 replaces squared_error in precision-weighted term). Variance term unchanged.
        0.0 preserves legacy behavior.

  Returns:
     loss: Tensor, A scalar loss (unless reduction='none').
     regression_loss_val: Tensor or None
     variance_loss_val: Tensor or None
  """
  labels = labels.detach()
  steps = steps.detach()

  if normalize_indices:
    float_seq_lens = seq_lens.float()
    check_nan(float_seq_lens, "float_seq_lens", "regression_loss")
    tile_seq_lens = float_seq_lens.unsqueeze(1).repeat(1, num_steps)
    # Add epsilon to prevent division by zero
    steps = steps.float() / (tile_seq_lens + 1e-8)
  else:
    steps = steps.float()
  
  check_nan(steps, "steps", "regression_loss input")
  check_nan(logits, "logits", "regression_loss input")

  beta = F.softmax(logits, dim=-1)
  check_nan(beta, "beta", "regression_loss")
  
  true_time = torch.sum(steps * labels, dim=1)
  check_nan(true_time, "true_time", "regression_loss")
  
  pred_time = torch.sum(steps * beta, dim=1)
  check_nan(pred_time, "pred_time", "regression_loss")

  if loss_type in ['regression_mse', 'regression_mse_var']:
    if 'var' in loss_type:
      # Variance aware regression.
      pred_time_tiled = pred_time.unsqueeze(1).repeat(1, num_steps)

      pred_time_variance = torch.sum(
          torch.square(steps - pred_time_tiled) * beta, dim=1)
      check_nan(pred_time_variance, "pred_time_variance", "regression_loss")

      # Using log of variance as it is numerically stabler.
      # Increased epsilon from 1e-10 to 1e-4.
      pred_time_log_var = torch.log(pred_time_variance + 1e-4)
      check_nan(pred_time_log_var, "pred_time_log_var", "regression_loss")
      
      abs_err = (true_time - pred_time).abs()
      check_nan(abs_err, "abs_err", "regression_loss")

      precision = torch.exp(-pred_time_log_var)
      check_nan(precision, "precision (exp(-log_var))", "regression_loss")

      # Main regression term: ReLU(|e|-m)^2 (margin=0 => e^2, same as legacy squared_error).
      main_err_sq = hinge_squared_abs_error(abs_err, tcc_regression_margin)
      check_nan(main_err_sq, "main_err_sq", "regression_loss")
      regression_loss_val = precision * main_err_sq
      variance_loss_val = variance_lambda * pred_time_log_var
      
      check_nan(regression_loss_val, "regression_loss_val", "regression_loss")
      check_nan(variance_loss_val, "variance_loss_val", "regression_loss")

      loss = regression_loss_val + variance_loss_val
      check_nan(loss, "loss (internal)", "regression_loss")
      
      if reduction == 'mean':
          return torch.mean(loss), torch.mean(regression_loss_val), torch.mean(variance_loss_val)
      elif reduction == 'sum':
          return torch.sum(loss), torch.sum(regression_loss_val), torch.sum(variance_loss_val)
      else:
          return loss, regression_loss_val, variance_loss_val

    else:
      loss = F.mse_loss(pred_time, true_time, reduction=reduction)
      return loss, None, None
  elif loss_type == 'regression_huber':
    loss = F.huber_loss(pred_time, true_time, delta=huber_delta, reduction=reduction)
    return loss, None, None
  else:
    raise ValueError('Unsupported regression loss %s.' % loss_type)

def causal_loss(weights, steps_source, steps_target, margin=0.1, reduction='sum',
                direction=None):
  """Computes a differentiable causal (monotonicity) constraint.

  Args:
    weights: Tensor, shape (B, T, T), alignment weights (Softmaxed similarity).
    steps_source: Tensor, shape (B, T), physical timestamps of source video.
    steps_target: Tensor, shape (B, T), physical timestamps of target video.
    margin: Float, small margin to prevent collapse.
    reduction: String, 'mean', 'sum' or 'none'.
    direction: Optional Tensor, shape (B,) or (B, 1), values +1 / -1. When
               provided (derived from explicit do_reverse flags) it is used
               directly; otherwise direction is inferred from steps_source.
  """
  # Expectation of target timestamp for each source frame
  # weights: (B, T, T), steps_target: (B, T)
  # result: (B, T)
  pred_time = torch.sum(weights * steps_target.unsqueeze(1), dim=-1)
  check_nan(pred_time, "pred_time", "causal_loss")

  # Temporal direction: 1 for forward, -1 for reverse.
  # Use ground-truth flag when available, else infer from step values.
  if direction is not None:
      direction = direction.float().view(-1, 1).to(weights.device)  # (B, 1)
  else:
      direction = torch.sign(steps_source[:, -1] - steps_source[:, 0]).unsqueeze(1)  # (B, 1)
  
  T = pred_time.shape[1]
  stride = max(1, int(T / 24))

  # Progress in aligned target time
  # diff: (B, T-stride)
  diff = pred_time[:, stride:] - pred_time[:, :-stride]
  check_nan(diff, "diff", "causal_loss")

  # Penalty: ReLU(margin - direction * diff)^2
  # If direction=1, we want diff > margin.
  # If direction=-1, we want diff < -margin (i.e., -diff > margin).
  penalty = torch.square(F.relu(margin - direction * diff))
  check_nan(penalty, "penalty", "causal_loss")
  
  if reduction == 'mean':
      return torch.mean(penalty)
  elif reduction == 'sum':
      return torch.sum(penalty)
  else:
      return penalty