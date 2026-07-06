# coding=utf-8
"""Loss functions imposing the cycle-consistency constraints."""

import torch
import torch.nn.functional as F


def hinge_squared_abs_error(abs_error, margin):
  """Squared hinge on absolute error: ReLU(|e| - margin)^2. margin=0 recovers e^2."""
  return torch.square(F.relu(abs_error - margin))


def regression_loss(logits, labels, num_steps, steps, seq_lens,
                    normalize_indices, variance_lambda, reduction='mean',
                    tcc_regression_margin=0.0):
  """Loss function based on regressing to the correct indices (variance-aware MSE).

  Args:
    logits: Tensor, Pre-softmax similarity scores.
    labels: Tensor, One hot labels containing the ground truth.
    num_steps: Integer, Number of steps in the sequence embeddings.
    steps: Tensor, step indices/frame indices of the embeddings.
    seq_lens: Tensor, Lengths of the sequences.
    normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
    variance_lambda: Float, Weight of the variance of the similarity predictions.
    reduction: String, 'mean', 'sum' or 'none'.
    tcc_regression_margin: Float, hinge on absolute time error
        (ReLU(|e|-m)^2 replaces squared_error in precision-weighted term).
        0.0 preserves legacy behavior.

  Returns:
     loss: Tensor, A scalar loss (unless reduction='none').
     regression_loss_val: Tensor
     variance_loss_val: Tensor
  """
  labels = labels.detach()
  steps = steps.detach()

  if normalize_indices:
    float_seq_lens = seq_lens.float()
    tile_seq_lens = float_seq_lens.unsqueeze(1).repeat(1, num_steps)
    # Add epsilon to prevent division by zero
    steps = steps.float() / (tile_seq_lens + 1e-8)
  else:
    steps = steps.float()

  beta = F.softmax(logits, dim=-1)

  true_time = torch.sum(steps * labels, dim=1)
  pred_time = torch.sum(steps * beta, dim=1)

  # Variance aware regression.
  pred_time_tiled = pred_time.unsqueeze(1).repeat(1, num_steps)

  pred_time_variance = torch.sum(
      torch.square(steps - pred_time_tiled) * beta, dim=1)

  # Using log of variance as it is numerically stabler.
  # Increased epsilon from 1e-10 to 1e-4.
  pred_time_log_var = torch.log(pred_time_variance + 1e-4)

  abs_err = (true_time - pred_time).abs()

  precision = torch.exp(-pred_time_log_var)

  # Main regression term: ReLU(|e|-m)^2 (margin=0 => e^2, same as legacy squared_error).
  main_err_sq = hinge_squared_abs_error(abs_err, tcc_regression_margin)
  regression_loss_val = precision * main_err_sq
  variance_loss_val = variance_lambda * pred_time_log_var

  loss = regression_loss_val + variance_loss_val

  if reduction == 'mean':
      return torch.mean(loss), torch.mean(regression_loss_val), torch.mean(variance_loss_val)
  elif reduction == 'sum':
      return torch.sum(loss), torch.sum(regression_loss_val), torch.sum(variance_loss_val)
  else:
      return loss, regression_loss_val, variance_loss_val
