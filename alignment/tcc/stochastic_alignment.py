# coding=utf-8
"""Stochastic alignment between sampled cycles in the sequences in a batch."""

import torch
import torch.nn.functional as F
from config import CONFIG
from .losses import classification_loss, regression_loss


def _align_single_cycle(cycle, embs, cycle_length, num_steps,
                        similarity_type, temperature):
  """Takes a single cycle and returns logits (simialrity scores) and labels."""
  # Choose random frame.
  n_idx = torch.randint(0, num_steps, (1,), device=embs.device).item()
  # Create labels
  onehot_labels = F.one_hot(torch.tensor([n_idx], device=embs.device), num_classes=num_steps).float()

  # Choose query feats for first frame.
  query_feats = embs[cycle[0], n_idx:n_idx+1]

  num_channels = float(query_feats.size(-1))
  for c in range(1, cycle_length+1):
    candidate_feats = embs[cycle[c]]

    if similarity_type == 'l2':
      # Find L2 distance.
      diff = query_feats.repeat(num_steps, 1) - candidate_feats
      mean_squared_distance = torch.sum(diff ** 2, dim=1)
      # Convert L2 distance to similarity.
      similarity = -mean_squared_distance

    elif similarity_type == 'cosine':
      # Dot product of embeddings.
      similarity = torch.matmul(candidate_feats, query_feats.t()).squeeze(1)
    else:
      raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale the distance  by number of channels. This normalization helps with
    # optimization.
    if not CONFIG.ALIGNMENT.NORMALIZE_EMBEDDINGS:
      similarity /= num_channels
    # Scale the distance by a temperature that helps with how soft/hard the
    # alignment should be.
    similarity /= temperature

    beta = F.softmax(similarity, dim=0)
    beta = beta.unsqueeze(1)

    # Find weighted nearest neighbour.
    query_feats = torch.sum(beta * candidate_feats,
                                axis=0, keepdims=True)

  return similarity, onehot_labels


def _align(cycles, embs, num_steps, num_cycles, cycle_length,
           similarity_type, temperature):
  """Align by finding cycles in embs."""
  logits_list = []
  labels_list = []
  for i in range(num_cycles):
    logits, labels = _align_single_cycle(cycles[i],
                                         embs,
                                         cycle_length,
                                         num_steps,
                                         similarity_type,
                                         temperature)
    logits_list.append(logits)
    labels_list.append(labels)

  logits = torch.stack(logits_list)
  labels = torch.stack(labels_list).squeeze(1)

  return logits, labels


def gen_cycles(num_cycles, batch_size, cycle_length=2):
  """Generates cycles for alignment."""
  cycles = []
  for _ in range(num_cycles):
      perm = torch.randperm(batch_size)
      cycle = perm[:cycle_length]
      cycle = torch.cat([cycle, cycle[0:1]])
      cycles.append(cycle)
  return torch.stack(cycles)


def compute_stochastic_alignment_loss(embs,
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
                                      num_cycles,
                                      cycle_length,
                                      normalize_embeddings=False,
                                      tcc_regression_margin=0.0):
  """Computes cycle-consistency loss for a set of random cycles."""
  if normalize_embeddings:
    embs = F.normalize(embs, p=2, dim=-1)

  cycles = gen_cycles(num_cycles, batch_size, cycle_length).to(embs.device)

  logits, labels = _align(cycles, embs, num_steps, num_cycles, cycle_length,
                          similarity_type, temperature)
  
  start_indices = cycles[:, 0]
  cycle_steps = steps[start_indices]
  cycle_seq_lens = seq_lens[start_indices]

  loss_dict = {}
  if loss_type == 'classification':
    loss = classification_loss(logits, labels, label_smoothing)
  elif 'regression' in loss_type:
    loss, reg_loss, var_loss = regression_loss(logits, labels, num_steps, cycle_steps, cycle_seq_lens,
                           loss_type, normalize_indices, variance_lambda,
                           huber_delta, tcc_regression_margin=tcc_regression_margin)
    if reg_loss is not None:
        loss_dict['regression_loss'] = reg_loss
    if var_loss is not None:
        loss_dict['variance_loss'] = var_loss
  elif loss_type == 'both':
    classif_loss = classification_loss(logits, labels, label_smoothing)
    reg_loss_combined, reg_loss, var_loss = regression_loss(logits, labels, num_steps, cycle_steps, cycle_seq_lens,
                           'regression_mse_var', normalize_indices, variance_lambda,
                           huber_delta, tcc_regression_margin=tcc_regression_margin)
    loss = classif_loss + reg_loss_combined
    loss_dict['classification_loss'] = classif_loss
    loss_dict['regression_loss'] = reg_loss
    loss_dict['variance_loss'] = var_loss
  else:
    raise ValueError('Unidentified loss_type %s.' % loss_type)

  return loss, loss_dict
