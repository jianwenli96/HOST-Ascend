# coding=utf-8
"""Variants of the cycle-consistency loss described in TCC paper."""

import torch
from .deterministic_alignment import compute_deterministic_alignment_loss, compute_deterministic_alignment_loss_paired
from .stochastic_alignment import compute_stochastic_alignment_loss


def compute_alignment_loss(embs,
                           batch_size,
                           steps=None,
                           seq_lens=None,
                           stochastic_matching=False,
                           normalize_embeddings=False,
                           loss_type='classification',
                           similarity_type='l2',
                           num_cycles=20,
                           cycle_length=2,
                           temperature=0.1,
                           label_smoothing=0.1,
                           variance_lambda=0.001,
                           huber_delta=0.1,
                           tcc_regression_margin=0.0,
                           normalize_indices=True,
                           paired_matching=False,
                           metadata=None,
                           causal_lambda=0.0,
                           causal_margin=0.05,
                           forward_variance_lambda=0.0,
                           raw_tokens=None,
                           gate_module=None,
                           precomputed_gates=None,
                           global_step=None,
                           training=True):
  """Computes alignment loss between sequences of embeddings."""
  
  num_steps = embs.size(1)
  
  if paired_matching:
      # The embeddings are concatenated in Algorithm.train_one_iter as [Main, Ref].
      # embs structure: [Main_0, ..., Main_B-1, Ref_0, ..., Ref_B-1]
      # batch_size passed here is the full batch size (2B)
      real_batch_size = batch_size // 2
      
      embs_main = embs[:real_batch_size]
      embs_ref = embs[real_batch_size:]
      
      steps_main = steps[:real_batch_size]
      steps_ref = steps[real_batch_size:]
      
      seq_lens_main = seq_lens[:real_batch_size]
      seq_lens_ref = seq_lens[real_batch_size:]

      raw_tokens_main = None
      raw_tokens_ref = None
      if raw_tokens is not None:
          raw_tokens_main = raw_tokens['mains_tokens']
          raw_tokens_ref = raw_tokens['refs_tokens']

      # --- REDUNDANT MERGING REMOVED: Rely on Algorithm.forward() for merging ---
      # In ZeRO-3, merging must happen in forward().
      loss, loss_dict = compute_deterministic_alignment_loss_paired(
          embs_main, embs_ref, steps_main, steps_ref, seq_lens_main, seq_lens_ref,
          num_steps, real_batch_size, loss_type, similarity_type, temperature,
          label_smoothing, variance_lambda, huber_delta, normalize_indices,
          normalize_embeddings=normalize_embeddings,
          tcc_regression_margin=tcc_regression_margin,
          causal_lambda=causal_lambda, causal_margin=causal_margin,
          forward_variance_lambda=forward_variance_lambda,
          raw_tokens_main=raw_tokens_main,
          raw_tokens_ref=raw_tokens_ref,
          gate_module=gate_module,
          precomputed_gates=precomputed_gates,
          global_step=global_step,
          training=training)
      
      return loss, loss_dict

  if stochastic_matching:
      return compute_stochastic_alignment_loss(
          embs, steps, seq_lens, num_steps, batch_size, loss_type,
          similarity_type, temperature, label_smoothing, variance_lambda,
          huber_delta, normalize_indices, num_cycles, cycle_length,
          normalize_embeddings=normalize_embeddings,
          tcc_regression_margin=tcc_regression_margin)
  else:
      return compute_deterministic_alignment_loss(
          embs, steps, seq_lens, num_steps, batch_size, loss_type,
          similarity_type, temperature, label_smoothing, variance_lambda,
          huber_delta, normalize_indices,
          normalize_embeddings=normalize_embeddings,
          forward_variance_lambda=forward_variance_lambda,
          tcc_regression_margin=tcc_regression_margin)
