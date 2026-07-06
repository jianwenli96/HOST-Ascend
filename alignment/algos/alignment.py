# coding=utf-8
"""Cycle consistency loss for unsupervised training."""

from algos.algorithm import Algorithm
from config import CONFIG
from tcc.alignment import compute_alignment_loss
import torch

class Alignment(Algorithm):
  """Uses cycle-consistency loss to perform unsupervised training."""

  def compute_loss(self, embs, steps, seq_lens, global_step, training,
                   frame_labels, seq_labels, metadata=None):

    merged_meta = None
    if isinstance(embs, dict):
        if 'steps' in embs:
            steps = embs['steps']
        if 'seq_lens' in embs:
            seq_lens = embs['seq_lens']
        merged_meta = embs.get('merged_metadata')
        embs = embs['embs']

    if training:
      # Use actual batch size from embeddings to handle potential concatenation (e.g. ref frames)
      batch_size = embs.size(0)
      num_steps = CONFIG.TRAIN.NUM_FRAMES
    else:
      batch_size = embs.size(0)
      num_steps = CONFIG.EVAL.NUM_FRAMES

    loss, loss_dict = compute_alignment_loss(
        embs,
        batch_size,
        steps=steps,
        seq_lens=seq_lens,
        stochastic_matching=CONFIG.ALIGNMENT.STOCHASTIC_MATCHING,
        normalize_embeddings=CONFIG.ALIGNMENT.NORMALIZE_EMBEDDINGS,
        loss_type=CONFIG.ALIGNMENT.LOSS_TYPE,
        similarity_type=CONFIG.ALIGNMENT.SIMILARITY_TYPE,
        num_cycles=int(batch_size * num_steps * CONFIG.ALIGNMENT.FRACTION),
        cycle_length=CONFIG.ALIGNMENT.CYCLE_LENGTH,
        temperature=CONFIG.ALIGNMENT.SOFTMAX_TEMPERATURE,
        label_smoothing=CONFIG.ALIGNMENT.LABEL_SMOOTHING,
        variance_lambda=CONFIG.ALIGNMENT.VARIANCE_LAMBDA,
        huber_delta=CONFIG.ALIGNMENT.HUBER_DELTA,
        tcc_regression_margin=CONFIG.ALIGNMENT.TCC_REGRESSION_MARGIN,
        normalize_indices=CONFIG.ALIGNMENT.NORMALIZE_INDICES,
        paired_matching=True,
        metadata=metadata,
        causal_lambda=CONFIG.ALIGNMENT.CAUSAL_LAMBDA if training else 0.0,
        causal_margin=CONFIG.ALIGNMENT.CAUSAL_MARGIN,
        forward_variance_lambda=CONFIG.ALIGNMENT.FORWARD_VARIANCE_LAMBDA,
        global_step=global_step,
        training=training)

    if merged_meta is not None:
        loss_dict['merged_metadata'] = merged_meta

    return loss, loss_dict
