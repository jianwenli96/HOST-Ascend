# coding=utf-8
"""Cycle consistency loss for unsupervised training."""

from algos.algorithm import Algorithm
from config import CONFIG
from tcc.deterministic_alignment import compute_deterministic_alignment_loss_paired

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

    # embs structure: [Main_0, ..., Main_B-1, Ref_0, ..., Ref_B-1]
    batch_size = embs.size(0)
    real_batch_size = batch_size // 2

    embs_main = embs[:real_batch_size]
    embs_ref = embs[real_batch_size:]

    steps_main = steps[:real_batch_size]
    steps_ref = steps[real_batch_size:]

    seq_lens_main = seq_lens[:real_batch_size]
    seq_lens_ref = seq_lens[real_batch_size:]

    # --- REDUNDANT MERGING REMOVED: Rely on Algorithm.forward() for merging ---
    # In ZeRO-3, merging must happen in forward().
    loss, loss_dict = compute_deterministic_alignment_loss_paired(
        embs_main, embs_ref, steps_main, steps_ref, seq_lens_main, seq_lens_ref,
        embs.size(1), real_batch_size,
        similarity_type=CONFIG.ALIGNMENT.SIMILARITY_TYPE,
        temperature=CONFIG.ALIGNMENT.SOFTMAX_TEMPERATURE,
        variance_lambda=CONFIG.ALIGNMENT.VARIANCE_LAMBDA,
        normalize_indices=CONFIG.ALIGNMENT.NORMALIZE_INDICES,
        normalize_embeddings=CONFIG.ALIGNMENT.NORMALIZE_EMBEDDINGS,
        tcc_regression_margin=CONFIG.ALIGNMENT.TCC_REGRESSION_MARGIN,
        forward_variance_lambda=CONFIG.ALIGNMENT.FORWARD_VARIANCE_LAMBDA)

    if merged_meta is not None:
        loss_dict['merged_metadata'] = merged_meta

    return loss, loss_dict
