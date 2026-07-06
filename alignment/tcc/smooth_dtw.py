"""SmoothDTW-based alignment probability computation.

Adapted from D2TW (Hadji et al., 2021) for PyTorch batched training.
Original TF reference: VideoAlignment/d2tw/smoothDTW.py

Two modes (controlled by ``bidirectional``):

  bidirectional=False  (forward-only, default)
  ─────────────────────────────────────────────
  beta[i,j] = softmax(-sdtw_fwd[i,:] / γ_s)
  sdtw_fwd[i,j] = "min cost from (0,0) to (i,j)"
  → temporal bias without strict monotone argmax

  bidirectional=True  (forward-backward, closer to hard DTW)
  ──────────────────────────────────────────────────────────
  beta[i,j] = softmax(-(sdtw_fwd[i,j] + sdtw_bwd[i,j] - cost[i,j]) / γ_s)
  sdtw_bwd[i,j] = "min cost from (i,j) to (T1,T2)"
  → P(cell (i,j) lies on the globally optimal path)
  → as γ_s → 0, argmax(beta[b, i, :]) converges to the hard DTW path column p_i

Two compute methods (controlled by ``method``):

  method="loop"       — O(T²) sequential Python loop, safe for T1≠T2.
  method="anti_diag"  — O(T) anti-diagonal parallel loop, requires T1==T2.
                        ~10–20× fewer Python iterations for typical T=20–40.
                        Falls back to "loop" with a warning when T1≠T2.

Usage in TCC soft-NN:
    beta = smooth_dtw_probs(sim_12, ...)          # [B, T1, T2], rows sum to 1
    nn_embs = torch.bmm(beta, embs2)              # [B, T1, D]
"""

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _soft_min_vec(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                  gamma_s: float) -> torch.Tensor:
    """Vectorised soft-min over three same-shape tensors.

    Args:
        a, b, c  : [...] float32 tensors (any broadcastable shape)
        gamma_s  : temperature

    Returns:
        soft-min result, same shape as inputs.
    """
    neighbors = torch.stack([a, b, c], dim=-1)   # [..., 3]
    probs = F.softmax(-neighbors / gamma_s, dim=-1)
    return probs[..., 0] * a + probs[..., 1] * b + probs[..., 2] * c


def _smooth_dtw_anti_diag(
    cost: torch.Tensor,
    gamma_s: float,
    bidirectional: bool,
    return_cost: bool = False,
) -> torch.Tensor:
    """Anti-diagonal parallel SmoothDTW.  Requires T1 == T2 == T.

    Loop count: 2T−1  (vs T² in the sequential loop).
    Each step does vectorised [B, T] tensor ops instead of B-element scalars.

    Forward pass shift rules (anti-diagonal k, position d = i − max(0, k−T+1)):
        k < T  : left = prev1,              up = shift_right(prev1),   diag = shift_right(prev2)
        k == T : left = shift_left(prev1),  up = prev1,                diag = prev2   ← special
        k > T  : left = shift_left(prev1),  up = prev1,                diag = shift_left(prev2)

    Backward pass shift rules (mirror of forward):
        k < T−1  : right = prev1_b,               down = shift_left(prev1_b),  diag_s = shift_left(prev2_b)
        k == T−2 : right = prev1_b,               down = shift_left(prev1_b),  diag_s = prev2_b  ← special
        k >= T−1 : right = shift_right(prev1_b),  down = prev1_b,              diag_s = shift_right(prev2_b)

    Reconstruction: stack all anti-diagonals then gather with pre-computed index tables
        (pure torch ops, fully differentiable, no Python loop).

    Args:
        cost         : [B, T, T] float32 per-cell DTW cost
        gamma_s      : soft-min temperature
        bidirectional: if True, also compute backward pass

    Returns:
        beta : [B, T, T] float32, row-normalised probability matrix
    """
    B, T, _ = cost.shape   # T1 == T2 == T guaranteed by caller
    device   = cost.device
    INF      = 1e9

    inf_col  = torch.full((B, 1), INF, device=device, dtype=torch.float32)

    # --- helpers -----------------------------------------------------------
    def shift_right(x: torch.Tensor) -> torch.Tensor:
        """Prepend INF column, drop last column: position d ← d+1."""
        return torch.cat([inf_col, x[:, :-1]], dim=-1)   # [B, T]

    def shift_left(x: torch.Tensor) -> torch.Tensor:
        """Append INF column, drop first column: position d ← d-1."""
        return torch.cat([x[:, 1:], inf_col], dim=-1)    # [B, T]

    def cost_antidiag(k: int) -> torch.Tensor:
        """Extract cost values on anti-diagonal k as [B, T], padded with INF."""
        row_start = max(0, k - (T - 1))
        length_k  = min(k + 1, T, 2 * T - 1 - k)
        rows = torch.arange(row_start, row_start + length_k, device=device)
        cols = k - rows                                    # j = k - i
        c_valid = cost[:, rows, cols]                      # [B, length_k]
        if length_k < T:
            c_valid = torch.cat(
                [c_valid,
                 torch.full((B, T - length_k), INF, device=device, dtype=torch.float32)],
                dim=-1,
            )
        return c_valid   # [B, T]

    def invalid_mask(k: int) -> torch.Tensor:
        """Boolean mask [1, T]: True at positions d >= length_k (invalid)."""
        length_k = min(k + 1, T, 2 * T - 1 - k)
        mask = torch.arange(T, device=device) >= length_k   # [T]
        return mask.unsqueeze(0)                             # [1, T]

    # --- Pre-compute reconstruction index tables (no grad) -----------------
    with torch.no_grad():
        i_idx = torch.arange(T, device=device).unsqueeze(1).expand(T, T)  # [T, T]
        j_idx = torch.arange(T, device=device).unsqueeze(0).expand(T, T)  # [T, T]
        k_idx = i_idx + j_idx                                               # [T, T]
        d_idx = i_idx - torch.clamp(k_idx - T + 1, min=0)                 # [T, T]

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------
    # k=0: only cell (0,0); origin boundary = 0 so fwd[0,0] = cost[0,0]
    c0       = cost_antidiag(0)                                     # [B, T]
    fwd_list = [c0]
    prev2    = torch.full((B, T), INF, device=device, dtype=torch.float32)
    prev1    = c0

    for k in range(1, 2 * T - 1):
        c_k = cost_antidiag(k)                                      # [B, T]

        if k < T:
            left = prev1
            up   = shift_right(prev1)
            diag = shift_right(prev2)
        elif k == T:
            left = shift_left(prev1)
            up   = prev1
            diag = prev2                                            # ← special: no shift
        else:   # k > T
            left = shift_left(prev1)
            up   = prev1
            diag = shift_left(prev2)

        cur = c_k + _soft_min_vec(left, diag, up, gamma_s)
        cur = cur.masked_fill(invalid_mask(k), INF)

        fwd_list.append(cur)
        prev2, prev1 = prev1, cur

    # Reconstruct inner table [B, T, T] via a single gather (no Python loop)
    # fwd_stacked[b, k, d] = fwd_list[k][b, d]
    fwd_stacked = torch.stack(fwd_list, dim=1)           # [B, 2T-1, T]
    inner       = fwd_stacked[:, k_idx, d_idx]           # [B, T, T]

    if not bidirectional:
        beta = F.softmax(-inner / gamma_s, dim=-1)
        if return_cost:
            return beta, inner[:, -1, -1]   # (B, T, T), (B,)
        return beta

    # -----------------------------------------------------------------------
    # Backward pass
    # bwd[i,j] = cost[i,j] + soft_min(bwd[i+1,j+1], bwd[i+1,j], bwd[i,j+1])
    # Terminal cell (T-1, T-1): bwd = cost[T-1, T-1], no successors.
    # -----------------------------------------------------------------------
    c_term       = cost_antidiag(2 * T - 2)              # [B, T], pos 0 = (T-1,T-1)
    bwd_list_rev = [c_term]
    prev2_b      = torch.full((B, T), INF, device=device, dtype=torch.float32)
    prev1_b      = c_term

    for k in range(2 * T - 3, -1, -1):
        c_k = cost_antidiag(k)                                      # [B, T]

        if k >= T - 1:
            right  = shift_right(prev1_b)
            down   = prev1_b
            diag_s = shift_right(prev2_b)
        else:   # k < T-1
            right  = prev1_b
            down   = shift_left(prev1_b)
            diag_s = prev2_b if k == T - 2 else shift_left(prev2_b)  # ← special at k=T-2

        cur_b = c_k + _soft_min_vec(right, diag_s, down, gamma_s)
        cur_b = cur_b.masked_fill(invalid_mask(k), INF)

        bwd_list_rev.append(cur_b)
        prev2_b, prev1_b = prev1_b, cur_b

    # Reverse so bwd_list[k] corresponds to anti-diagonal k
    bwd_list    = bwd_list_rev[::-1]
    bwd_stacked = torch.stack(bwd_list, dim=1)           # [B, 2T-1, T]
    bwd_inner   = bwd_stacked[:, k_idx, d_idx]           # [B, T, T]

    # Forward-backward combined (subtract cost to avoid double-counting)
    combined = inner + bwd_inner - cost                  # [B, T, T]
    beta = F.softmax(-combined / gamma_s, dim=-1)
    if return_cost:
        return beta, inner[:, -1, -1]   # forward terminal cost, (B,)
    return beta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smooth_dtw_probs(
    sim: torch.Tensor,
    gamma_s: float = 0.1,
    gamma_f: float = 0.1,
    bidirectional: bool = False,
    method: str = "loop",
    return_cost: bool = False,
) -> torch.Tensor:
    """Compute soft-DTW-based matching probability matrix.

    Args:
        sim           : [B, T1, T2]  — scaled similarity matrix from
                        ``get_scaled_similarity``.  Higher = more similar.
        gamma_s       : Temperature for the DTW soft-min recurrence and the
                        final row-wise softmax.  Smaller → harder, more decisive.
        gamma_f       : Column-normalisation temperature (D2TW Eq. 4).
                        Set to 0 to skip column normalisation.
        bidirectional : If False (default), use forward-only table (original).
                        If True, use forward + backward table so that
                        ``argmax(beta[b, i, :])`` converges to the hard DTW
                        path column as γ_s → 0 (closes train/infer gap).
        method        : ``'loop'``       — O(T²) sequential Python loop.
                                           Safe for any T1, T2.
                        ``'anti_diag'``  — O(T) anti-diagonal parallel loop.
                                           Requires T1 == T2; falls back to
                                           ``'loop'`` with a warning otherwise.
        return_cost   : If True, also return the soft-DTW path cost ``sdtw[-1,-1]``
                        as a second output of shape ``(B,)``.  This is the cumulative
                        cost of the globally optimal alignment path and is the same
                        quantity minimised by the D2TW loss in the original paper.
                        When method='anti_diag' and T1==T2, falls back to 'loop'
                        automatically so that ``inner`` is available.
                        Default: False (original single-output behaviour).

    Returns:
        beta : [B, T1, T2], same dtype as ``sim``.
               Row-normalised: ``beta[b, i, :].sum() == 1``.
               Drop-in replacement for ``F.softmax(sim, dim=-1)``.
        path_cost (only when return_cost=True) : [B] float32 tensor.
               The soft-DTW terminal cost ``sdtw[T1, T2]`` for each sample.

    Shape contract:
        sim      : (B, T1, T2)     — batched similarity matrix
        cost     : (B, T1, T2)     — per-cell DTW cost (non-negative)
        inner    : (B, T1, T2)     — forward cumulative cost (inner table)
        bwd_inner: (B, T1, T2)     — backward cumulative cost (bidirectional only)
        combined : (B, T1, T2)     — fwd + bwd - cost  (bidirectional)
        beta     : (B, T1, T2)     — row-softmax probability
    """
    assert sim.dim() == 3, (
        f"smooth_dtw_probs expects [B, T1, T2], got shape {sim.shape}"
    )
    B, T1, T2 = sim.shape
    orig_dtype = sim.dtype
    device     = sim.device

    # Work in float32 throughout for BF16/FP16 stability
    s = sim.float()   # [B, T1, T2]

    # ------------------------------------------------------------------
    # Step 1  —  Per-cell cost
    # ------------------------------------------------------------------
    if gamma_f > 0:
        col_log_probs = F.log_softmax(s / gamma_f, dim=1)  # [B, T1, T2]
        cost = -col_log_probs
    else:
        cost = -s   # [B, T1, T2]

    # ------------------------------------------------------------------
    # Dispatch to the requested compute method
    # ------------------------------------------------------------------
    if method == "anti_diag":
        if T1 != T2:
            logger.warning(
                "smooth_dtw_probs: method='anti_diag' requires T1==T2 "
                f"(got T1={T1}, T2={T2}); falling back to method='loop'."
            )
        else:
            _ad_result = _smooth_dtw_anti_diag(cost, gamma_s, bidirectional, return_cost=return_cost)
            if return_cost:
                return _ad_result[0].to(orig_dtype), _ad_result[1]
            return _ad_result.to(orig_dtype)

    # ------------------------------------------------------------------
    # method="loop"  (sequential, O(T²), original implementation)
    # Also used as fallback when anti_diag is requested but T1≠T2.
    # ------------------------------------------------------------------
    INF       = 1e9
    inf_cell  = torch.full((B,), INF, dtype=torch.float32, device=device)
    zero_cell = torch.zeros(B,        dtype=torch.float32, device=device)

    def _soft_min(a, b, c):
        """Soft-min over three [B] cost tensors → [B]."""
        neighbors = torch.stack([a, b, c], dim=-1)   # [B, 3]
        probs = F.softmax(-neighbors / gamma_s, dim=-1)
        return probs[:, 0] * a + probs[:, 1] * b + probs[:, 2] * c

    # Forward pass -------------------------------------------------------
    row0     = [zero_cell] + [inf_cell] * T2
    fwd_rows = [row0]

    for i in range(1, T1 + 1):
        prev = fwd_rows[i - 1]
        cur  = [inf_cell]
        for j in range(1, T2 + 1):
            c    = cost[:, i - 1, j - 1]
            left = cur[j - 1]
            diag = prev[j - 1]
            up   = prev[j]
            cur.append(c + _soft_min(left, diag, up))
        fwd_rows.append(cur)

    sdtw_fwd = torch.stack(
        [torch.stack(row, dim=1) for row in fwd_rows], dim=1
    )   # [B, T1+1, T2+1]
    inner = sdtw_fwd[:, 1:, 1:]   # [B, T1, T2]

    # Backward pass (bidirectional only) ---------------------------------
    if bidirectional:
        bwd_dict: dict = {}

        for i in range(T1, 0, -1):
            for j in range(T2, 0, -1):
                c = cost[:, i - 1, j - 1]
                if i == T1 and j == T2:
                    new_val = c
                else:
                    s_diag  = bwd_dict.get((i + 1, j + 1), inf_cell)
                    s_down  = bwd_dict.get((i + 1, j),     inf_cell)
                    s_right = bwd_dict.get((i,     j + 1), inf_cell)
                    new_val = c + _soft_min(s_diag, s_down, s_right)
                bwd_dict[(i, j)] = new_val

        bwd_inner = torch.stack(
            [torch.stack([bwd_dict[(i, j)] for j in range(1, T2 + 1)], dim=1)
             for i in range(1, T1 + 1)],
            dim=1,
        )   # [B, T1, T2]

        combined = inner + bwd_inner - cost
        beta = F.softmax(-combined / gamma_s, dim=-1)
    else:
        beta = F.softmax(-inner / gamma_s, dim=-1)

    beta = beta.to(orig_dtype)
    if return_cost:
        # inner[:, -1, -1] = sdtw_fwd[T1, T2]: the soft-DTW terminal path cost.
        # Shape: (B,), float32 (inner is computed in float32).
        return beta, inner[:, -1, -1]
    return beta
