# Guide: Adding Forward Alignment Indices

This document summarizes the modifications required to extract and visualize **Forward Alignment Indices** ($S \to T$) in addition to the standard **Cycle-Consistency Indices** ($S \to T \to S$).

## 1. Context

*   **`alignment_indices` (Original)**: Represents the index in the *Source* sequence that the model returns to after a full cycle ($Source \to Target \to Source$). This measures cycle consistency (ideally $i \to i$).
*   **`forward_alignment_indices` (New)**: Represents the index in the *Target* sequence that is most similar to the current frame in the *Source* sequence ($Source \to Target$). This measures direct alignment/matching.

## 2. Code Modifications

### 2.1. `tcc/deterministic_alignment.py`

**Goal**: Expose the intermediate forward logits (similarity matrix) from `align_pair_of_sequences`.

**Step 1: Modify `align_pair_of_sequences`**
Update the return signature to include `sim_12` (the Source->Target logits).

```python
def align_pair_of_sequences(embs1, embs2, similarity_type, temperature):
    # ... existing calculation of sim_12 ...
    sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
    
    # ... calculation of logits (sim_21) ...

    # OLD: return logits, labels
    # NEW:
    return logits, labels, sim_12
```

**Step 2: Modify `compute_deterministic_alignment_loss_paired`**
Update the call sites and capture the new `sim` output.

```python
def compute_deterministic_alignment_loss_paired(...):
    # ...
    # Main -> Ref -> Main
    logits_mr, labels_mr, sim_mr = align_pair_of_sequences(embs_main, embs_ref, ...)

    # Ref -> Main -> Ref
    logits_rm, labels_rm, sim_rm = align_pair_of_sequences(embs_ref, embs_main, ...)

    # ... existing concatenation of logits ...
    
    # NEW: Concatenate forward similarities
    sim_forward = torch.cat([sim_mr, sim_rm], dim=0) # (2B, T, T)

    # ... existing loss calculation ...
    
    # ... inside 'classification' or 'regression' block where loss_dict is populated ...
    
    # NEW: Calculate and store Forward Indices
    forward_alignment_indices = torch.argmax(sim_forward, dim=-1) # (2B, T)
    loss_dict['forward_alignment_indices'] = forward_alignment_indices
    
    return loss, loss_dict
```

### 2.2. `utils.py`

**Goal**: Save the newly computed indices to the JSONL log file.

**Step 1: Modify `log_and_save_high_loss_samples`**
Add logic to extract `forward_alignment_indices` from `loss_dict` and write it to the record.

```python
def log_and_save_high_loss_samples(logdir, step, loss_dict, data):
    # ... existing logic ...
    
    # Inside the loop for writing records:
    if 'alignment_indices' in loss_dict:
        align_idx = loss_dict['alignment_indices'][idx].cpu().tolist()
        record["alignment_indices"] = align_idx

    # NEW BLOCK
    if 'forward_alignment_indices' in loss_dict:
        fwd_align_idx = loss_dict['forward_alignment_indices'][idx].cpu().tolist()
        record["forward_alignment_indices"] = fwd_align_idx
        
    f.write(json.dumps(record) + "\n")
```

### 2.3. `visualize_high_loss.py`

**Goal**: Visualize the direct alignment instead of the cycle consistency metric.

**Step 1: Modify Log Reading Logic**
Prioritize `forward_alignment_indices` when available.

```python
# Inside main loop
main_paths = record.get('frame_paths', [])
ref_paths = record.get('ref_frame_paths', [])

# MODIFIED: Prefer forward indices
alignment = record.get('forward_alignment_indices', record.get('alignment_indices', []))
```

## 3. Summary of Files Touched

1.  `tcc/deterministic_alignment.py`: Core logic update to calculate `argmax(sim_12)`.
2.  `utils.py`: Logging update to persist the new data.
3.  `visualize_high_loss.py`: Visualization tool update to consume the new data.
