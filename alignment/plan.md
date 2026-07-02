# Plan: Inject Discretized Joint State Tokens into TCC Alignment Pipeline

## Context

The TCC alignment pipeline (`datasets.py`) is vision-only. The VLA dataset (`mydatasets_for_vla.py`) loads per-frame joint states from episode JSONs, normalizes to [-1,1], and discretizes into language tokens injected into the model's input sequence. The user wants the same joint tokenization injected into `datasets.py` so the Qwen3-VL backbone receives proprioceptive context alongside visual frames, enriching the frame-level embeddings used for temporal alignment.

**Important differences from VLA reference:**
- VLA uses Emu3 model with `<|extra_N|>` tokens — Qwen3-VL does NOT have these tokens
- Qwen3-VL vocab_size=151936, but added tokens only go up to ID 151668 → IDs **151669–151935** (267 slots) exist in the embedding matrix but are unmapped
- We add 192 new tokens `<|joint_0|>` through `<|joint_191|>` mapped to these unused IDs (no embedding resize needed)
- Token order: **joint tokens BEFORE chunk video** (not after), then `<|file_sep|>`

## Files to Modify

| File | Changes |
|---|---|
| `config.py` | Add `CONFIG.JOINTS` section (incl. token ID range) |
| `datasets.py` | Joint token registration, loading, M-chunking propagation, token injection |
| `train.py` | Add joint tokens to processor tokenizer at init |

## Reference Code (read-only)

- `mydatasets_for_vla.py:1655-1813` — `_load_episode_raw()` + `_load_action_and_joint()`: JSON loading, norm vectors, normalization
- `mydatasets_for_vla.py:3326-3332` — Discretization: `np.digitize` → token strings
- `mydatasets_for_vla.py:383-386,1104-1111` — Joint-action mapping cache init (`/open_data/cgy/joint_action_mapping/`)
- `mydatasets_for_vla.py:1582-1616` — `_load_joint_action_mapping()` + `_resolve_mapping_entry()`
- Qwen3 tokenizer: `added_tokens.json` — last used ID is 151668; `config.json` — vocab_size=151936

---

## Implementation Steps

### 1. Config flags — `config.py` (after line ~202)

```python
CONFIG.JOINTS = edict()
CONFIG.JOINTS.USE_JOINTS = False
CONFIG.JOINTS.NUM_BINS = 192
CONFIG.JOINTS.JOINT_ACTION_MAPPING_DIR = '/open_data/cgy/joint_action_mapping'
CONFIG.JOINTS.JOINT_TOKEN_START_ID = 151669   # first unused ID in Qwen3 vocab (vocab_size=151936)
```

### 2. Register joint tokens on the tokenizer — `train.py` (after processor load, ~line 250)

After `processor = AutoProcessor.from_pretrained(...)`, add the 192 joint tokens:

```python
if getattr(CONFIG.JOINTS, 'USE_JOINTS', False):
    num_bins = CONFIG.JOINTS.NUM_BINS
    joint_tokens = [f"<|joint_{i}|>" for i in range(num_bins)]
    num_added = processor.tokenizer.add_tokens(joint_tokens, special_tokens=True)
    if is_master:
        logging.info(f"[joints] Added {num_added} joint tokens to tokenizer "
                     f"(IDs {processor.tokenizer.convert_tokens_to_ids(joint_tokens[0])}"
                     f"–{processor.tokenizer.convert_tokens_to_ids(joint_tokens[-1])})")
    # No resize needed: 151669+192=151861 < vocab_size=151936
```

### 3. Joint-action mapping cache — `datasets.py` `LiberoDataset.__init__()` (after cam_mapping block, ~line 988)

Load all `*_joint_action_mapping.json` files eagerly, same pattern as cam_mapping:

```python
self._joint_action_mapping_cache = {}
if getattr(CONFIG.JOINTS, 'USE_JOINTS', False):
    _jam_dir = CONFIG.JOINTS.JOINT_ACTION_MAPPING_DIR
    if _jam_dir and os.path.isdir(_jam_dir):
        for fname in os.listdir(_jam_dir):
            if fname.endswith('_joint_action_mapping.json'):
                ds = fname.replace('_joint_action_mapping.json', '')
                with open(os.path.join(_jam_dir, fname), 'r') as f:
                    self._joint_action_mapping_cache[ds] = json.load(f)
```

### 4. Helper methods on `LiberoDataset` (after `_context_steps_to_paths`, ~line 1386)

**4a. `_resolve_joint_mapping(dataset_name, video_dir)`** — Looks up mapping entry from cache. Single-key → return value directly. Multi-key → prefix-match `dirname(dirname(video_dir))`. Returns `(joint_keys, norm_min_delta)` or `([], None)`.

**4b. `_load_joint_data(video_info, dataset_name)`** — Opens `{video_dir}/{ep_name}.json`, reads `raw['data']`, extracts active `joint_keys` present in data, builds norm vectors from `nmd[key]['min']`/`nmd[key]['delta']`, assembles `[T, D_joint]` raw matrix, normalizes to `[-1,1]`. Returns `np.float32 [T, D_joint]` or `None`.

**4c. Module-level `_discretize_joints(joint_vec, num_bins=192)`** — Discretizes a normalized [-1,1] joint vector into `<|joint_N|>` token string:

```python
def _discretize_joints(joint_vec, num_bins=192):
    bins = np.linspace(-1, 1, num_bins + 1)[:-1]
    discretized = np.clip(np.digitize(joint_vec, bins=bins) - 1, 0, num_bins - 1).astype(int)
    return "".join(f"<|joint_{b}|>" for b in discretized)
```

### 5. Return `video_info`/`ref_video_info` from `_load_video_data_from_json()` (~line 1824)

Add two keys to the return dict so `_get_item_impl` can pass them to `_load_joint_data`:
```python
'video_info': video_info,
'ref_video_info': ref_video_info,
```

### 6. Load + subsample joints in `_get_item_impl()` (~after line 1905)

After `main_steps` and `ref_steps` are computed:
- Call `_load_joint_data()` for main and ref episodes → `main_joint_all [T_main, D]`, `ref_joint_all [T_ref, D]`
- Index at anchor steps: `main_joint_per_step = [main_joint_all[s] if s >= 0 else None for s in main_steps]`
- Same for ref. Handle dustbin (-1) → `None`.
- Add `'main_joint_per_step'` and `'ref_joint_per_step'` (list of `np.ndarray|None`, len=num_steps) to the return dict at line 2021.

### 7. Propagate through `_apply_m_chunking()` (~line 404-438)

Inside the per-chunk loop, slice joint lists using same `chunk_indices`:
```python
if orig_data.get('main_joint_per_step') is not None:
    chunk_data['main_joint_per_step'] = [orig_data['main_joint_per_step'][idx] for idx in chunk_indices]
if orig_data.get('ref_joint_per_step') is not None:
    chunk_data['ref_joint_per_step'] = [orig_data['ref_joint_per_step'][idx] for idx in chunk_indices]
```

### 8. Thread joints through `_build_qwen_input()` → `_pack_sequence()` (lines 584-683)

**8a. `_build_qwen_input` prepared dict** (line 639): Add `'main_joints': batch[i].get('main_joint_per_step')` and `'ref_joints': batch[i].get('ref_joint_per_step')`.

**8b. Pass 1 (main rows, line 652)**: Pass `joint_per_chunk=p['main_joints']` to `_pack_sequence`.

**8c. Pass 2 (ref rows, line 670)**: Pass `joint_per_chunk=p['ref_joints']` to `_pack_sequence`.

### 9. Inject tokens in `_pack_sequence()` (line 498-520)

Add `joint_per_chunk=None` parameter. Modify the chunk loop — **joint tokens go BEFORE the chunk video**:

```python
# Current (line 518-520):
for chunk in chunk_imgs:
    content.append({"type": "video", "video": chunk})
    content.append({"type": "text", "text": "<|file_sep|>"})

# New:
for ci, chunk in enumerate(chunk_imgs):
    # Joint tokens BEFORE chunk video
    if (joint_per_chunk is not None
            and ci < len(joint_per_chunk)
            and joint_per_chunk[ci] is not None):
        joint_str = _discretize_joints(joint_per_chunk[ci], CONFIG.JOINTS.NUM_BINS)
        content.append({"type": "text", "text": joint_str})
    content.append({"type": "video", "video": chunk})
    content.append({"type": "text", "text": "<|file_sep|>"})
```

Result per chunk: `[joint_tokens, chunk_video, "<|file_sep|>"]`

### 10. Teacher path — no special handling needed

Teacher calls `_build_qwen_input()` with the same `batch` dict. Joints are deterministic (no augmentation), so student and teacher get identical joint tokens automatically.

### 11. Align half videos — no joints

The align halves (global summary) do NOT get joint tokens. Only per-chunk videos get them. This keeps the align structure unchanged.

---

## Key Design Decisions

1. **Qwen3-VL joint tokens: `<|joint_0|>` through `<|joint_191|>`.** The VLA reference uses Emu3's `<|extra_N|>` tokens which don't exist in Qwen3. Qwen3's vocab_size=151936 but only IDs up to 151668 are mapped, leaving 267 unused embedding slots. We add 192 `<|joint_N|>` tokens starting at ID 151669 — no `resize_token_embeddings()` needed.

2. **Joint tokens placed BEFORE chunk video.** This provides proprioceptive context to the transformer before it processes the visual frames for that chunk, matching the natural conditioning order (state → observation).

3. **One joint vector per anchor step (not per context frame).** Each chunk in TCC represents one anchor step expanded to `num_ctx` neighboring frames per view. The joint state at the anchor step is the natural proprioceptive snapshot for that chunk. This adds ~14 tokens per chunk (typical D_joint=14), not 14*context_size.

4. **Self-contained norm logic (no UniVLA import).** The VLA dataset imports from `tools.joint_action_mapping_norms` via `sys.path.append("/share/project/yuqi.wang/UniVLA")`. Instead, we replicate the simple norm vector construction inline (~10 lines) to avoid the external dependency.

5. **Graceful degradation.** When `USE_JOINTS=False`, no mapping loaded, no JSON read, no tokens injected. When a dataset has no `joint_keys` or episode JSON is missing, `joint_per_step` is `None` and the pipeline behaves as before.

## Verification

1. **Config off (default):** Run training with `CONFIG.JOINTS.USE_JOINTS = False` → verify no behavior change, no errors.
2. **Config on:** Set `USE_JOINTS = True`, point `JOINT_ACTION_MAPPING_DIR` to mapping dir. Run a small training job and:
   - Check logs for `[joints] Added 192 joint tokens to tokenizer (IDs 151669–151860)` message
   - Add a temporary `print(text)` inside `_pack_sequence` (after line 551) to verify `<|joint_N|>` tokens appear BEFORE chunk videos
   - Verify `input_ids` shape grows by ~`D_joint * num_chunks` tokens per row
3. **Pickle mode:** Confirm no errors when using pickle data (joints will be `None`, no tokens injected).
4. **Loss convergence:** Compare training loss curves with/without joints over ~1000 steps to confirm the extra tokens don't break alignment learning.
