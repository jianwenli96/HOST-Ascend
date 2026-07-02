## 2. Training Fix: DeepSpeed ZeRO-3 Compatibility
**Issue**: Calling the gate module inside `compute_loss` triggers `RuntimeError: mat2 must be a matrix` because ZeRO-3 shards parameters and only gathers them during the `forward()` pass.

**Constraint**: **NEVER** call learnable modules inside `compute_loss` under ZeRO-3.

**Solution**: 
- **Move Logic Forward**: Move grouping/merging of video chunks from `alignment.py` into `Algorithm.forward()`.
- **Execute in Forward**: Call `self.gate(raw_tokens_main, raw_tokens_ref)` inside `forward()` immediately after merging.
- **Data Propagation**: Return a dictionary from `forward()` containing merged embeddings, steps, seq_lens, and pre-computed `gates`.