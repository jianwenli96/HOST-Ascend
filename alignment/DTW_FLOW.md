# DTW 相关流程说明

本文档详细说明 `tcc/deterministic_alignment.py` 中与 DTW（Dynamic Time Warping）相关的数据流、索引空间和 remap 逻辑。

---

## 1. 总体目的

- **Cycle-consistency 损失**：用 student 的相似度矩阵和 softmax 算 MR/RM 的分类/回归损失（与 DTW 解耦）。
- **DTW 索引**：用一条**稳定的**对齐路径（teacher 或 student 的 sim 矩阵上做 DTW）得到「每个 main 帧应对齐到哪个 ref 帧」的离散索引。
- **DTW Guidance Loss**：用 DTW 索引作为回归目标，让 student 的 softmax 分布在时间轴上向 DTW 对齐靠拢（`pred_time` vs `true_time` 的 MSE）。
- **日志/可视化**：把 DTW 索引写入 `loss_dict['forward_dtw_indices']` 等，供 JSONL 与 MP4 使用。

因此 DTW 流程只负责：**在某个 sim 矩阵上算出一条对齐路径 → 得到 (B, T_main) 的 ref 下标 → 可选地 remap 到 student 空间 → 用于回归损失和保存**。

---

## 2. 两条 DTW 输入路径：Teacher vs Student

在 `compute_deterministic_alignment_loss_paired` 中，DTW 使用的相似度矩阵和 steps 二选一：

| 条件 | 使用的嵌入与 sim | direction 传给 DTW |
|------|------------------|--------------------|
| `teacher_embs_main is not None` 且 `CONFIG.ALIGNMENT.USE_TEACHER_FOR_DTW` | Teacher 嵌入：`t_main`, `t_ref[:, 1:]`（去 dustbin）→ `get_scaled_similarity` 得到 `dtw_sim_mr_input` / `dtw_sim_rm_input` | 全 +1（forward×forward） |
| 否则 | Student：`raw_sim_mr_real` / `raw_sim_rm`，`steps_ref[:, 1:]` 等 | `direction_main`, `direction_ref`（来自 do_reverse） |

- **Teacher 路径**：Teacher 始终看**正序**帧，DTW 在「正序 main × 正序 ref」空间做，得到的索引是 **forward 空间**的 ref 下标（0, 1, …, T_ref-1）。之后若提供 `do_reverse_main` / `do_reverse_ref`，会用 `_dtw_fwd_to_student` 把整张索引表 remap 到 **student 空间**（见第 5 节）。
- **Student 路径**：Sim 矩阵已是 student main × student ref（可能含逆序），DTW 内部会根据 `direction_source` / `direction_target` 对距离矩阵做 flip，输出**直接是 student 空间的列下标**，不再做 remap。

相关代码位置：约 754–788 行（分支）、779–788 行（direction 设置）。

---

## 3. 相似度与距离

- **相似度**：`get_scaled_similarity(embs1, embs2, similarity_type, temperature)`  
  - 支持 `cosine`（内积）或 `l2`（负的 L2 平方），再除以 `channels`（若未 normalize）和 `temperature`。  
  - 输出形状：(B, T1, T2)。
- **DTW 用距离**：在 `extract_alignment_indices_from_sim_matrix_dtw` 内，由相似度转成距离：  
  `dist = sqrt(max(-sim * temperature, 0))`（与 `sim = -dist^2/temperature` 对应）。  
  得到 `dist_matrices` (B, T1, T2)，再按 batch 做 DTW。

---

## 4. DTW 索引提取：`extract_alignment_indices_from_sim_matrix_dtw`

- **输入**：  
  - `sim_matrix` (B, T1, T2)，未 softmax；  
  - `steps_source` (B, T1)、`steps_target` (B, T2)（可选，用于推断方向）；  
  - `window`、`alignment_strategy`（"first" / "middle" / "last"）；  
  - `direction_source`、`direction_target` (B,)：±1，由 do_reverse 得到或由 steps 推断。
- **步骤概览**：  
  1. 由 sim 得到每样本的 `dist_matrix` (T1, T2)。  
  2. **按方向 flip**：若 `dir_src < 0` 则 flip 第 0 维；若 `dir_tgt < 0` 则 flip 第 1 维，使 DTW 在「逻辑正序」空间做。  
  3. 在 flip 后的距离矩阵上做 **DP**（带可选 window），得到最优路径 `path`（(i,j) 列表）。  
  4. 按 `alignment_strategy` 把每个 source 位置 i 映射到一个 target j（first/middle/last）。  
  5. **映射回原始（可能逆序）空间**：若 source 曾 flip，则 `actual_i = T1-1-i`；若 target 曾 flip，则 `actual_j = T2-1-j`；写入 `alignment_indices[b, actual_i] = actual_j`。
- **输出**：`alignment_indices` (B, T1)，表示「每个 source 位置对应的 target 列下标」。  
  - 走 **teacher** 时，传入的 direction 全 +1，不做 flip，因此输出是 **forward 空间**的 ref 下标。  
  - 走 **student** 时，输出已是 **student 空间**的 ref 列下标。

MR 调用：`steps_source=steps_main`，`steps_target=dtw_steps_ref_mr`（ref 去 dustbin）；  
RM 调用：`steps_source=steps_ref_source`，`steps_target=steps_main`。  
代码位置：约 420–576 行。

---

## 5. Teacher → Student 的 Remap：`_dtw_fwd_to_student`

**仅当**使用 teacher DTW（`_use_teacher_dtw`）**且** `do_reverse_main is not None` 时，会对 `dtw_indices_mr` / `dtw_indices_rm` 做 remap，把 **forward 空间**的索引变为 **student 空间**的索引（与 `ref_frame_paths` 等一致）。

- **含义**：  
  - `dtw_indices` (B, T_src)：当前是「forward 时间」下的 target 下标。  
  - `do_reverse_src`：student 的 **source** 是否逆序；  
  - `do_reverse_tgt`：student 的 **target** 是否逆序；  
  - `T_tgt`：target 的 forward 长度（ref 去 dustbin 后的长度）。
- **两步**：  
  1. **Source 逆序**：对需要 remap 的 batch 行，把整行按 dim=1 flip。即 student source 位置 0 对应 forward 位置 T_src-1，所以对齐结果也要按 source 轴翻转。  
  2. **Target 逆序**：对需要 remap 的 batch 行，把每个 target 下标 j（forward）映射为 student target 下标：`T_tgt - 1 - j`（因为 student ref 顺序是「最后一帧, …, 第一帧」）。
- **结果**：  
  - 仅 **do_reverse_tgt=True**（ref 逆序）时：forward [0,1,…,T-1] → student [T-1,…,0]，**递减**。  
  - **do_reverse_src 与 do_reverse_tgt 都为 True** 时：先 flip 行再做 `T_tgt-1-j`，得到的是 **student ref 下标**，可能呈递增 [0,1,…]（student ref 0 = 最后一帧），用于索引 `ref_frame_paths` 是正确的。

MR：`_dtw_fwd_to_student(dtw_indices_mr, do_reverse_main, do_reverse_ref, T_ref_real)`；  
RM：`_dtw_fwd_to_student(dtw_indices_rm, do_reverse_ref, do_reverse_main, T_main)`。  
代码位置：约 579–606 行（函数）、829–834 行（调用）。

---

## 6. DTW Guidance Loss（回归监督）

在 `CONFIG.ALIGNMENT.USE_DTW` 且 `dtw_guidance_lambda > 0` 时：

- **MR**：  
  - `sim_mr_for_pred`：student 的 MR softmax（去 dustbin），形状 (B, T_main, T_ref_real)。  
  - `pred_time_mr = sum(sim_mr_for_pred * steps_ref_norm, dim=-1)`：每个 main 位置一个标量时间（软对齐）。  
  - `true_time_mr`：用 **已 remap 的** `dtw_indices_mr` 在 `steps_ref_norm` 上 gather，得到每个 main 位置对应的 ref 时间。  
  - 损失：`(pred_time_mr - true_time_mr)^2` 的均值。
- **RM**：同理，用 `dtw_indices_rm` 和 `steps_main_norm` 得到 `true_time_rm`，与 `pred_time_rm` 做 MSE。

这里用到的 `dtw_indices_mr` / `dtw_indices_rm` 必须是 **student 空间**且与 `steps_ref` / `steps_main` 的索引一致（dustbin 已在前文 steps 切片中去掉）。  
代码位置：约 851–921 行。

---

## 7. 写入 loss_dict 与日志

- **DTW 索引**：  
  - `forward_dtw_indices`：MR 的 (B, T_main) ref 下标。若 `use_dustbin`，保存前会 `+1`，使 0..T_ref-1 变为 1..T_ref，与「ref 序列第 0 位是 dustbin」一致，便于日志里和 `ref_frame_paths` 对齐。  
  - `backward_dtw_indices`：RM 的 (B, T_ref_source)，不做 +1（RM 侧无 dustbin）。
- **主对齐索引**：若 `CONFIG.ALIGNMENT.USE_DTW`，则 `forward_indices` / `backward_indices` 取 DTW；否则取 argmax。
- 这些会进入 `loss_dict`，被 `utils.log_and_save_high_loss_samples` 写入 JSONL（如 `forward_dtw_indices`），可视化脚本再用来画对齐。

代码位置：约 1002–1058 行。

---

## 8. 配置项（config.py）

| 配置项 | 含义 |
|--------|------|
| `USE_TEACHER_FOR_DTW` | 是否用 teacher 嵌入算 DTW 的 sim 并得到索引（再 remap）；否则用 student sim。 |
| `USE_DTW` | 是否启用 DTW guidance loss，以及 primary 对齐是否用 DTW 索引。 |
| `DTW_GUIDANCE_LAMBDA` | DTW 回归损失的权重。 |
| `DTW_WINDOW` | DTW 的 band 窗口约束（None 表示全矩阵）。 |
| `DTW_ALIGNMENT_STRATEGY` | 多对一时取 "first" / "middle" / "last"。 |

---

## 9. 数据流简图

```
Teacher 路径:
  teacher_embs_main, teacher_embs_ref (正序)
    → get_scaled_similarity → dtw_sim_mr_input, dtw_sim_rm_input
    → extract_alignment_indices_from_sim_matrix_dtw(..., direction=+1)
    → dtw_indices_mr, dtw_indices_rm (forward 空间)
    → _dtw_fwd_to_student(..., do_reverse_main, do_reverse_ref, T_ref_real)
    → dtw_indices_mr/rm (student 空间)

Student 路径:
  raw_sim_mr_real, raw_sim_rm (student，可能逆序)
    → extract_alignment_indices_from_sim_matrix_dtw(..., direction_main, direction_ref)
    → dtw_indices_mr, dtw_indices_rm (已是 student 空间，不 remap)

后续统一:
  dtw_indices_mr/rm
    → DTW guidance loss（pred_time vs true_time）
    → forward_dtw_indices = dtw_indices_mr + (1 if use_dustbin else 0)
    → loss_dict → JSONL / 可视化
```

---

## 10. 小结

- **DTW 只负责**：在给定 sim 矩阵上算一条对齐路径，得到 (B, T_src) 的 target 下标。  
- **Teacher 路径**：索引在 forward 空间，需用 `_dtw_fwd_to_student` 转到 student 空间再参与 loss 和保存。  
- **Student 路径**：索引已在 student 空间，无需 remap。  
- **保存的 `forward_dtw_indices`**：一律是 **student ref 下标**（可带 dustbin +1），用于索引 `ref_frame_paths`；在 do_reverse_ref 且 do_reverse_main 同时为 True 时呈递增是预期行为。
