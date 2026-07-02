# Qwen3-VL/TCC 深度调试指南 (Expert Edition)

## 1. 核心修复回顾 (Stability Checklist)
当环境重建或代码迁移时，请确保以下三个“保险丝”已就位：
- **Loss 稳定性**: `tcc/losses.py` 中的 `variance` epsilon 必须 $\ge 1e-4$。
- **RoPE 严格对齐**: 禁止使用减法偏移，必须使用 `position_ids` 的切片覆盖逻辑。
- **边界 ID**: 必须识别 `151653` (`<|vision_end|>`) 来确定 Packed 组的末尾，否则最后组的 RoPE 会因 Assistant Header 的干扰而溢出。

## 2. 三级监控埋点 (Hot-Spots)

### 第一级：数据流入口 (Forward Path)
**位置**: `monkey_patch_forward.py`
**检查点**:
- 在 `inputs_embeds` 进入 LLM 之前，插入：
  ```python
  if not torch.isfinite(inputs_embeds).all():
      # 此时如果 NaN，说明 Vision Tower 或 Projector 权重已报废
      print(f"FAILED: inputs_embeds stats: {inputs_embeds.abs().max()}")
  ```

### 第二级：损失函数 (Loss Calculation)
**位置**: `tcc/losses.py`
**检查点**:
- 监控 `exp(-log_var)` 的值。如果该值超过 1000，梯度极易在 BF16 下溢出。
- 检查 `beta` (Similarity Softmax) 是否退化为 One-hot。如果方差极小且 Loss 极高，必崩。

### 第三级：梯度回传 (Backward Path)
**位置**: `models.py`
**操作**: 注册 `register_full_backward_hook`。
**原理**: TCC 的 Loss 经过 Projector 放大后传回给视觉端，如果 Projector 的梯度范数超过 10.0，建议开启 `gradient_clipping`。

## 3. 常见数值问题诊断表

| 现象 | 检测位置 | 根本原因 | 解决方案 |
| :--- | :--- | :--- | :--- |
| **CUDA error: illegal memory access** | `_apply_group_position_ids` | `position_ids` 出现了负数或超大值 | 检查 `group_starts` 是否正确识别了 `<|vision_end|>` |
| **Loss 突然变成 0.000 或 NaN** | `tcc/losses.py` | Softmax 后的 `beta` 全自 0 或物理距离爆炸 | 检查 `pairwise_l2_distance` 是否加了平方，确认为 `cdist` 模式 |
| **权重在几个 Step 后缓慢塌陷** | `optimizer` | 学习率过高配合 BF16 精度不足 | 减小 LR 至 5e-6 或增加 `eps=1e-8` 于 AdamW |

## 4. Packing (打包装箱) 关键逻辑
这个项目使用了 Packed Sequence 提升 3 倍速度，维持对齐的核心在于：
- **物理组长度**：由 `input_ids` 中的视觉标记决定。
- **逻辑组边界**：由 `monkey_patch` 逻辑强制对齐。
- **冲突处理**：如果 Assistant 的回答被错误地卷入视觉组，LLM 会因为位置编码错误（RoPE 冲突）而产生极大的 Loss，进而导致 NaN。
