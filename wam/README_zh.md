# Self-Grounded Prediction (wam/)

本模块实现 **self-grounded prediction（自我校准预测）**，这是 **HOST (Human-to-robot One-shot
Skill Transfer)** 中负责从耦合的视觉示范中恢复执行的机制：先定位机器人在视觉示范中的当前进度，
再基于该定位片段预测机器人自身的未来观测，最后从预测的未来中推导出运动指令。整个过程由一个
带双专家（视频专家 + 动作专家）的 Mixture-of-Transformers 架构自回归扩散模型实现，代码库构建于
[Fast-WAM](https://arxiv.org/abs/2603.16666) 之上 —— 详见 [致谢](#致谢)。

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

## 目录

- [文件结构](#文件结构)
- [环境配置](#环境配置)
- [模型准备](#模型准备)
- [数据准备](#数据准备)
- [训练](#训练)
- [评测](#评测)
- [致谢](#致谢)
- [BibTeX](#bibtex)

## 文件结构

```text
wam/
├── configs/
│   ├── data/                 # 数据集配置（data_path、cam_mapping_dir、joint_action_mapping_dir 等）
│   ├── model/                # 模型架构配置（self_grounded_predictor_joint*.yaml）
│   └── task/                 # 任务级配置（组合 data + model 配置，设置训练超参）
├── scripts/
│   ├── train.py
│   ├── train_zero1_real_pac_headwise_ncp_ve.sh  # Deepspeed ZeRO-1 训练入口（已验证）
│   ├── eval_openloop.sh                          # 真机开环评测入口
│   ├── preprocess_action_dit_backbone.py         # 训练前预处理 ActionDiT backbone
│   └── precompute_text_embeds.py                 # 训练前预计算 T5 文本 embedding 缓存
├── eval/real_openloop/       # 真机开环评测工具
├── src/self_grounded_prediction/   # 核心代码（包名；见环境配置）
│   ├── models/wan22/         # SelfGroundedPredictor(Joint)、视频/动作 DiT、MoT、VAE、文本编码器
│   ├── datasets/custom/      # CustomDataset 加载器 —— 见数据准备
│   └── runtime.py            # Hydra 工厂函数（create_self_grounded_predictor）
├── runs/                     # 训练输出（ckpt、日志）
├── checkpoints/              # 预训练或外部权重
└── data/                     # 数据目录
```

## 环境配置

```bash
conda create -n self_grounded_prediction python=3.10 -y
conda activate self_grounded_prediction
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

安装后即可以 `self_grounded_prediction` 包名导入。注意仓库内实际的训练/评测脚本并不依赖这次
pip 安装 —— `scripts/_ensure_project_src.py` 会直接把 `src/` 加入 `sys.path`；只有当你想从本目录
之外 `import self_grounded_prediction` 时才需要 `pip install -e .`。

## 模型准备

以下步骤在训练和推理前都需要完成。

第一步：设置 Wan 模型目录（可选，默认 `./checkpoints`）：

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

第二步：预生成 ActionDiT backbone（由 Wan2.2 DiT 插值得到）：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/self_grounded_predictor_joint.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

## 数据准备

训练时每个数据集需要三样东西：**视频路径列表**、**相机映射**、**关节/动作归一化映射**。仓库自带的
配置（`configs/data/custom*.yaml`）指向的是内部集群路径 —— 请将 `data_path` /
`cam_mapping_dir` / `joint_action_mapping_dir` 替换成你自己的数据，格式如下。

### 1. 视频路径列表 —— `{dataset_id}_video_paths.json`

一个 JSON 数组，元素是每条 episode 的目录路径。dataset id 取自文件名（如
`10042_video_paths.json` → dataset id 为 `"10042"`），并作为查找相机映射、关节/动作映射、
`dataset_fps`/`dataset_image_size` 等配置字典的 key。

```json
["/data/episodes/task_a/episode_001", "/data/episodes/task_a/episode_002"]
```

`exclude_episode_json`（数据配置中的可选字段）格式相同，用于从训练中排除指定 episode。

### 2. Episode 目录内容

```text
episode_001/
├── episode_001.json     # 动作/关节轨迹 —— 若启用 actions/joints 则必需
├── instruction.txt      # 纯文本任务指令 —— 必需
├── instruction.pt       # 预计算的 T5 文本 embedding {"context": [L,4096], "mask": [L]}
├── info_dtw.json        # {"aligned_progress": {"<frame_idx>": <0..1>, ...}} —— 可选
├── task_paths.json      # {"same": ["/data/episodes/task_a/episode_002", ...]} —— 同任务的其他 episode
├── images/              # 一个相机视角的帧序列：0.jpg, 1.jpg, ...
└── gripper_images/      # 第二个相机视角的帧序列（若为双视角）
```

也支持用视频文件（`{view}.mp4`）代替帧序列子目录。帧数必须与 `episode_001.json` 中 `"data"`
数组的条目数一致 —— 不会做静默截断。

`episode_001.json`（动作/关节轨迹）：

```json
{
  "data": [
    {"follow_left_position": [x, y, z], "follow_left_rotation": [r, p, y], "follow_left_gripper": g,
     "follow_right_position": [x, y, z], "follow_right_rotation": [r, p, y], "follow_right_gripper": g},
    "... 每帧一条 ..."
  ]
}
```

旋转量为 `[roll, pitch, yaw]` 欧拉角三元组；当 `use_6d_rotation: true` 时，加载过程中会转换为
6D 旋转矩阵表示。

### 3. 相机映射 —— `{cam_mapping_dir}/{dataset_id}_cam_mapping.json`

将每个任务目录（episode 目录的上一级）映射到其磁盘上的相机/视角名有序列表：

```json
{"/data/episodes/task_a": ["images", "gripper_images"]}
```

`num_view_probs`（如 `'{"2": 1.0}'`）决定每个 batch 从列出的视角中采样几个。

### 4. 关节/动作归一化 —— `{joint_action_mapping_dir}/{dataset_id}_joint_action_mapping.json`

声明从 `episode_001.json` 读取哪些字段，以及各维度的 min/delta，用于将动作归一化到 `[-1, 1]`
（`norm = 2*(raw - min)/delta - 1`）：

```json
{
  "action_keys": ["follow_left_position", "follow_left_rotation", "follow_left_gripper",
                   "follow_right_position", "follow_right_rotation", "follow_right_gripper"],
  "joint_keys": ["left_rotation", "right_rotation", "left_position", "right_position"],
  "norm_min_delta": {
    "follow_left_position": {"min": [-0.08, -0.08, -0.08], "delta": [0.16, 0.16, 0.16]},
    "follow_left_rotation": {"min": [-0.30, -0.30, -0.30], "delta": [0.60, 0.60, 0.60]},
    "follow_left_gripper":  {"min": [-0.5], "delta": [6.5]}
  }
}
```

### 5. 其余按数据集配置的字段

在 `configs/data/custom.yaml` 中还需按 dataset id 设置：`dataset_fps`（用于最短长度过滤与动作
帧采样）和 `dataset_image_size`（`[width, height]`）。如果你的数据集动作维度与示例不同，需同步
更新 `processor.action_output_dim` / `processor.proprio_output_dim`，以及模型配置里对应的
`action_dim`。

## 训练

### 1) 训练前预计算 T5 embedding 缓存

```bash
python scripts/precompute_text_embeds.py task=real_joint_2cam_224_1e-4_pac_headwise_ncp_ve
# 多卡:
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=real_joint_2cam_224_1e-4_pac_headwise_ncp_ve
```

### 2) 训练

首次跑一个新任务时，先将对应 `configs/data/*.yaml` 中的 `pretrained_norm_stats` 设为 `null`。
跑完一次训练后，运行目录下会生成 `dataset_stats.json`
（`runs/{task_name}/{run_id}/dataset_stats.json`）；之后的训练可将 `pretrained_norm_stats`
指向该文件。

```bash
bash scripts/train_zero1_real_pac_headwise_ncp_ve.sh
```

这是已验证的生产训练入口 —— 它组合了
`configs/task/real_joint_2cam_224_1e-4_pac_headwise_ncp_ve.yaml`，该配置选择了
`self_grounded_predictor_joint_cross_attn_ve` 模型配置和 `custom_cross_all` 数据配置。可在命令行
覆盖任意 Hydra 字段，例如 `batch_size=4`。

## 评测

`eval/real_openloop/` 用你自己的 episode 数据（格式同[数据准备](#数据准备)）对训练好的模型做
开环评测，即根据当前观测预测一段动作 chunk 并与真值对比 —— **不是**实时机器人控制闭环，而是回放
一个已录制的数据集：

```bash
bash scripts/eval_openloop.sh
```

Checkpoint 和 dataset-stats 路径在脚本内设置；视觉编码器
（`configs/model/*.yaml` 中的 `visual_encoder.backbone_local_repo`/`backbone_weights_path`/
`siglip_local_weights_path`）目前默认指向内部权重路径，并在这些字段为空时自动回退到公开下载
（通过 `torch.hub` 下载 `facebookresearch/dinov2`，通过 TIMM 下载 SigLIP）—— 内部路径配置化的
当前进度见仓库根目录的 `OPEN_SOURCE_PATH_TODOS.md`。

## 致谢

本模块的代码库构建于
[Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
（Yuan et al.）之上，感谢 Fast-WAM 作者开源其代码库。

## BibTeX

如果我们的工作对你有帮助，欢迎引用：

```bibtex
@article{host2026,
  title={HOST: Human-to-robot One-shot Skill Transfer},
  % TODO: 待补充作者、发表信息、年份、DOI/arXiv
}
```
