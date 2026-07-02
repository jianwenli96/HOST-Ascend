# Segmented Video Format Support

## 概述

本次修改为数据集系统添加了**分段视频格式**支持，同时完全保持向后兼容旧格式。新格式允许一个视频被分成多个片段，每个片段可以独立作为训练样本使用。

## 数据格式说明

### 新格式语法

```
/path/to/video:segment_id:start_frame-end_frame
```

**示例**:
```
/mnt/oss/.../videos:1:456-651
```

- **第一部分**: 视频目录路径 (包含mp4文件)
- **第二部分**: segment_id - 该数据是视频的第几段 (0, 1, 2, ...)
- **第三部分**: 帧范围 - 该段在视频中的起止帧 (start-end)

### 旧格式 (完全兼容)

```
/path/to/video
```

**示例**:
```
/mnt/oss/.../observations/732/894608
```

- 简单的目录路径，指向包含完整视频数据的目录

## 目录结构

### 新格式目录结构

```
/mnt/.../videos/
├── camera1.mp4          # view由mp4文件名指定
├── camera2.mp4          # 多个mp4 = 多个视角
├── 0/                   # segment文件夹
│   └── task_paths.json  # 引用其他segments
├── 1/
│   └── task_paths.json
├── 2/
│   └── task_paths.json
└── instruction.txt      # 可选的任务描述
```

### task_paths.json格式

在新格式中，task_paths.json位于segment子文件夹中，引用的路径也使用新格式：

```json
{
  "same": [
    "/mnt/.../videos:0:100-200",
    "/mnt/.../videos:2:700-800"
  ],
  "100-95": [
    "/other/path/videos:1:300-400"
  ]
}
```

## View (视角) 处理逻辑

### 关键设计：通用的View判断

系统通过检查目录下是否存在mp4文件来自动决定view的处理方式：

```python
# 伪代码
if 目录下有mp4文件:
    views = [mp4文件名(不含扩展名), ...]
    # 例如: camera1.mp4 -> view="camera1"
else:
    views = [子目录名, ...]
    # 例如: images/, gripper_images/ -> views=["images", "gripper_images"]
```

**优势**:
- ✅ 不依赖格式标记或segment_id
- ✅ 新旧格式无缝共存
- ✅ 旧数据迁移到mp4时无需修改代码

## 核心修改说明

### 1. 路径解析 (`_parse_video_path`)

**位置**: `datasets.py` - LiberoDataset类

**功能**: 解析视频路径字符串，支持新旧两种格式

```python
def _parse_video_path(self, path_str):
    """
    返回:
        {
            'video_dir': 视频目录路径,
            'segment_id': 分段ID (None for old format),
            'frame_start': 起始帧 (None for old format),
            'frame_end': 结束帧 (None for old format),
            'original_path': 原始路径字符串
        }
    """
```

### 2. 延迟路径解析（Lazy Parsing）

**设计原则**: task_paths.json中可能包含上百个路径，但每次只会选择其中一个使用

**优化策略**:
- task_paths.json加载后**保持字符串格式**
- 只在 `random.choice` 选中路径后，才调用 `_parse_video_path` 解析
- 避免解析100个路径但只使用1个的性能浪费

**性能提升**:
- 旧方案: 加载task_paths → 解析100个路径 → 选择1个 = O(100)
- 新方案: 加载task_paths → 选择1个 → 解析1个 = O(1)

```python
# 伪代码示例
task_paths = load_json()  # {"same": [path1, path2, ..., path100]}
selected = random.choice(task_paths["same"])  # 先选择
parsed = _parse_video_path(selected)  # 只解析选中的
```

### 3. 统一View获取 (`_get_available_views`)

**位置**: `datasets.py` - LiberoDataset类

**功能**: 统一处理view获取，自动适配mp4和目录两种形式

```python
def _get_available_views(self, video_info):
    """
    返回可用的view名称列表
    - 有mp4: 返回mp4文件名列表
    - 无mp4: 返回子目录名列表
    """
```

### 4. 文件获取 (`_get_files`)

**修改**: 参数从 `path` 改为 `video_info` + `view`

**新功能**:
- 支持从mp4文件名匹配view
- 支持帧范围裁剪 (frame_start, frame_end)
- 完全兼容旧的图片目录格式

```python
def _get_files(self, video_info, view=None):
    """
    参数:
        video_info: dict或str
        view: 指定的视角名称 (mp4文件名或子目录名)
    
    返回:
        文件路径或帧路径列表
    """
```

### 5. 数据加载 (`_load_video_data_from_json`)

**修改**:
- 使用 `_get_available_views` 获取views
- 从segment子文件夹加载 task_paths.json
- 调用 `_get_files` 时传递 video_info 和 view

### 6. 视频数据集 (`LiberoVideoDataset`)

**修改**:
- `__init__`: 传递 video_info 到 `_get_files`
- `_get_item_impl`: 处理 video_info，支持segment格式的task_paths加载

## 测试验证

### 运行测试

```bash
# 简单的路径解析测试
python3 test_path_parsing_simple.py
```

### 测试结果

```
✓ Old format parsing
✓ New format parsing  
✓ All tests PASSED
```

## 使用示例

### 创建JSON配置文件

**agibot_video_paths.json**:
```json
[
  "/mnt/.../observations/732/894608/videos:1:456-651",
  "/mnt/.../observations/732/912239/videos:1:456-726",
  "/mnt/.../observations/732/894484/videos:1:426-621"
]
```

### 混合使用新旧格式

```json
[
  "/old/format/video1",
  "/old/format/video2",
  "/new/format/videos:0:100-200",
  "/new/format/videos:1:300-400"
]
```

### 训练脚本使用

```bash
python3 train.py \
  --video_paths /path/to/agibot_video_paths.json \
  --network Qwen3-VL-2B \
  --logdir logs/
```

## 兼容性保证

### 向后兼容

- ✅ 所有旧格式路径无需修改
- ✅ 旧的pickle数据格式完全支持
- ✅ 现有训练脚本无需改动

### 数据格式混合

- ✅ 同一个JSON文件可以混合新旧格式
- ✅ task_paths.json可以引用不同格式的路径
- ✅ 不同数据集可以使用不同格式

## 代码改动统计

| 文件 | 函数/方法 | 改动类型 | 行数 |
|------|-----------|----------|------|
| datasets.py | `_parse_video_path` | 新增 | +75 |
| datasets.py | `_get_available_views` | 新增 | +25 |
| datasets.py | `__init__` (LiberoDataset) | 修改 | +3 |
| datasets.py | `_get_files` | 重构 | +50 |
| datasets.py | `_load_video_data_from_json` | 重构（延迟解析） | +45 |
| datasets.py | `LiberoVideoDataset.__init__` | 修改 | +3 |
| datasets.py | `LiberoVideoDataset._get_item_impl` | 修改（延迟解析） | +40 |

**总计**: ~241行新增/修改代码

**性能优化**:
- ✅ 延迟解析task_paths：只解析被选中的路径（O(1) vs O(N)）
- ✅ 减少不必要的字符串解析开销

## 性能优化

### 1. 延迟路径解析（Lazy Parsing）
- task_paths.json 加载后保持字符串格式
- 只在 `random.choice` 选中路径后才解析
- 避免解析100个路径但只使用1个的浪费

### 2. 移除task_paths缓存
- **不使用缓存**：每次直接读取 task_paths.json 文件
- **理由**：数据量很大，每个video只访问一次，不会重复读取
- **优势**：
  - 节省内存（无需维护大量cache字典）
  - 减少内存泄漏风险
  - 适合大规模数据集训练

### 3. 其他优化
- ✅ 帧范围裁剪减少内存占用
- ✅ mp4文件统一处理，保持原有性能

## task_paths.json 路径转换

### 功能说明

为了支持数据迁移或重新组织的场景，我们添加了 `task_paths.json` 文件路径的自动转换机制。这允许针对不同数据集定义路径前缀替换规则。

### 配置方式

在 `config.py` 中配置 `CONFIG.DATA.TASK_PATHS_TRANSFORMS`:

```python
CONFIG.DATA.TASK_PATHS_TRANSFORMS = {
    'AgiBotWorld': {
        '/open_data': '/open_data/cgy'
    },
    'SomeOtherDataset': {
        '/old/path': '/new/path'
    }
}
```

### 工作原理

1. **数据集类型推断**: 从 JSON 文件名直接提取（如 `AgiBotWorld_video_paths.json` → `AgiBotWorld`）
2. **路径转换时机**: 在 `_load_video_data_from_json` 确定 `task_paths_file` 路径后自动应用
3. **转换逻辑**: 检查路径前缀是否匹配配置中的 `old_prefix`，如果匹配则替换为 `new_prefix`
4. **无匹配处理**: 如果数据集类型不在配置中，或路径不匹配任何前缀，则保持原路径不变

### 示例

**原始路径**:
```
/open_data/AgiBotWorld-Beta-Full/observations/732/894484/videos/0/task_paths.json
```

**转换后路径** (对于 AgiBotWorld 数据集):
```
/open_data/cgy/AgiBotWorld-Beta-Full/observations/732/894484/videos/0/task_paths.json
```

### 测试

运行 `test_path_transform.py` 进行测试:
```bash
conda run -n emu_vla_rl python test_path_transform.py
```

## 后续扩展

本实现为以下功能留有扩展空间：

1. **动态segment采样**: 在训练时随机选择不同segments
2. **Segment元数据**: 为每个segment添加额外属性
3. **层次化task_paths**: 支持更复杂的任务关系图
4. **更复杂的路径转换**: 支持正则表达式或函数式转换规则

## 注意事项

1. **路径格式**: 冒号(`:`)和连字符(`-`)是关键分隔符，避免在路径中使用
2. **帧索引**: frame_start和frame_end都是**包含**的 (inclusive)
3. **Segment文件夹**: 命名必须是数字字符串 (0, 1, 2, ...)
4. **MP4文件名**: 将直接作为view名称使用，命名应清晰易懂

## 故障排查

### 常见问题

**Q: 路径解析失败，返回原路径**
A: 检查格式是否正确: `path:segment_id:start-end`，所有部分必须存在

**Q: 找不到mp4文件**
A: 确保mp4文件直接在video_dir下，不在子目录中

**Q: task_paths.json加载失败**
A: 检查segment文件夹是否存在，task_paths.json是否在segment文件夹内

**Q: View选择错误**
A: 使用 `_get_available_views` 检查系统识别的views列表

## 更新日期

- 2026-02-14: 添加 task_paths.json 路径转换功能
- 2026-02-12: 初始版本

## 作者

Ethan Chen (with AI assistance)
