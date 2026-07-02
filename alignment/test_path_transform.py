#!/usr/bin/env python3
"""
测试脚本: 验证 task_paths.json 路径转换功能
"""
import sys
sys.path.insert(0, '/x2robot_v2/ethanchen/code/tcc_py_Qwen3_video_fast_3_3_aug_high_reverse_causal_dustbin_eval_var_attn_pool_2_e_3_vid_2_large_2_mp4_dtw_3')

from config import CONFIG

def test_path_transform():
    """测试路径转换功能"""
    
    # 创建一个模拟的 LiberoDataset 路径转换方法
    def apply_task_paths_transform(task_paths_file, dataset_type):
        """
        根据数据集类型对 task_paths_file 路径进行转换
        """
        # 检查是否有该数据集的转换规则
        if dataset_type not in CONFIG.DATA.TASK_PATHS_TRANSFORMS:
            return task_paths_file
        
        transform_rules = CONFIG.DATA.TASK_PATHS_TRANSFORMS[dataset_type]
        transformed_path = task_paths_file
        
        # 应用所有转换规则（按顺序）
        for old_prefix, new_prefix in transform_rules.items():
            if transformed_path.startswith(old_prefix):
                transformed_path = transformed_path.replace(old_prefix, new_prefix, 1)
                break  # 只应用第一个匹配的规则
        
        return transformed_path
    
    print("=" * 70)
    print("测试路径转换功能")
    print("=" * 70)
    
    # 测试案例1: AgiBotWorld 数据集
    test_cases = [
        {
            "dataset_type": "AgiBotWorld",
            "original_path": "/open_data/AgiBotWorld-Beta-Full/observations/732/894484/videos/0/task_paths.json",
            "expected_path": "/open_data/cgy/AgiBotWorld-Beta-Full/observations/732/894484/videos/0/task_paths.json"
        },
        {
            "dataset_type": "AgiBotWorld",
            "original_path": "/open_data/some/other/path/task_paths.json",
            "expected_path": "/open_data/cgy/some/other/path/task_paths.json"
        },
        {
            "dataset_type": "SomeOtherDataset",
            "original_path": "/open_data/AgiBotWorld-Beta-Full/observations/732/894484/videos/task_paths.json",
            "expected_path": "/open_data/AgiBotWorld-Beta-Full/observations/732/894484/videos/task_paths.json"  # 无变化
        },
        {
            "dataset_type": "AgiBotWorld",
            "original_path": "/mnt/oss/some/path/task_paths.json",
            "expected_path": "/mnt/oss/some/path/task_paths.json"  # 不匹配前缀，无变化
        }
    ]
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        dataset_type = test_case["dataset_type"]
        original_path = test_case["original_path"]
        expected_path = test_case["expected_path"]
        
        transformed_path = apply_task_paths_transform(original_path, dataset_type)
        
        print(f"\n测试案例 {i}:")
        print(f"  数据集类型: {dataset_type}")
        print(f"  原始路径:   {original_path}")
        print(f"  期望路径:   {expected_path}")
        print(f"  转换路径:   {transformed_path}")
        
        if transformed_path == expected_path:
            print(f"  ✓ PASSED")
        else:
            print(f"  ✗ FAILED")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("所有测试通过! ✓")
    else:
        print("部分测试失败! ✗")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    success = test_path_transform()
    sys.exit(0 if success else 1)
