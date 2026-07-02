"""
Path transformation configuration for JSON mode.

Supports prefix-based path remapping for different datasets to handle
data migration or reorganization scenarios.

Example: '/open_data' -> '/open_data/cgy'
"""

# Path transformations by dataset name
# Format: {dataset_name: {old_prefix: new_prefix, ...}, ...}
# The dataset_name should match the dataset name extracted from JSON filename
# or stored in video_info['dataset']
PATH_TRANSFORMS = {
    'AgiBotWorld': {
        '/open_data': '/open_data/cgy/anns'
    },
    'InternDataA1': {
        '/x2robot_v2/share/xzn/datasets/x2robot': '/open_data/cgy/anns'
    },
    "10042": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10058": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10053": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "AgiBotWorld-Beta-v2": {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    "bridge_data_v2_img1_v2": {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    "bridge_data_v2_img3_v2": {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    "bridge_data_v2_img4_v2": {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    "droid_success_only_v2": {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    "fractal20220817_data_v2": {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    "Robochallenge_x2": {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    "robocoin_v2": {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    "robomind_v2.0_v2": {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    'robomindv2': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    'robomind': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    }
    # Add more datasets as needed:
    # 'BridgeV2': {
    #     '/old/bridge/path': '/new/bridge/path'
    # },
    # 'RT1': {
    #     '/old/rt1/path': '/new/rt1/path'
    # },
}


def get_path_transforms(dataset_name):
    """
    Get path transformation rules for a dataset.
    
    Args:
        dataset_name: Dataset name (e.g., 'AgiBotWorld', 'BridgeV2')
        
    Returns:
        Dictionary of {old_prefix: new_prefix} or empty dict if no transforms
    """
    if dataset_name is None:
        return {}
    
    # Direct lookup (case-sensitive)
    return PATH_TRANSFORMS.get(dataset_name, {})


def add_dataset_transforms(dataset_name, transforms):
    """
    Add or update path transformation rules for a dataset.
    
    Args:
        dataset_name: Dataset name
        transforms: Dictionary of {old_prefix: new_prefix}
    """
    PATH_TRANSFORMS[dataset_name] = transforms


def has_transforms(dataset_name):
    """
    Check if a dataset has any path transformation rules configured.
    
    Args:
        dataset_name: Dataset name
        
    Returns:
        Boolean indicating if transforms exist
    """
    if dataset_name is None:
        return False
    
    return dataset_name in PATH_TRANSFORMS and bool(PATH_TRANSFORMS[dataset_name])
