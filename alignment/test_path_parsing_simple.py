#!/usr/bin/env python3
"""
Simple standalone test for path parsing logic.
Does not require full dataset imports.
"""

def parse_video_path(path_str):
    """
    Parse video path (standalone version for testing).
    """
    if ':' not in path_str:
        # Old format
        return {
            'video_dir': path_str,
            'segment_id': None,
            'frame_start': None,
            'frame_end': None,
            'original_path': path_str
        }
    
    # New format
    parts = path_str.rsplit(':', 2)
    
    if len(parts) != 3:
        return {
            'video_dir': path_str,
            'segment_id': None,
            'frame_start': None,
            'frame_end': None,
            'original_path': path_str
        }
    
    video_dir, segment_id_str, frame_range_str = parts
    
    try:
        segment_id = int(segment_id_str)
        
        if '-' in frame_range_str:
            frame_start_str, frame_end_str = frame_range_str.split('-')
            frame_start = int(frame_start_str)
            frame_end = int(frame_end_str)
        else:
            frame_start = None
            frame_end = None
        
        return {
            'video_dir': video_dir,
            'segment_id': segment_id,
            'frame_start': frame_start,
            'frame_end': frame_end,
            'original_path': path_str
        }
    except (ValueError, AttributeError) as e:
        print(f"Warning: Failed to parse '{path_str}': {e}")
        return {
            'video_dir': path_str,
            'segment_id': None,
            'frame_start': None,
            'frame_end': None,
            'original_path': path_str
        }

def main():
    print("=" * 70)
    print("Path Parsing Test - Segmented Video Format")
    print("=" * 70)
    
    # Test cases
    test_cases = [
        {
            "name": "Old format (no segments)",
            "path": "/mnt/oss/data/observations/732/894608",
            "expected": {
                'video_dir': "/mnt/oss/data/observations/732/894608",
                'segment_id': None,
                'frame_start': None,
                'frame_end': None
            }
        },
        {
            "name": "New format (with segment and frame range)",
            "path": "/mnt/oss/data/observations/732/894608/videos:1:456-651",
            "expected": {
                'video_dir': "/mnt/oss/data/observations/732/894608/videos",
                'segment_id': 1,
                'frame_start': 456,
                'frame_end': 651
            }
        },
        {
            "name": "New format (segment 0)",
            "path": "/mnt/oss/data/observations/732/912239/videos:0:100-200",
            "expected": {
                'video_dir': "/mnt/oss/data/observations/732/912239/videos",
                'segment_id': 0,
                'frame_start': 100,
                'frame_end': 200
            }
        },
        {
            "name": "New format (large frame numbers)",
            "path": "/mnt/oss/data/videos:2:1000-5000",
            "expected": {
                'video_dir': "/mnt/oss/data/videos",
                'segment_id': 2,
                'frame_start': 1000,
                'frame_end': 5000
            }
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"  Input: {test['path']}")
        
        result = parse_video_path(test['path'])
        
        print(f"  Result:")
        print(f"    video_dir: {result['video_dir']}")
        print(f"    segment_id: {result['segment_id']}")
        print(f"    frame_start: {result['frame_start']}")
        print(f"    frame_end: {result['frame_end']}")
        
        # Check expectations
        success = True
        for key, expected_val in test['expected'].items():
            if result[key] != expected_val:
                print(f"  ❌ FAILED: {key} = {result[key]}, expected {expected_val}")
                success = False
                failed += 1
                break
        
        if success:
            print(f"  ✓ PASSED")
            passed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)
    
    if failed == 0:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) FAILED!")
        return 1

if __name__ == "__main__":
    exit(main())
