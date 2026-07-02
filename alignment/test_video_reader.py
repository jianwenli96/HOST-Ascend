#!/usr/bin/env python3
"""
Test script to verify video_reader-rs works correctly.
"""
import os
import glob
from video_reader import PyVideoReader

def test_video_reader():
    """Test PyVideoReader with a sample video."""
    # Find a sample video (you may need to adjust this path)
    sample_dirs = [
        "/mnt/oss/zbl-open-data/users/ethanchen/AgiBotWorld-Beta-Full/observations/732/894484/videos",
        "."  # current directory as fallback
    ]
    
    video_path = None
    for directory in sample_dirs:
        if os.path.exists(directory):
            mp4_files = glob.glob(os.path.join(directory, "*.mp4"))
            if mp4_files:
                video_path = mp4_files[0]
                break
    
    if not video_path:
        print("Warning: No video files found for testing")
        print("Please provide a path to a video file to test")
        return
    
    print(f"Testing PyVideoReader with: {video_path}")
    print("-" * 80)
    
    try:
        # Test 1: Basic initialization
        print("Test 1: Basic initialization...")
        vr = PyVideoReader(video_path, oob_mode="skip", threads=0)
        num_frames = len(vr)
        print(f"  ✓ Video loaded successfully")
        print(f"  ✓ Number of frames: {num_frames}")
        
        # Test 2: Read single frame
        print("\nTest 2: Read single frame...")
        frame = vr[0]
        print(f"  ✓ Frame shape: {frame.shape}")
        print(f"  ✓ Frame dtype: {frame.dtype}")
        
        # Test 3: Batch reading
        print("\nTest 3: Batch reading...")
        indices = list(range(min(10, num_frames)))
        frames = vr.get_batch(indices)
        print(f"  ✓ Batch shape: {frames.shape}")
        print(f"  ✓ Expected: ({len(indices)}, H, W, 3)")
        
        del vr
        
        # Test 4: With resizing
        print("\nTest 4: With resizing...")
        vr = PyVideoReader(
            video_path,
            target_width=224,
            target_height=224,
            oob_mode="skip",
            threads=0
        )
        frame = vr[0]
        print(f"  ✓ Resized frame shape: {frame.shape}")
        assert frame.shape[:2] == (224, 224), f"Expected (224, 224), got {frame.shape[:2]}"
        print(f"  ✓ Resize working correctly")
        
        del vr
        
        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("video_reader-rs is working correctly as a decord replacement.")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_video_reader()
