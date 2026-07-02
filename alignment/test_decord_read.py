#!/usr/bin/env python
import decord

video_path = "/open_data/AgiBotWorld-Beta/327/327_648642/faceImg.mp4"

print(f"Testing decord reading: {video_path}")
print("-" * 60)

try:
    # Set bridge to native
    decord.bridge.set_bridge('native')
    print("✓ Set decord bridge to 'native'")
    
    # Try to open the video
    vr = decord.VideoReader(video_path)
    print(f"✓ Successfully opened video with decord")
    print(f"  - Number of frames: {len(vr)}")
    print(f"  - FPS: {vr.get_avg_fps()}")
    
    # Try to read first frame
    frame = vr[0]
    print(f"✓ Successfully read first frame")
    print(f"  - Frame shape: {frame.shape}")
    print(f"  - Frame dtype: {frame.dtype}")
    
    # Try to read a batch of frames
    indices = [0, 1, 2]
    frames = vr.get_batch(indices)
    print(f"✓ Successfully read batch of {len(indices)} frames")
    print(f"  - Batch shape: {frames.shape}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: decord can read this video file!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n" + "=" * 60)
    print("ERROR: decord failed to read this video file")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
