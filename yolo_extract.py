import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Test dữ liệu đã được xử lý
PROCESSED_DATA_ROOT = "/home/kuan/Work_Space/Thuc_tap/processed_data_auto"
FRAMES_DIR = os.path.join(PROCESSED_DATA_ROOT, "frames")
DATASET_FILE = os.path.join(PROCESSED_DATA_ROOT, "dataset.json")

import sys
import argparse

def load_dataset_or_single_image():
    """Load dataset hoặc single image từ command line args"""
    parser = argparse.ArgumentParser(description='Test processed data visualization')
    parser.add_argument('--image', '-i', type=str, help='Path to single RGB image to test')
    parser.add_argument('--depth', '-d', type=str, help='Path to corresponding depth file (.npy)')
    parser.add_argument('--pose', '-p', type=str, help='Path to corresponding pose file (.json)')
    parser.add_argument('--mask', '-m', type=str, help='Path to corresponding mask file (.npy)')
    parser.add_argument('--max_frames', type=int, default=10, help='Max frames from dataset (default: 10)')
    
    args = parser.parse_args()
    
    if args.image:
        # Single image mode
        print(f"🖼️ Single image mode: {args.image}")
        
        # Create single frame data structure
        single_frame = {
            "frame_id": 0,
            "session": "manual_input",
            "action": "Unknown", 
            "rgb_path": args.image,
            "depth_path": args.depth or "",
            "pose_path": args.pose or "",
            "mask_path": args.mask or "",
            "has_pose": bool(args.pose)
        }
        
        return [single_frame], 1
    else:
        # Dataset mode
        print("🔄 Loading processed dataset...")
        with open(DATASET_FILE, 'r') as f:
            full_dataset = json.load(f)
        
        dataset = full_dataset[:args.max_frames] if len(full_dataset) > args.max_frames else full_dataset
        print(f"✅ Dataset loaded! Testing on {len(dataset)} frames (out of {len(full_dataset)} total)")
        
        return dataset, len(full_dataset)

# Load data based on command line arguments
dataset, total_frames = load_dataset_or_single_image()

def draw_pose_keypoints(img, pose_data, color=(0, 255, 0)):
    """Vẽ hand pose keypoints lên ảnh (MediaPipe Hands format)"""
    if not pose_data.get('pose2d') or len(pose_data['pose2d']) == 0:
        return img
    
    # MediaPipe hand connections (21 keypoints per hand)
    hand_connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger  
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky finger
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    # Vẽ từng bàn tay
    for hand_idx, hand_keypoints in enumerate(pose_data['pose2d']):
        # Chọn màu khác nhau cho mỗi tay
        hand_color = color if hand_idx == 0 else (255, 0, 0)  # Green for first hand, Red for second
        
        # Vẽ connections
        for start_idx, end_idx in hand_connections:
            if start_idx < len(hand_keypoints) and end_idx < len(hand_keypoints):
                start_point = hand_keypoints[start_idx]
                end_point = hand_keypoints[end_idx]
                
                # Format: [x, y, confidence]
                if len(start_point) >= 3 and len(end_point) >= 3:
                    if start_point[2] > 0.3 and end_point[2] > 0.3:  # Confidence threshold
                        pt1 = (int(start_point[0]), int(start_point[1]))
                        pt2 = (int(end_point[0]), int(end_point[1]))
                        cv2.line(img, pt1, pt2, hand_color, 2)
        
        # Vẽ keypoints
        for i, point in enumerate(hand_keypoints):
            if len(point) >= 3 and point[2] > 0.3:  # Confidence threshold
                center = (int(point[0]), int(point[1]))
                cv2.circle(img, center, 4, hand_color, -1)
                cv2.putText(img, str(i), (center[0]+5, center[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, hand_color, 1)
    
    return img

def create_depth_colormap(depth_array):
    """Tạo depth colormap từ depth array"""
    # Normalize depth values
    depth_normalized = np.nan_to_num(depth_array, nan=0.0)
    
    # Scale to 0-255 for visualization
    if depth_normalized.max() > 0:
        depth_vis = (depth_normalized / depth_normalized.max() * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth_normalized, dtype=np.uint8)
    
    # Apply colormap
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth_colormap

def visualize_frame(frame_data):
    """Hiển thị tất cả thông tin của một frame"""
    frame_id = frame_data['frame_id']
    action = frame_data['action']
    session = frame_data['session']
    
    print(f"\n🖼️ Frame {frame_id} - Action: {action} - Session: {session}")
    
    # Load RGB image
    rgb_path = frame_data['rgb_path']
    if not rgb_path.startswith('/') and not os.path.isabs(rgb_path):
        rgb_path = os.path.join("/home/kuan/Work_Space/Thuc_tap", rgb_path)
    
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        print(f"❌ Cannot load RGB image: {rgb_path}")
        return
    
    # Load depth data
    depth_path = frame_data['depth_path']
    depth_colormap = np.zeros_like(rgb_img)
    depth_data = None
    
    if depth_path and os.path.exists(depth_path):
        try:
            depth_data = np.load(depth_path)
            depth_colormap = create_depth_colormap(depth_data)
        except Exception as e:
            print(f"⚠️ Cannot load depth data: {depth_path} - {e}")
    elif depth_path and not depth_path.startswith('/') and not os.path.isabs(depth_path):
        depth_path_full = os.path.join("/home/kuan/Work_Space/Thuc_tap", depth_path)
        try:
            depth_data = np.load(depth_path_full)
            depth_colormap = create_depth_colormap(depth_data)
        except Exception as e:
            print(f"⚠️ Cannot load depth data: {depth_path_full} - {e}")
    else:
        print(f"⚠️ No depth data provided, using placeholder")
    
    # Load mask data
    mask_path = frame_data['mask_path']
    mask_colored = np.zeros_like(rgb_img)
    mask_data = None
    
    if mask_path and os.path.exists(mask_path):
        try:
            mask_data = np.load(mask_path)
            mask_colored = np.zeros_like(rgb_img)
            mask_colored[:, :, 2] = mask_data  # Red channel for mask
        except Exception as e:
            print(f"⚠️ Cannot load mask data: {mask_path} - {e}")
    elif mask_path and not mask_path.startswith('/') and not os.path.isabs(mask_path):
        mask_path_full = os.path.join("/home/kuan/Work_Space/Thuc_tap", mask_path)
        try:
            mask_data = np.load(mask_path_full)
            mask_colored = np.zeros_like(rgb_img)
            mask_colored[:, :, 2] = mask_data  # Red channel for mask
        except Exception as e:
            print(f"⚠️ Cannot load mask data: {mask_path_full} - {e}")
    else:
        print(f"⚠️ No mask data provided, using placeholder")
    
    # Load pose data
    pose_path = frame_data['pose_path']
    pose_data = {"pose2d": [], "pose3d": []}
    
    if pose_path and os.path.exists(pose_path):
        try:
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Cannot load pose data: {pose_path} - {e}")
    elif pose_path and not pose_path.startswith('/') and not os.path.isabs(pose_path):
        pose_path_full = os.path.join("/home/kuan/Work_Space/Thuc_tap", pose_path)
        try:
            with open(pose_path_full, 'r') as f:
                pose_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Cannot load pose data: {pose_path_full} - {e}")
    else:
        print(f"⚠️ No pose data provided, using placeholder")
    
    # Create visualization
    rgb_with_pose = rgb_img.copy()
    if pose_data.get('pose2d'):
        rgb_with_pose = draw_pose_keypoints(rgb_with_pose, pose_data)
    
    # Create overlay with mask
    rgb_with_mask = rgb_img.copy()
    if mask_data is not None and np.any(mask_data):
        mask_overlay = cv2.addWeighted(rgb_img, 0.7, mask_colored, 0.3, 0)
        rgb_with_mask = mask_overlay
    
    # Resize images for display
    height = 400
    scale = height / rgb_img.shape[0]
    width = int(rgb_img.shape[1] * scale)
    
    rgb_display = cv2.resize(rgb_img, (width, height))
    pose_display = cv2.resize(rgb_with_pose, (width, height))
    mask_display = cv2.resize(rgb_with_mask, (width, height))
    depth_display = cv2.resize(depth_colormap, (width, height))
    
    # Add labels
    label_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(rgb_display, "Original RGB", (10, 30), font, 0.7, label_color, 2)
    cv2.putText(pose_display, f"Pose ({len(pose_data.get('pose2d', []))} keypoints)", 
                (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(mask_display, "YOLO Mask Overlay", (10, 30), font, 0.7, (0, 0, 255), 2)
    cv2.putText(depth_display, "Depth Map", (10, 30), font, 0.7, label_color, 2)
    
    # Combine images into grid
    top_row = np.hstack([rgb_display, pose_display])
    bottom_row = np.hstack([mask_display, depth_display])
    combined = np.vstack([top_row, bottom_row])
    
    # Add frame info
    info_text = f"Frame {frame_id} | Action: {action} | Session: {session}"
    cv2.putText(combined, info_text, (10, combined.shape[0] - 20), 
                font, 0.8, (0, 255, 255), 2)
    
    # Display
    cv2.imshow("Data Visualization", combined)
    
    # Print statistics
    pose_count = len(pose_data.get('pose2d', []))
    mask_pixels = np.count_nonzero(mask_data) if mask_data is not None else 0
    depth_valid = np.count_nonzero(~np.isnan(depth_data)) if depth_data is not None else 0
    
    print(f"   📊 Pose keypoints: {pose_count}")
    print(f"   📊 Mask pixels: {mask_pixels}")
    print(f"   📊 Valid depth pixels: {depth_valid}")
    print(f"   📊 Has pose data: {frame_data.get('has_pose', False)}")

def main():
    """Main function to test processed data"""
    print("🎯 PROCESSED DATA VISUALIZATION TOOL")
    print("="*50)
    
    if len(dataset) == 1 and dataset[0]['session'] == 'manual_input':
        print("🔧 Single image mode detected")
        print("Usage examples:")
        print("  python yolo_extract.py --image /path/to/image.jpg")
        print("  python yolo_extract.py -i image.jpg -d depth.npy -p pose.json -m mask.npy")
        print("")
    
    # Filter frames with different actions
    actions = {}
    for frame in dataset:
        action = frame['action']
        if action not in actions:
            actions[action] = []
        actions[action].append(frame)
    
    print(f"\n� Available actions:")
    for action, frames in actions.items():
        pose_frames = sum(1 for f in frames if f.get('has_pose', False))
        print(f"   {action}: {len(frames)} frames ({pose_frames} with pose)")
    
    print(f"\n🎮 Controls:")
    print("  [SPACE] - Next random frame")
    print("  [1-5] - Show specific action type")
    print("  [p] - Show frame with pose data")
    print("  [q] - Quit")
    
    current_frames = dataset
    frame_index = 0
    
    try:
        while True:
            if not current_frames:
                print("❌ No frames available")
                break
            
            # Show current frame
            frame = current_frames[frame_index % len(current_frames)]
            visualize_frame(frame)
            
            # Handle input
            key = cv2.waitKey(0) & 0xFF
            print(f"🔍 Key pressed: {key} ('{chr(key) if 32 <= key <= 126 else '?'}')")  # Debug
            
            if key == ord('q') or key == 27:  # 'q' hoặc ESC
                print("👋 Exiting...")
                break
            elif key == ord(' '):  # Space - random frame
                frame_index = random.randint(0, len(current_frames) - 1)
            elif key == ord('p'):  # Show frame with pose
                pose_frames = [f for f in dataset if f.get('has_pose', False)]
                if pose_frames:
                    current_frames = pose_frames
                    frame_index = 0
                    print(f"🤸 Switched to pose frames ({len(pose_frames)} frames)")
            elif key >= ord('1') and key <= ord('5'):  # Action selection
                action_names = list(actions.keys())
                action_idx = key - ord('1')
                if action_idx < len(action_names):
                    action_name = action_names[action_idx]
                    current_frames = actions[action_name]
                    frame_index = 0
                    print(f"🎯 Switched to {action_name} frames ({len(current_frames)} frames)")
            else:
                frame_index = (frame_index + 1) % len(current_frames)
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Ensure windows are properly destroyed
        print("👋 Visualization completed!")

if __name__ == "__main__":
    import random
    main()