#!/usr/bin/env python3
"""
Multi-session data processor - Auto mode
Tự động xử lý tất cả sessions có markers
"""

import os
import json
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import glob
import sys

# ==== CONFIG ====
DATASET_ROOT = "/home/kuan/Work_Space/Thuc_tap/recorded_dataset"
OUTPUT_PATH = "processed_data_auto"
YOLO_MODEL_PATH = "/home/kuan/Work_Space/Thuc_tap/Human_action/best.pt"

# Tạo thư mục output
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "frames"), exist_ok=True)

def main():
    print("🚀 AUTO MULTI-SESSION PROCESSOR")
    print("="*50)
    
    # Load YOLO model
    print("🔄 Loading YOLO model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Using device: {device}")
    
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO model loaded!")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return
    
    # Find all sessions
    all_sessions = sorted(glob.glob(os.path.join(DATASET_ROOT, "session_*")))
    print(f"🔍 Found {len(all_sessions)} sessions")
    
    # Filter sessions with markers
    sessions_with_markers = []
    for session_path in all_sessions:
        markers_path = os.path.join(session_path, "markers.json")
        if os.path.exists(markers_path):
            with open(markers_path, 'r') as f:
                markers = json.load(f)
            if markers:  # Has markers
                sessions_with_markers.append(session_path)
    
    print(f"📍 Found {len(sessions_with_markers)} sessions with markers")
    
    if not sessions_with_markers:
        print("❌ No sessions with markers found!")
        return
    
    # Process sessions
    all_entries = []
    global_frame_id = 0
    
    for i, session_path in enumerate(sessions_with_markers, 1):
        session_name = os.path.basename(session_path)
        print(f"\n📁 [{i}/{len(sessions_with_markers)}] Processing: {session_name}")
        
        try:
            # Load session data
            with open(os.path.join(session_path, "metadata.json"), "r") as f:
                metadata = json.load(f)
            with open(os.path.join(session_path, "markers.json"), "r") as f:
                markers = json.load(f)
            with open(os.path.join(session_path, "timestamps.json"), "r") as f:
                timestamps = json.load(f)
            
            print(f"   📊 {len(timestamps)} frames, {len(markers)} markers")
            
            # Process frames
            session_count = 0
            for frame_info in timestamps:
                frame_idx = frame_info["frame_idx"]
                ts = frame_info["ts_system"]
                
                # Get action label from markers
                action_label = "None"
                for marker in sorted(markers, key=lambda x: x["timestamp"]):
                    if ts >= marker["timestamp"]:
                        action_label = marker["label"]
                
                # File paths - convert relative to absolute
                rgb_rel_path = frame_info["rgb_path"]
                depth_rel_path = frame_info["depth_path"]
                
                # Convert to absolute paths
                if rgb_rel_path.startswith("recorded_dataset/"):
                    rgb_path = os.path.join("/home/kuan/Work_Space/Thuc_tap", rgb_rel_path)
                    depth_path = os.path.join("/home/kuan/Work_Space/Thuc_tap", depth_rel_path)
                else:
                    rgb_path = rgb_rel_path
                    depth_path = depth_rel_path
                
                pose_path = os.path.join(session_path, "pose", f"pose_{frame_idx:06d}.json")
                
                if not os.path.exists(rgb_path):
                    print(f"   ⚠️ Missing RGB: {rgb_path}")
                    continue
                
                # Process RGB with YOLO
                rgb_img = cv2.imread(rgb_path)
                if rgb_img is None:
                    continue
                
                # YOLO inference
                yolo_results = yolo_model.predict(rgb_img, verbose=False)
                if yolo_results and yolo_results[0].masks is not None:
                    mask = yolo_results[0].masks.data[0].cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
                
                # Save processed data
                new_rgb = os.path.join(OUTPUT_PATH, "frames", f"rgb_{global_frame_id:06d}.png")
                new_depth = os.path.join(OUTPUT_PATH, "frames", f"depth_{global_frame_id:06d}.npy")
                new_pose = os.path.join(OUTPUT_PATH, "frames", f"pose_{global_frame_id:06d}.json")
                new_mask = os.path.join(OUTPUT_PATH, "frames", f"mask_{global_frame_id:06d}.npy")
                
                cv2.imwrite(new_rgb, rgb_img)
                np.save(new_mask, mask)
                
                if os.path.exists(depth_path):
                    depth_data = np.load(depth_path)
                    np.save(new_depth, depth_data)
                
                # Pose data
                pose_data = {"pose2d": [], "pose3d": []}
                if os.path.exists(pose_path):
                    with open(pose_path, 'r') as f:
                        pose_data = json.load(f)
                
                with open(new_pose, 'w') as f:
                    json.dump(pose_data, f)
                
                # Add to dataset
                all_entries.append({
                    "frame_id": global_frame_id,
                    "session": session_name,
                    "original_frame": frame_idx,
                    "timestamp": ts,
                    "action": action_label,
                    "rgb_path": new_rgb,
                    "depth_path": new_depth,
                    "pose_path": new_pose,
                    "mask_path": new_mask,
                    "has_pose": len(pose_data.get("pose2d", [])) > 0
                })
                
                global_frame_id += 1
                session_count += 1
            
            print(f"   ✅ Processed {session_count} frames")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Save final dataset
    dataset_path = os.path.join(OUTPUT_PATH, "dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(all_entries, f, indent=2)
    
    # Create summary
    actions_count = {}
    pose_count = 0
    for entry in all_entries:
        action = entry["action"]
        actions_count[action] = actions_count.get(action, 0) + 1
        if entry["has_pose"]:
            pose_count += 1
    
    summary = {
        "total_frames": len(all_entries),
        "sessions_processed": len(sessions_with_markers),
        "actions": actions_count,
        "frames_with_pose": pose_count
    }
    
    with open(os.path.join(OUTPUT_PATH, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("🎉 PROCESSING COMPLETED!")
    print("="*60)
    print(f"📊 Total frames: {len(all_entries):,}")
    print(f"📁 Sessions: {len(sessions_with_markers)}")
    print(f"💾 Output: {OUTPUT_PATH}")
    print(f"\n📊 Actions:")
    for action, count in actions_count.items():
        print(f"   {action}: {count:,}")
    if len(all_entries) > 0:
        print(f"\n🤸 Frames with pose: {pose_count:,} ({pose_count/len(all_entries)*100:.1f}%)")
    else:
        print(f"\n❌ No frames processed! Check file paths.")

if __name__ == "__main__":
    main()