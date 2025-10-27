import os
import json
import cv2
import numpy as np
import glob
from pathlib import Path
import mediapipe as mp

# Setup MediaPipe
mp_hands = mp.solutions.hands

class HandPoseDetector:
    def __init__(self):
        """
        Chỉ sử dụng MediaPipe Hands để detect bàn tay
        """
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_hands(self, rgb_image):
        """Detect hands using MediaPipe Hands"""
        results = self.hands.process(rgb_image)
        
        pose2d = []
        pose3d = []
        
        if results.multi_hand_landmarks:
            h, w, _ = rgb_image.shape
            
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[i].classification[0].label  # 'Left' or 'Right'
                
                # Convert landmarks to pixel coordinates (2D)
                hand_pose2d = []
                for landmark in hand_landmarks.landmark:
                    x = landmark.x * w
                    y = landmark.y * h
                    hand_pose2d.append([x, y, 1.0])  # [x, y, confidence]
                
                pose2d.append(hand_pose2d)
                
                # For 3D, we use MediaPipe's relative Z
                hand_pose3d = []
                for landmark in hand_landmarks.landmark:
                    x = landmark.x * w
                    y = landmark.y * h
                    z = landmark.z  # Relative depth from MediaPipe
                    hand_pose3d.append([x, y, z, 1.0])  # [x, y, z, confidence]
                
                pose3d.append(hand_pose3d)
        
        return pose2d, pose3d
    
    def process_image(self, rgb_image):
        """Process một ảnh và trả về pose data theo format cũ"""
        # Convert BGR to RGB for MediaPipe  
        rgb_image_mp = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        pose2d, pose3d = self.detect_hands(rgb_image_mp)
        
        return {
            "pose2d": pose2d,
            "pose3d": pose3d
        }

def sample_depth_median(depth_m, x, y, win=5):
    """Sample depth value at (x,y) with median filter"""
    if depth_m is None:
        return None
        
    H, W = depth_m.shape
    r = win // 2
    x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[~np.isnan(patch)]
    if valid.size == 0: 
        return None
    return float(np.median(valid))

def enhance_pose_with_depth(pose_data, depth_data):
    """Enhance pose with real depth values"""
    if depth_data is None:
        return pose_data
    
    # Enhance pose3d with real depth values
    enhanced_pose3d = []
    for hand_pose3d in pose_data["pose3d"]:
        enhanced_hand = []
        for kp in hand_pose3d:
            x, y, z_rel, conf = kp
            # Get real depth from depth image
            z_depth = sample_depth_median(depth_data, x, y)
            if z_depth is not None:
                enhanced_hand.append([x, y, z_depth, conf])  # Use real depth
            else:
                enhanced_hand.append([x, y, z_rel, conf])     # Fallback to relative
        enhanced_pose3d.append(enhanced_hand)
    
    pose_data["pose3d"] = enhanced_pose3d
    return pose_data

def reprocess_session(session_path, detector):
    """Reprocess một session với hand detector mới"""
    session_name = os.path.basename(session_path)
    print(f"\n📁 Reprocessing session: {session_name}")
    
    # Check directories
    rgb_dir = os.path.join(session_path, "rgb")
    depth_dir = os.path.join(session_path, "depth")
    pose_dir = os.path.join(session_path, "pose")
    
    if not os.path.exists(rgb_dir):
        print(f"   ❌ No RGB directory found")
        return 0
    
    # Get RGB files
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "frame_*.png")))
    print(f"   📊 Found {len(rgb_files)} RGB frames")
    
    if not rgb_files:
        return 0
    
    processed_count = 0
    
    for rgb_file in rgb_files:
        # Extract frame number
        frame_name = os.path.basename(rgb_file)
        frame_idx = int(frame_name.split('_')[1].split('.')[0])
        
        # Load RGB image
        rgb_image = cv2.imread(rgb_file)
        if rgb_image is None:
            continue
        
        # Load corresponding depth (optional)
        depth_file = os.path.join(depth_dir, f"depth_{frame_idx:06d}.npy")
        depth_data = None
        if os.path.exists(depth_file):
            try:
                depth_data = np.load(depth_file)
            except:
                pass
        
        # Process with hand detector
        pose_data = detector.process_image(rgb_image)
        
        # Enhance with real depth if available
        pose_data = enhance_pose_with_depth(pose_data, depth_data)
        
        # Save new pose file (overwrite old one)
        pose_file = os.path.join(pose_dir, f"pose_{frame_idx:06d}.json")
        with open(pose_file, 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        processed_count += 1
        
        # Progress update
        if processed_count % 50 == 0:
            print(f"   🔄 Processed {processed_count}/{len(rgb_files)} frames")
    
    print(f"   ✅ Reprocessed {processed_count} frames")
    return processed_count

def main():
    print("🔄 HAND POSE RE-DETECTION TOOL")
    print("="*60)
    
    # Setup detector
    detector = HandPoseDetector()
    
    # Find all sessions
    dataset_root = "/home/kuan/Work_Space/Thuc_tap/recorded_dataset"
    sessions = sorted(glob.glob(os.path.join(dataset_root, "session_*")))
    
    print(f"🔍 Found {len(sessions)} sessions to reprocess")
    
    # Auto proceed
    print(f"\n⚠️  This will OVERWRITE existing pose data!")
    print(f"📁 Dataset path: {dataset_root}")
    print(f"🤲 Using: MediaPipe Hands (21 keypoints per hand)")
    print(f"🚀 Starting automatic reprocessing...")
    
    # Process all sessions
    total_frames = 0
    successful_sessions = 0
    
    for i, session_path in enumerate(sessions, 1):
        try:
            print(f"\n[{i}/{len(sessions)}] Processing {os.path.basename(session_path)}")
            frames_processed = reprocess_session(session_path, detector)
            total_frames += frames_processed
            if frames_processed > 0:
                successful_sessions += 1
        except Exception as e:
            print(f"   ❌ Error processing session: {e}")
            continue
    
    # Summary
    print(f"\n" + "="*60)
    print(f"🎉 REPROCESSING COMPLETED!")
    print(f"="*60)
    print(f"📊 Sessions processed: {successful_sessions}/{len(sessions)}")
    print(f"🖼️  Total frames reprocessed: {total_frames}")
    print(f"🤲 Using: MediaPipe Hands (21 keypoints per hand)")
    print(f"💾 All pose files have been updated with new hand data")
    
    # Show example of new data structure
    if sessions:
        example_session = sessions[0]
        example_pose = os.path.join(example_session, "pose", "pose_000000.json")
        if os.path.exists(example_pose):
            print(f"\n📄 Example new pose data structure:")
            with open(example_pose, 'r') as f:
                data = json.load(f)
            print(f"   - pose2d: {len(data.get('pose2d', []))} hands detected (2D keypoints)")
            print(f"   - pose3d: {len(data.get('pose3d', []))} hands detected (3D with depth)")

if __name__ == "__main__":
    main()