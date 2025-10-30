#!/usr/bin/env python3
import os, cv2, torch, numpy as np
from tqdm import tqdm
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
import torch.nn.functional as F

from models import ObjectEncoder, PoseGNN_PyG, FusionLSTM, get_hand_edges
from preprocess_ultils import compute_kin_features

# ------------------------------------------
# Utilities
# ------------------------------------------
def load_checkpoint(model, ckpt_path, key_list=["model","encoder","pose_gnn","state_dict"]):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    for k in key_list:
        if k in ckpt:
            try:
                model.load_state_dict(ckpt[k])
                print(f"[Load] ✅ Loaded '{k}' from {os.path.basename(ckpt_path)}")
                return
            except Exception as e:
                print(f"[Load] ⚠️ Key '{k}' failed: {e}")
    model.load_state_dict(ckpt)
    print(f"[Load] ✅ Direct loaded from {os.path.basename(ckpt_path)}")

def draw_overlay(frame, label, conf, color=(0,255,0)):
    cv2.putText(frame, f"{label} ({conf:.2f})", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return frame

# ------------------------------------------
# Main inference
# ------------------------------------------
@torch.no_grad()
def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Test] Using device: {device}")

    # Models
    obj_enc = ObjectEncoder(out_dim=args.obj_dim, pretrained=False).to(device).eval()
    pose_enc = PoseGNN_PyG(in_dim=3, hidden=args.hidden, out_dim=args.pose_dim, model_type=args.gnn_type).to(device).eval()
    fusion = FusionLSTM(input_dim=args.obj_dim+args.pose_dim+args.kin_dim,
                        hidden_dim=args.hidden, num_classes=args.num_classes,
                        bidirectional=False).to(device).eval()

    load_checkpoint(obj_enc, args.obj_ckpt)
    load_checkpoint(pose_enc, args.pose_ckpt)
    load_checkpoint(fusion, args.stageC_ckpt)

    # Detector and pose estimator
    yolo = YOLO(args.yolo_ckpt)
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    edges = get_hand_edges().to(device)

    rgb_dir = os.path.join(args.session_path, "rgb")
    depth_dir = os.path.join(args.session_path, "depth")
    frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png") or f.endswith(".jpg")])
    print(f"[Session] {args.session_path}, total {len(frames)} frames")

    out_path = os.path.join(args.out_dir, f"{os.path.basename(args.session_path)}.mp4")
    os.makedirs(args.out_dir, exist_ok=True)

    writer = None
    window = deque(maxlen=args.seq_len)

    for fname in tqdm(frames, desc="[Processing]"):
        rgb_path = os.path.join(rgb_dir, fname)
        depth_path = os.path.join(depth_dir, fname.replace("frame_", "depth_").replace(".png", ".npy"))

        frame = cv2.imread(rgb_path)
        h, w, _ = frame.shape
        depth = np.load(depth_path) if os.path.exists(depth_path) else np.zeros((h,w), dtype=np.float32)

        # (1) Detect object
        results = yolo.predict(frame, verbose=False)
        obj_centroid = np.zeros(3, dtype=np.float32)
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            z = depth[int(cy), int(cx)] if 0 <= int(cy) < h and 0 <= int(cx) < w else 0
            obj_centroid[:] = [cx/w, cy/h, z/1000.0]  # chuẩn hóa
            crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            obj_img = cv2.resize(crop, (224,224))
        else:
            obj_img = cv2.resize(frame, (224,224))

        # (2) Mediapipe Hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_hands.process(rgb)
        hand_kps3d = np.zeros((21,3), dtype=np.float32)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            for i, pt in enumerate(lm.landmark):
                x, y = int(pt.x*w), int(pt.y*h)
                z = depth[y, x] if 0 <= y < h and 0 <= x < w else 0
                hand_kps3d[i] = [pt.x, pt.y, z/1000.0]

        kin_feat = compute_kin_features(obj_centroid, hand_kps3d)

        # (3) Encode
        obj_tensor = torch.from_numpy(obj_img).permute(2,0,1).unsqueeze(0).float().to(device)/255.
        pose_tensor = torch.from_numpy(hand_kps3d).unsqueeze(0).float().to(device)
        kin_tensor = torch.from_numpy(kin_feat).unsqueeze(0).float().to(device)

        obj_feat = obj_enc(obj_tensor).squeeze(0)
        pose_feat = pose_enc(pose_tensor[0], edges)
        if pose_feat.ndim == 2: pose_feat = pose_feat.mean(0)
        fused = torch.cat([obj_feat, pose_feat, kin_tensor.squeeze(0)], dim=-1)
        window.append(fused)

        # (4) Predict
        if len(window) == args.seq_len:
            seq_tensor = torch.stack(list(window), dim=0).unsqueeze(0).to(device)
            out = fusion(seq_tensor)
            probs = F.softmax(out, dim=-1)[0]
            cls = torch.argmax(probs).item()
            conf = probs[cls].item()
            frame = draw_overlay(frame, args.class_names[cls], conf)

        # (5) Write video
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, 20, (w,h))
        writer.write(frame)

    if writer: writer.release()
    print(f"[✅ Saved] {out_path}")

# ------------------------------------------
# CLI
# ------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--session_path", type=str, required=True, help="folder contains rgb/ and depth/")
    p.add_argument("--obj_ckpt", type=str, required=True)
    p.add_argument("--pose_ckpt", type=str, required=True)
    p.add_argument("--stageC_ckpt", type=str, required=True)
    p.add_argument("--yolo_ckpt", type=str, default="yolov8n.pt")
    p.add_argument("--out_dir", type=str, default="outputs/test_videos")
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--obj_dim", type=int, default=128)
    p.add_argument("--pose_dim", type=int, default=128)
    p.add_argument("--kin_dim", type=int, default=4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--gnn_type", choices=["gcn","sage"], default="gcn")
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--class_names", nargs="+", default=["Approach","Grasp","None","Release","Transport"])
    args = p.parse_args()

    run_inference(args)
