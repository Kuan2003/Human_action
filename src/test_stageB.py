#!/usr/bin/env python3
"""
Test Stage B - Extract pose embeddings from trained PoseGNN
"""

import os
import torch
import numpy as np
from tqdm import tqdm

from datasets import HOISequenceDataset
from preprocess_ultils import load_pose_json, load_depth, keypoint_2d_to_3d
import models as models_module


# --------------------------- Dataset ---------------------------
class PoseTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_json, intrinsics=None):
        base_ds = HOISequenceDataset(dataset_json, seq_len=8, stride=8,
                                     intrinsics=intrinsics, drop_no_pose=True)
        self.base_dir = os.path.dirname(os.path.abspath(dataset_json))
        if self.base_dir.endswith('processed_data_auto'):
            self.base_dir = os.path.dirname(self.base_dir)
        self.samples = []
        for sess, frames in base_ds.windows:
            for f in frames:
                if f.get("has_pose", True):
                    self.samples.append(f)
        self.label2id = base_ds.label2id
        self.id2label = base_ds.id2label
        self.intrinsics = base_ds.intrinsics
        print(f"[TestB] Loaded {len(self.samples)} pose frames")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f = self.samples[idx]
        pose_path = f["pose_path"]
        if not os.path.isabs(pose_path):
            pose_path = os.path.join(self.base_dir, pose_path)
        depth_path = f["depth_path"]
        if not os.path.isabs(depth_path):
            depth_path = os.path.join(self.base_dir, depth_path)
        pose2d = load_pose_json(pose_path)
        depth = load_depth(depth_path)
        pose3d = keypoint_2d_to_3d(pose2d, depth, self.intrinsics)
        label = f.get("action", "None")
        label_id = self.label2id.get(label, self.label2id["None"])
        return torch.tensor(pose3d, dtype=torch.float32), label_id


# --------------------------- Utility ---------------------------
def get_edges():
    f = getattr(models_module, "get_hand_edges", None)
    if f:
        return f()
    import torch
    return torch.tensor([
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20]
    ], dtype=torch.long).t()


# --------------------------- Main ---------------------------
def extract_embeddings(args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    ds = PoseTestDataset(args.dataset_json)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    edges = get_edges().to(device)

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device)
    PoseGNN = getattr(models_module, "PoseGNN_PyG", None)
    pose_gnn = PoseGNN(in_dim=3, hidden=args.hidden, out_dim=args.out_dim, model_type=args.gnn_type).to(device)
    pose_gnn.load_state_dict(ckpt["pose_gnn"])
    pose_gnn.eval()

    all_feats, all_labels = [], []
    with torch.no_grad():
        for pose3d, label in tqdm(loader, desc="[TestB] Extracting"):
            pose3d = pose3d.to(device)
            feats = pose_gnn(pose3d, edges)  # [B, out_dim]
            all_feats.append(feats.cpu().numpy())
            all_labels.append(label.numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"[TestB] Extracted {all_feats.shape[0]} embeddings of dim {all_feats.shape[1]}")

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "pose_embeddings.npy"), all_feats)
    np.save(os.path.join(args.out_dir, "pose_labels.npy"), all_labels)
    print(f"[TestB] Saved embeddings to {args.out_dir}")


# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json", required=True, help="Path to dataset.json")
    p.add_argument("--ckpt", required=True, help="Checkpoint from Stage B (.pth)")
    p.add_argument("--out_dir", default="outputs/stageB_embeddings")
    p.add_argument("--gnn_type", choices=["gcn", "sage"], default="sage")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()
    extract_embeddings(args)
