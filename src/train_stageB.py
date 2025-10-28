#!/usr/bin/env python3
"""
Stage B - Train Pose GNN Encoder (GCN or GraphSAGE)
- Input: hand 3D keypoints [21, 3]
- Output: action classification + learned embedding
- Flatten sequence dataset into single-frame pose samples
"""

import os
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import HOISequenceDataset
from preprocess_ultils import load_pose_json, keypoint_2d_to_3d, load_depth
import models as models_module
from environment_config import setup_environment


# ===================== Dataset Wrapper =====================
class SingleFramePoseDataset(torch.utils.data.Dataset):
    """
    Flatten HOISequenceDataset windows -> single frame pose samples
    """
    def __init__(self, dataset_json, intrinsics=None, drop_no_pose=True):
        # Use HOISequenceDataset which has smart path resolution
        base_ds = HOISequenceDataset(dataset_json, seq_len=8, stride=8,
                                     intrinsics=intrinsics, drop_no_pose=drop_no_pose)
        
        self.samples = []
        for sess, frames in base_ds.windows:
            for f in frames:
                if not f.get("has_pose", True):
                    continue
                self.samples.append(f)
        self.label2id = base_ds.label2id
        self.id2label = base_ds.id2label
        self.intrinsics = base_ds.intrinsics

        print(f"[B] Flattened to {len(self.samples)} pose frames from {len(base_ds.windows)} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f = self.samples[idx]

        # Paths are already resolved by HOISequenceDataset
        pose_path = f["pose_path"]
        depth_path = f["depth_path"]

        pose2d = load_pose_json(pose_path)
        depth = load_depth(depth_path)
        pose3d = keypoint_2d_to_3d(pose2d, depth, self.intrinsics)  # [21,3]

        label = f.get("action", "None")
        label_id = self.label2id.get(label, self.label2id["None"])
        return torch.tensor(pose3d, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)


# ===================== Hand Skeleton Edges =====================
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


# ===================== Utils =====================
def session_split(ds, val_ratio=0.2, seed=123):
    total = len(ds)
    idx = list(range(total))
    random.Random(seed).shuffle(idx)
    val_count = int(total * val_ratio)
    return idx[val_count:], idx[:val_count]


# ===================== Training =====================
def train(args):
    setup_environment()
    
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    ds = SingleFramePoseDataset(args.dataset_json)
    print(f"[B] Dataset loaded with {len(ds)} frames, classes: {ds.label2id}")

    train_idx, val_idx = session_split(ds, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = DataLoader(ds, batch_size=args.batch, sampler=SubsetRandomSampler(train_idx), num_workers=args.workers)
    val_loader = DataLoader(ds, batch_size=args.batch, sampler=SubsetRandomSampler(val_idx), num_workers=args.workers)

    edges = get_edges().to(device)
    PoseGNN = getattr(models_module, "PoseGNN_PyG", None)
    pose_gnn = PoseGNN(in_dim=3, hidden=args.hidden, out_dim=args.out_dim, model_type=args.gnn_type).to(device)
    head = nn.Linear(args.out_dim, len(ds.label2id)).to(device)

    params = list(pose_gnn.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb")) if args.tensorboard else None
    best = -1.0

    # ===================== TRAIN LOOP =====================
    for epoch in range(args.epochs):
        pose_gnn.train(); head.train()
        running_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"[B] Epoch {epoch}")

        for pose3d, label in pbar:
            pose3d, label = pose3d.to(device), label.to(device)
            feats = pose_gnn(pose3d, edges)     # [B, out_dim]
            logits = head(feats)                # [B, num_classes]
            loss = loss_fn(logits, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * pose3d.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += pose3d.size(0)
            pbar.set_postfix(loss=loss.item(), acc=correct/max(1, total))

        train_acc = correct / max(1, total)
        print(f"[B] Epoch {epoch} train_acc={train_acc:.4f}")

        # ===================== VALIDATION =====================
        pose_gnn.eval(); head.eval()
        v_loss, v_corr, v_total = 0, 0, 0
        with torch.no_grad():
            for pose3d, label in val_loader:
                pose3d, label = pose3d.to(device), label.to(device)
                feats = pose_gnn(pose3d, edges)
                logits = head(feats)
                loss = loss_fn(logits, label)
                v_loss += loss.item() * pose3d.size(0)
                pred = logits.argmax(dim=-1)
                v_corr += (pred == label).sum().item()
                v_total += pose3d.size(0)

        val_acc = v_corr / max(1, v_total)
        val_loss = v_loss / max(1, v_total)
        print(f"[B] Epoch {epoch} val_acc={val_acc:.4f} val_loss={val_loss:.4f}")

        if writer:
            writer.add_scalar("loss/train", running_loss/max(1,total), epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)

        # Save best
        if val_loss < best:
            best = val_loss
            torch.save({
                "pose_gnn": pose_gnn.state_dict(),
                "head": head.state_dict(),
                "label2id": ds.label2id
            }, os.path.join(args.out_dir, "pose_encoder_best.pth"))
            print("[B] Saved best PoseGNN checkpoint!")

    if writer: writer.close()
    print(f"[B] Done. Best val loss: {best:.4f}")


# ===================== CLI =====================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json", type=str, required=True, help="Path to dataset.json")
    p.add_argument("--out_dir", default="outputs/stageB")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--gnn_type", choices=["gcn", "sage"], default="sage")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()

    train(args)
