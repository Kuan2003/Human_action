#!/usr/bin/env python3
"""
Stage B (Self-Supervised) - Pose GNN pretraining (SimCLR-style)
- Input: hand 3D keypoints [21,3]
- Train: contrastive (NT-Xent) on augmented pose pairs
- Output: checkpoint containing pose_gnn weights (pose_encoder_best.pth)
"""

import os
import argparse
import random
import math
from tqdm import tqdm
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import HOISequenceDataset
from preprocess_ultils import load_pose_json, keypoint_2d_to_3d, load_depth
import models as models_module

# --------------------------
# Dataset: flatten windows -> frames (reuse your SingleFrame wrapper)
# --------------------------
class SingleFramePoseDataset(torch.utils.data.Dataset):
    """
    Flatten HOISequenceDataset windows -> single frame pose samples
    Returns pose3d: np.array shape [21,3]
    """
    def __init__(self, dataset_json, intrinsics=None, drop_no_pose=True):
        base_ds = HOISequenceDataset(dataset_json, seq_len=8, stride=8,
                                     intrinsics=intrinsics, drop_no_pose=drop_no_pose)
        self.samples = []
        for sess, frames in base_ds.windows:
            for f in frames:
                if not f.get("has_pose", True):
                    continue
                # assume HOISequenceDataset resolved paths already
                self.samples.append(f)
        self.intrinsics = base_ds.intrinsics
        print(f"[B-selfsup] Flattened to {len(self.samples)} pose frames from {len(base_ds.windows)} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f = self.samples[idx]
        pose2d = load_pose_json(f["pose_path"])
        depth = load_depth(f["depth_path"])
        pose3d = keypoint_2d_to_3d(pose2d, depth, self.intrinsics)  # [21,3] numpy
        return pose3d.astype(np.float32)  # return numpy array


# --------------------------
# Pose augmentations (3D): rotation, scaling, jitter
# --------------------------
def augment_pose(pose, rot_deg=10.0, scale_jitter=0.1, jitter_std=0.01):
    """
    pose: np.array [21,3]
    - rotate around x/y/z by small random angles (degrees)
    - uniform scale in [1-scale_jitter, 1+scale_jitter]
    - gaussian jitter per coordinate
    returns augmented pose (np array)
    """
    p = pose.copy()
    # center at wrist (assume wrist index 0)
    center = p[0:1].copy()  # [1,3]
    p = p - center  # translate to origin

    # random rotation angles in radians
    ax = math.radians(random.uniform(-rot_deg, rot_deg))
    ay = math.radians(random.uniform(-rot_deg, rot_deg))
    az = math.radians(random.uniform(-rot_deg, rot_deg))

    Rx = np.array([[1,0,0],
                   [0, math.cos(ax), -math.sin(ax)],
                   [0, math.sin(ax),  math.cos(ax)]], dtype=np.float32)
    Ry = np.array([[ math.cos(ay),0, math.sin(ay)],
                   [0,1,0],
                   [-math.sin(ay),0, math.cos(ay)]], dtype=np.float32)
    Rz = np.array([[math.cos(az), -math.sin(az),0],
                   [math.sin(az),  math.cos(az),0],
                   [0,0,1]], dtype=np.float32)

    R = Rz @ Ry @ Rx
    p = (p @ R.T)

    # scale
    s = random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
    p = p * s

    # jitter
    p = p + np.random.normal(scale=jitter_std, size=p.shape).astype(np.float32)

    # translate back
    p = p + center
    return p


# --------------------------
# NT-Xent loss (stable)
# --------------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: tensors [N, D]
        returns scalar loss
        """
        assert z_i.shape == z_j.shape
        device = z_i.device
        N = z_i.shape[0]

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)  # [2N, D]

        sim = torch.matmul(z, z.t()) / self.temperature  # [2N,2N]
        # mask diagonal
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, -1e9)

        # positive targets: for row i (0..N-1) positive is i+N, for row i (N..2N-1) positive is i-N
        pos_idx = torch.arange(N, device=device)
        targets = torch.cat([pos_idx + N, pos_idx], dim=0)  # [2N]

        loss = F.cross_entropy(sim, targets)
        return loss


# --------------------------
# Helper: collate to batch numpy poses -> torch tensor
# --------------------------
def collate_fn(batch):
    # batch: list of np arrays [21,3]
    poses = np.stack(batch, axis=0)  # [B,21,3]
    poses = torch.from_numpy(poses)  # float32
    return poses


# --------------------------
# Training main
# --------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"[B-selfsup] device: {device}")

    ds = SingleFramePoseDataset(args.dataset_json)
    print(f"[B-selfsup] total frames: {len(ds)}")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_fn, pin_memory=True)

    # edges and model
    edges = models_module.get_hand_edges().to(device) if hasattr(models_module, "get_hand_edges") else None
    PoseGNN = getattr(models_module, "PoseGNN_PyG", None)
    if PoseGNN is None:
        raise RuntimeError("PoseGNN_PyG not found in models.py")

    pose_gnn = PoseGNN(in_dim=3, hidden=args.hidden, out_dim=args.out_dim, model_type=args.gnn_type).to(device)

    projector = nn.Sequential(
        nn.Linear(args.out_dim, args.proj_dim),
        nn.ReLU(),
        nn.Linear(args.proj_dim, args.proj_dim)
    ).to(device)

    opt = torch.optim.AdamW(list(pose_gnn.parameters()) + list(projector.parameters()),
                            lr=args.lr, weight_decay=args.wd)

    criterion = NTXentLoss(temperature=args.temp)

    os.makedirs(args.out_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        pose_gnn.train(); projector.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"[B-selfsup] Epoch {epoch}")

        for poses in pbar:
            # poses: [B,21,3] (torch tensor, CPU or pinned)
            B = poses.shape[0]
            # convert to numpy for augmentation (cheap) then back (could be optimized)
            poses_np = poses.numpy()

            # create two augmented views
            view1 = np.stack([augment_pose(poses_np[i],
                                          rot_deg=args.rot_deg,
                                          scale_jitter=args.scale_jitter,
                                          jitter_std=args.jitter_std) for i in range(B)], axis=0)
            view2 = np.stack([augment_pose(poses_np[i],
                                          rot_deg=args.rot_deg,
                                          scale_jitter=args.scale_jitter,
                                          jitter_std=args.jitter_std) for i in range(B)], axis=0)

            view1 = torch.from_numpy(view1).to(device)  # [B,21,3]
            view2 = torch.from_numpy(view2).to(device)

            # forward through pose_gnn: PoseGNN.forward accepts [B,N,F]
            z1 = pose_gnn(view1, edges)   # [B, out_dim]
            z2 = pose_gnn(view2, edges)   # [B, out_dim]

            # projection
            p1 = projector(z1)  # [B,proj_dim]
            p2 = projector(z2)

            # normalize
            p1 = F.normalize(p1, dim=-1)
            p2 = F.normalize(p2, dim=-1)

            loss = criterion(p1, p2)

            if not torch.isfinite(loss):
                print("⚠️ Non-finite loss detected. Skipping batch")
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(pose_gnn.parameters()) + list(projector.parameters()), max_norm=1.0)
            opt.step()

            running_loss += loss.item() * B
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(ds)
        print(f"[B-selfsup] Epoch {epoch} avg_loss={avg_loss:.6f}")

        # save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "pose_gnn": pose_gnn.state_dict(),
                "projector": projector.state_dict(),
                "meta": {
                    "out_dim": args.out_dim,
                    "proj_dim": args.proj_dim,
                    "gnn_type": args.gnn_type
                }
            }, os.path.join(args.out_dir, "pose_encoder_best.pth"))
            print(f"[B-selfsup] ✅ Saved best encoder (loss={best_loss:.6f})")
        log_path = os.path.join(args["out_dir"], "train_log.txt")
        with open(log_path, "a") as f:
            f.write(f"{datetime.datetime.now()} | epoch={epoch} | avg_loss={avg_loss:.4f} \n")

    print("[B-selfsup] Done. Best loss:", best_loss)


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json", type=str, required=True)
    p.add_argument("--out_dir", default="outputs/stageB")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--gnn_type", choices=["gcn", "sage"], default="gcn")
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--rot_deg", type=float, default=10.0, help="max rotation degree for augment")
    p.add_argument("--scale_jitter", type=float, default=0.1)
    p.add_argument("--jitter_std", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()

    train(args)
