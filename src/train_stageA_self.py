#!/usr/bin/env python3
# =============================================================
# Stage A (Self-Supervised) - Object Encoder Pretraining (SimCLR)
# =============================================================
# Huấn luyện encoder bằng contrastive learning: tạo 2 view từ cùng 1 ảnh (augmentation),
# rồi tối đa hóa similarity giữa chúng và tối thiểu với các ảnh khác trong batch.
# =============================================================

import os
import json
import cv2
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import ObjectEncoder
from preprocess_ultils import load_rgb, load_mask
from path_utils import SmartPathResolver


# =============================================================
# Dataset cho contrastive learning (SimCLR)
# =============================================================
class ObjectFrameDataset(Dataset):
    def __init__(self, dataset_json, transform=None, drop_no_mask=True):
        self.path_resolver = SmartPathResolver(anchor_file=dataset_json)
        print(f"[StageA] Root: {self.path_resolver.get_project_root()}")

        with open(dataset_json) as f:
            items = json.load(f)

        self.items = []
        base_dir = os.path.dirname(os.path.abspath(dataset_json))

        for it in items:
            if drop_no_mask and not it.get("has_mask", True):
                continue

            rgb_path = self.path_resolver.resolve(it["rgb_path"], must_exist=False)
            mask_path = self.path_resolver.resolve(it["mask_path"], must_exist=False)

            if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
                continue

            it["rgb_path"], it["mask_path"] = rgb_path, mask_path
            self.items.append(it)

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        # Augmentation cho contrastive learning
        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(128, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        rgb = load_rgb(it["rgb_path"])
        mask = load_mask(it["mask_path"])

        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            pad = 5
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(rgb.shape[1]-1, x1 + pad), min(rgb.shape[0]-1, y1 + pad)
            rgb = rgb[y0:y1, x0:x1]

        rgb = cv2.resize(rgb, (128, 128))
        img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Hai view khác nhau (augmentation ngẫu nhiên)
        view1 = self.augment(img)
        view2 = self.augment(img)
        return view1, view2


# =============================================================
# NT-Xent Loss (fixed and stable)
# =============================================================
class NTXentLoss(nn.Module):
    """
    Correct NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)
    Used in SimCLR for contrastive learning.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        assert z_i.shape == z_j.shape
        N = z_i.shape[0]
        device = z_i.device

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)  # [2N, D]

        # Similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # [2N,2N]
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, -1e9)  # tránh self-similarity

        # Positive indices
        pos_indices = torch.arange(N, device=device)
        targets = torch.cat([pos_indices + N, pos_indices], dim=0)

        # Cross-entropy loss
        loss = F.cross_entropy(sim, targets)
        return loss


# =============================================================
# Training Loop
# =============================================================
def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    ds = ObjectFrameDataset(args.dataset_json)
    print(f"[A-selfsup] Loaded {len(ds)} frames")

    train_loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    encoder = ObjectEncoder(out_dim=args.obj_dim, pretrained=True).to(device)
    projector = nn.Sequential(
        nn.Linear(args.obj_dim, args.proj_dim),
        nn.ReLU(),
        nn.Linear(args.proj_dim, args.proj_dim)
    ).to(device)

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(projector.parameters()),
                                  lr=args.lr, weight_decay=args.wd)
    criterion = NTXentLoss(temperature=args.temp)

    os.makedirs(args.out_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        encoder.train(); projector.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[A] Epoch {epoch}")
        for v1, v2 in pbar:
            v1, v2 = v1.to(device), v2.to(device)

            # Forward pass
            f1 = encoder(v1)
            f2 = encoder(v2)
            z1 = projector(f1)
            z2 = projector(f2)

            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)

            loss = criterion(z1, z2)

            if not torch.isfinite(loss):
                print("⚠️ Non-finite loss detected. Skipping batch.")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(projector.parameters()), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * v1.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(ds)
        print(f"[A][Epoch {epoch}] Avg Loss = {avg_loss:.6f}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict()
            }, os.path.join(args.out_dir, "obj_encoder_best.pth"))
            print(f"[A] ✅ Saved best encoder (loss={best_loss:.6f})")
        log_path = os.path.join(args["out_dir"], "train_log.txt")
        with open(log_path, "a") as f:
            f.write(f"{datetime.datetime.now()} | epoch={epoch} | avg_loss={avg_loss:.4f} \n")

    print(f"[A] Done. Best contrastive loss = {best_loss:.6f}")

    
# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage A - Self-Supervised Contrastive Pretraining (SimCLR)")
    p.add_argument("--dataset_json", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/stageA_selfsup")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--obj_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()
    train(args)
