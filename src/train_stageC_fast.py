#!/usr/bin/env python3
"""
train_stageC.py

Train Fusion LSTM on pre-extracted features saved by extract_features_stageC_v3.py.

- Supports block_xxx.pt files (each file is a list of dicts) and single .pt dict files.
- Per-frame supervision with masking (handles variable-length sequences).
- TensorBoard + matplotlib logging.
"""

import os
import time
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# -----------------------
# Utilities: loading blocks
# -----------------------
def load_feature_blocks(feature_dir: str):
    """
    Load sequences from feature_dir. Accepts:
      - file.pt -> dict {"features": Tensor[T,D], "labels": Tensor[T], "session": str}
      - block_*.pt -> list[ dict(...) ]
    Returns list of dicts each with "features"(Tensor), "labels"(Tensor), "session"(str)
    Skips invalid files with warnings.
    """
    samples = []
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"Feature dir not found: {feature_dir}")
    files = sorted(os.listdir(feature_dir))
    for fname in files:
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(feature_dir, fname)
        try:
            data = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[⚠️ Skip] Failed to load {fname}: {e}")
            continue

        # case: block list
        if isinstance(data, list):
            added = 0
            for i, entry in enumerate(data):
                if isinstance(entry, dict) and "features" in entry and "labels" in entry:
                    # standardize types
                    try:
                        entry["features"] = torch.as_tensor(entry["features"]).float()
                        entry["labels"] = torch.as_tensor(entry["labels"]).long()
                    except Exception:
                        print(f"[⚠️ Skip] Invalid types in {fname}[{i}]")
                        continue
                    samples.append(entry)
                    added += 1
                else:
                    print(f"[⚠️ Skip] Invalid item in {fname} at idx {i}")
            if added == 0:
                print(f"[⚠️ Skip] No valid sequences inside {fname}")
        # case: single dict
        elif isinstance(data, dict):
            if "features" in data and "labels" in data:
                try:
                    data["features"] = torch.as_tensor(data["features"]).float()
                    data["labels"] = torch.as_tensor(data["labels"]).long()
                except Exception:
                    print(f"[⚠️ Skip] Invalid types in {fname}")
                    continue
                samples.append(data)
            else:
                print(f"[⚠️ Skip] Invalid file structure: {fname}")
        else:
            print(f"[⚠️ Skip] Unknown file type: {fname} ({type(data)})")

    print(f"[Dataset] Loaded {len(samples)} valid sequences from {feature_dir}")
    return samples


# -----------------------
# Dataset wrapper
# -----------------------
class FeatureSequenceDataset(Dataset):
    """
    In-memory dataset on top of samples (list of dicts).
    Each sample dict contains:
       "features": Tensor[T, D]
       "labels": Tensor[T]
       "session": str (optional)
    """
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        feats = s["features"]          # Tensor [T, D]
        labels = s["labels"]           # Tensor [T]
        session = s.get("session", f"seq_{idx}")
        return feats, labels, session


def collate_pad(batch):
    """
    Pad batch of variable-length sequences to max_T in batch.
    Returns:
      feats: [B, T_max, D]
      labels: [B, T_max] (padded with -100 for CE ignore_index)
      mask: [B, T_max] (1.0 for valid, 0 for padded)
      sessions: list
    """
    feats_list, labels_list, sessions = zip(*batch)
    B = len(feats_list)
    T_list = [f.shape[0] for f in feats_list]
    T_max = max(T_list)
    D = feats_list[0].shape[1]

    feat_pad = torch.zeros((B, T_max, D), dtype=torch.float32)
    label_pad = torch.full((B, T_max), fill_value=-100, dtype=torch.long)  # -100 to ignore in loss
    mask = torch.zeros((B, T_max), dtype=torch.float32)

    for i, (f, l) in enumerate(zip(feats_list, labels_list)):
        T = f.shape[0]
        feat_pad[i, :T] = f
        # ensure labels tensor shape and type
        label_pad[i, :T] = l.reshape(-1)
        mask[i, :T] = 1.0

    return feat_pad, label_pad, mask, list(sessions)


# -----------------------
# Model
# -----------------------
class FusionLSTMPerFrame(nn.Module):
    """
    LSTM that outputs per-frame logits (B, T, C).
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.mult = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_dim * self.mult, num_classes)

    def forward(self, x):
        # x: [B, T, D]
        h, _ = self.lstm(x)            # [B, T, H*mult]
        logits = self.classifier(h)    # [B, T, C]
        return logits


# -----------------------
# Metrics
# -----------------------
def masked_ce_loss(logits, labels, mask, ignore_index=-100):
    """
    logits: [B, T, C], labels: [B, T], mask: [B, T] (float)
    compute average CE per valid token
    """
    B, T, C = logits.shape
    logits_flat = logits.view(B * T, C)
    labels_flat = labels.view(B * T)
    loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
    total_loss = loss(logits_flat, labels_flat)  # sum over non-ignored
    valid = (labels_flat != ignore_index).float().sum().item()
    if valid == 0:
        return torch.tensor(0.0, device=logits.device), 0
    return total_loss / valid, int(valid)


def masked_accuracy(logits, labels, mask, ignore_index=-100):
    preds = logits.argmax(dim=-1)  # [B, T]
    valid = (labels != ignore_index)
    if valid.sum().item() == 0:
        return 0.0
    correct = ((preds == labels) & valid).sum().item()
    total = valid.sum().item()
    return correct / total


# -----------------------
# Train routine
# -----------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"[Train] Using device: {device}")

    samples = load_feature_blocks(args.feature_dir)
    if len(samples) == 0:
        raise RuntimeError("No valid extracted feature sequences found. Check feature_dir and extract step.")

    # build dataset & dataloaders
    dataset = FeatureSequenceDataset(samples)

    # detect num classes robustly
    all_label_tensors = []
    for i, s in enumerate(samples):
        lbl = s.get("labels", None)
        if lbl is None:
            continue
        if not isinstance(lbl, torch.Tensor):
            try:
                lbl = torch.tensor(lbl, dtype=torch.long)
            except Exception:
                print(f"[⚠️ Skip labels from sample {i}]")
                continue
        all_label_tensors.append(lbl.reshape(-1))
    if len(all_label_tensors) == 0:
        raise RuntimeError("No label tensors available in samples.")

    labels_cat = torch.cat(all_label_tensors)
    num_classes = int(labels_cat.unique().numel())
    print(f"[Dataset] {len(dataset)} sequences. Detected num_classes={num_classes}")

    # split
    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"[Dataset] Train={len(train_ds)}, Val={len(val_ds)} (val_ratio={args.val_ratio})")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_pad, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate_pad, num_workers=args.workers, pin_memory=True)

    # model
    sample_feat = samples[0]["features"]
    input_dim = sample_feat.shape[1]
    model = FusionLSTMPerFrame(input_dim=input_dim, hidden_dim=args.hidden,
                               num_classes=num_classes, num_layers=args.num_layers,
                               bidirectional=args.bidirectional, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tensorboard"))

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_acc = -1.0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_valid_tokens = 0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] train")
        for feats, labels, mask, sessions in pbar:
            feats = feats.to(device)            # [B, T, D]
            labels = labels.to(device)          # [B, T] (may contain -100)
            # mask not needed except for metrics
            mask = mask.to(device)

            logits = model(feats)               # [B, T, C]
            loss_avg, n_valid = masked_ce_loss(logits, labels, mask, ignore_index=-100)
            # loss_avg already normalized by valid tokens
            optimizer.zero_grad()
            loss_avg.backward()
            optimizer.step()

            # metrics
            acc = masked_accuracy(logits.detach().cpu(), labels.detach().cpu(), mask.detach().cpu(), ignore_index=-100)
            epoch_loss_sum += float(loss_avg.item()) * max(1, n_valid)
            epoch_valid_tokens += n_valid
            epoch_correct += int((logits.argmax(dim=-1).detach().cpu() == labels.detach().cpu()).logical_and(labels.detach().cpu() != -100).sum().item())
            epoch_total += n_valid
            print("Batch label dist:", torch.bincount(labels.flatten()))
            pbar.set_postfix({"loss": f"{loss_avg.item():.4f}", "acc": f"{acc:.3f}"})

        train_epoch_loss = epoch_loss_sum / max(1, epoch_valid_tokens)
        train_epoch_acc = epoch_correct / max(1, epoch_total) if epoch_total > 0 else 0.0

        # validation
        model.eval()
        val_loss_sum = 0.0
        val_valid_tokens = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats, labels, mask, sessions in tqdm(val_loader, desc=f"[Epoch {epoch}] val"):
                feats = feats.to(device)
                labels = labels.to(device)
                mask = mask.to(device)

                logits = model(feats)
                loss_avg, n_valid = masked_ce_loss(logits, labels, mask, ignore_index=-100)

                val_loss_sum += float(loss_avg.item()) * max(1, n_valid)
                val_valid_tokens += n_valid
                val_correct += int((logits.argmax(dim=-1).cpu() == labels.cpu()).logical_and(labels.cpu() != -100).sum().item())
                val_total += n_valid

        val_epoch_loss = val_loss_sum / max(1, val_valid_tokens)
        val_epoch_acc = val_correct / max(1, val_total) if val_total > 0 else 0.0

        # scheduler step (ReduceLROnPlateau uses metric)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_epoch_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"[Train] Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        

        # logging
        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(val_epoch_loss)
        train_acc_history.append(train_epoch_acc)
        val_acc_history.append(val_epoch_acc)

        writer.add_scalar("Loss/Train", train_epoch_loss, epoch)
        writer.add_scalar("Loss/Val", val_epoch_loss, epoch)
        writer.add_scalar("Acc/Train", train_epoch_acc, epoch)
        writer.add_scalar("Acc/Val", val_epoch_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        elapsed = time.time() - start_time
        print(f"[Epoch {epoch}] train_loss={train_epoch_loss:.4f} train_acc={train_epoch_acc:.3f} | "
              f"val_loss={val_epoch_loss:.4f} val_acc={val_epoch_acc:.3f} | elapsed={elapsed/60:.2f}m")

        # save checkpoint
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "meta": {
                "input_dim": input_dim,
                "hidden": args.hidden,
                "num_classes": num_classes
            }
        }
        

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(ckpt, os.path.join(args.out_dir, "fusion_best.pth"))
            print(f"[Checkpoint] Saved best model (val_acc={best_val_acc:.4f})")

    # final plotting
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss_history, label="train_loss")
    plt.plot(val_loss_history, label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(train_acc_history, label="train_acc")
    plt.plot(val_acc_history, label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend(); plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "training_curves.png"))
    plt.close()

    writer.close()
    print(f"[Done] Best val_acc = {best_val_acc:.4f}. Results saved to {args.out_dir}")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--feature_dir", type=str, required=True,
                   help="directory with extracted features (block_*.pt or single .pt files)")
    p.add_argument("--out_dir", type=str, default="outputs/stageC")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()

    train(args)
