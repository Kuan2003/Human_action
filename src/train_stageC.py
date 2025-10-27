#!/usr/bin/env python3
"""
Stage C - Train temporal LSTM (freeze encoders)
- Uses HOISequenceDataset (seq_len>1)
- Per-frame loss (CrossEntropy over each frame)
Compatible with:
 - models.ObjectEncoder
 - models.PoseGNN_PyG (returns [B, pose_dim] for batch input or [pose_dim] for single)
 - models.FusionLSTM
Checkpoint loader is robust (accepts dicts saved from Stage A/B).
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
import models as models_module
from environment_config import setup_environment

ObjectEncoder = getattr(models_module, "ObjectEncoder", None)
PoseGNN = getattr(models_module, "PoseGNN_PyG", None)
FusionLSTM = getattr(models_module, "FusionLSTM", None)


def session_split(ds, val_ratio=0.2, seed=123):
    """
    Split dataset into train/val based on session ids (windows in HOISequenceDataset)
    """
    sess2idx = {}
    for i, w in enumerate(ds.windows):
        sess = w[0]
        sess2idx.setdefault(sess, []).append(i)
    sessions = list(sess2idx.keys())
    random.Random(seed).shuffle(sessions)
    total = len(ds.windows)
    val_idx = []
    cur = 0
    for s in sessions:
        if cur / total >= val_ratio:
            break
        val_idx += sess2idx[s]
        cur += len(sess2idx[s])
    train_idx = [i for i in range(total) if i not in val_idx]
    return train_idx, val_idx


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


def safe_load_state_dict(target_module, ckpt_path, key_candidates=None, map_location=None):
    """
    Load state dict robustly:
    - if file contains dict with specific keys, try to find matching key.
    - otherwise assume it's a raw state_dict and load directly.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False
    data = torch.load(ckpt_path, map_location=map_location)
    if isinstance(data, dict):
        # try useful keys
        keys = key_candidates or ["encoder", "model", "state_dict", "obj_encoder", "pose_gnn", "backbone"]
        for k in keys:
            if k in data:
                try:
                    target_module.load_state_dict(data[k])
                    return True
                except Exception:
                    # sometimes nested further
                    try:
                        target_module.load_state_dict(data[k + "_state_dict"])
                        return True
                    except Exception:
                        pass
        # fallback: if dict seems to be a state_dict (tensor values), try load directly
        try:
            target_module.load_state_dict(data)
            return True
        except Exception:
            return False
    else:
        # file is a state_dict (unlikely) - attempt direct load
        try:
            target_module.load_state_dict(data)
            return True
        except Exception:
            return False


def train(args):
    setup_environment()
    
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"[C] Using device: {device}")

    ds = HOISequenceDataset(args.dataset_json, seq_len=args.seq_len, stride=args.stride)
    print(f"[C] sequences: {len(ds.windows)}, classes: {ds.label2id}")

    train_idx, val_idx = session_split(ds, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = DataLoader(ds, batch_size=args.batch, sampler=SubsetRandomSampler(train_idx),
                              collate_fn=lambda x: x, num_workers=args.workers)
    val_loader = DataLoader(ds, batch_size=args.batch, sampler=SubsetRandomSampler(val_idx),
                            collate_fn=lambda x: x, num_workers=args.workers)

    edges = get_edges().to(device)

    # instantiate encoders and fusion
    obj_enc = ObjectEncoder(out_dim=args.obj_dim).to(device)
    # PoseGNN signature supports in_dim, hidden, out_dim, model_type
    pose_enc = PoseGNN(in_dim=3, hidden=args.pose_hidden, out_dim=args.pose_dim, model_type=args.gnn_type).to(device)

    # try to load pretrained checkpoints (robust)
    if args.obj_ckpt and os.path.exists(args.obj_ckpt):
        ok = safe_load_state_dict(obj_enc, args.obj_ckpt, key_candidates=["encoder", "obj_encoder", "model", "state_dict"], map_location=device)
        if ok:
            print("[C] Loaded obj ckpt")
        else:
            print("[C] WARNING: failed to load obj ckpt (format mismatch)")

    if args.pose_ckpt and os.path.exists(args.pose_ckpt):
        ok = safe_load_state_dict(pose_enc, args.pose_ckpt, key_candidates=["pose_gnn", "model", "state_dict"], map_location=device)
        if ok:
            print("[C] Loaded pose ckpt")
        else:
            print("[C] WARNING: failed to load pose ckpt (format mismatch)")

    # freeze encoders
    obj_enc.eval()
    pose_enc.eval()
    for p in obj_enc.parameters():
        p.requires_grad = False
    for p in pose_enc.parameters():
        p.requires_grad = False

    fusion = FusionLSTM(input_dim=args.obj_dim + args.pose_dim + args.kin_dim,
                        hidden=args.hidden, num_classes=len(ds.label2id),
                        num_layers=args.lstm_layers, bidirectional=args.bidirectional).to(device)

    opt = torch.optim.AdamW(fusion.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb")) if args.tensorboard else None
    best = -1.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        fusion.train()
        running_loss = 0.0; tot_frames = 0; corr = 0
        pbar = tqdm(train_loader, desc=f"[C] Epoch {epoch}")
        for batch in pbar:
            # batch is list of windows (each window is dict returned by HOISequenceDataset)
            B = len(batch)
            T = args.seq_len

            # stack tensors (dataset returns keys: obj_imgs, pose_nodes, kin_feats, label, session_id)
            obj_imgs = torch.stack([item["obj_imgs"] for item in batch], dim=0).to(device)    # [B,T,3,H,W]
            pose_nodes = torch.stack([item["pose_nodes"] for item in batch], dim=0).to(device) # [B,T,21,3]
            kin = torch.stack([item["kin_feats"] for item in batch], dim=0).to(device)        # [B,T,kin_dim]
            # dataset returns "label" per frame (shape [T])
            labels = torch.stack([item["label"] for item in batch], dim=0).to(device)         # [B,T]

            # extract features with frozen encoders
            with torch.no_grad():
                B_, T_, C, H, W = obj_imgs.shape
                imgs_flat = obj_imgs.view(B_ * T_, C, H, W)
                obj_feats_flat = obj_enc(imgs_flat)                 # [B*T, obj_dim]
                obj_feats = obj_feats_flat.view(B_, T_, -1)         # [B,T,obj_dim]

                # pose embeddings per sample/frame
                # Our PoseGNN forward supports batch input [B', N, F] -> [B', D]
                pose_feats = []
                # process per time-step to avoid huge memory if needed; here we process per-batch/time in loops
                for t in range(T_):
                    # pose_nodes[:, t] -> [B, 21, 3]
                    p_batch = pose_nodes[:, t]   # [B,21,3]
                    emb = pose_enc(p_batch, edges)  # [B, pose_dim]
                    # ensure shape
                    if emb.ndim == 1:
                        emb = emb.unsqueeze(0)
                    pose_feats.append(emb)
                pose_feats = torch.stack(pose_feats, dim=1)  # [B, T, pose_dim]

            fused = torch.cat([obj_feats, pose_feats, kin], dim=-1)  # [B,T,input_dim]
            logits = fusion(fused)  # [B,T,C]

            # per-frame loss: reshape
            Bf, Tf, Cc = logits.shape
            loss = loss_fn(logits.view(-1, Cc), labels.view(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * (Bf * Tf)
            preds = logits.argmax(-1)
            corr += (preds == labels).sum().item()
            tot_frames += (Bf * Tf)
            pbar.set_postfix(loss=running_loss / max(1, tot_frames), acc=corr / max(1, tot_frames))

        # validation
        fusion.eval()
        v_loss = 0.0; v_corr = 0; v_tot = 0
        with torch.no_grad():
            for batch in val_loader:
                B_ = len(batch)
                obj_imgs = torch.stack([item["obj_imgs"] for item in batch], dim=0).to(device)
                pose_nodes = torch.stack([item["pose_nodes"] for item in batch], dim=0).to(device)
                kin = torch.stack([item["kin_feats"] for item in batch], dim=0).to(device)
                labels = torch.stack([item["label"] for item in batch], dim=0).to(device)

                B_, T_, C, H, W = obj_imgs.shape
                imgs_flat = obj_imgs.view(B_ * T_, C, H, W)
                obj_feats = obj_enc(imgs_flat).view(B_, T_, -1)

                pose_feats = []
                for t in range(T_):
                    p_batch = pose_nodes[:, t]  # [B,21,3]
                    emb = pose_enc(p_batch, edges)  # [B,pose_dim]
                    if emb.ndim == 1:
                        emb = emb.unsqueeze(0)
                    pose_feats.append(emb)
                pose_feats = torch.stack(pose_feats, dim=1)  # [B,T,pose_dim]

                fused = torch.cat([obj_feats, pose_feats, kin], dim=-1)
                logits = fusion(fused)
                loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
                v_loss += loss.item() * (B_ * T_)
                preds = logits.argmax(-1)
                v_corr += (preds == labels).sum().item()
                v_tot += (B_ * T_)

        val_acc = v_corr / max(1, v_tot)
        print(f"[C] Epoch {epoch} val_acc={val_acc:.4f} val_loss={v_loss / max(1, v_tot):.4f}")

        if writer:
            writer.add_scalar("loss/train", running_loss / max(1, tot_frames), epoch)
            writer.add_scalar("loss/val", v_loss / max(1, v_tot), epoch)
            writer.add_scalar("acc/val", val_acc, epoch)

        if val_acc > best:
            best = val_acc
            torch.save({
                "fusion": fusion.state_dict(),
                "label2id": ds.label2id
            }, os.path.join(args.out_dir, "fusion_best.pth"))
            print("[C] Saved best fusion")

    if writer:
        writer.close()
    print("[C] Done. Best val acc:", best)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json", default="dataset.json")
    p.add_argument("--out_dir", default="outputs/stageC")
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--obj_dim", type=int, default=128)
    p.add_argument("--pose_dim", type=int, default=128)
    p.add_argument("--pose_hidden", type=int, default=128)
    p.add_argument("--kin_dim", type=int, default=4)
    p.add_argument("--gnn_type", choices=["gcn", "sage"], default="gcn")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--obj_ckpt", default="outputs/stageA/obj_encoder_best.pth")
    p.add_argument("--pose_ckpt", default="outputs/stageB/pose_encoder_best.pth")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()
    train(args)
