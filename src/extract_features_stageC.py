#!/usr/bin/env python3
"""
extract_features_stageC_v3.py
Fast, stable feature extraction for Stage C using DataLoader (parallel I/O)
- Dataset returns a sequence item: obj_imgs [T,C,H,W], pose_nodes [T,N,3], kin_feats [T,K], label [T]
- DataLoader yields batches of sequences: we batch GPU inference across (B*T) frames
- Save features in blocks (reduces file-count I/O)
- Auto cleanup (del + torch.cuda.empty_cache) to avoid slowdown over time
"""

import os, time, torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import HOISequenceDataset
from models import ObjectEncoder, PoseGNN_PyG, get_hand_edges

# -------------------------
# Utils
# -------------------------
def try_load_state(model, ckpt_path):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[Extract] ⚠️ Checkpoint not found: {ckpt_path}")
        return False
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        candidates = ["encoder", "pose_gnn", "model", "state_dict", "state"]
        for k in candidates:
            if k in ckpt:
                try:
                    model.load_state_dict(ckpt[k])
                    print(f"[Extract] ✅ Loaded model weights using key '{k}'")
                    return True
                except Exception as e:
                    print(f"[Extract] ⚠️ Key '{k}' failed: {e}")
        try:
            model.load_state_dict(ckpt)
            print("[Extract] ✅ Loaded direct state_dict")
            return True
        except Exception as e:
            print(f"[Extract] ❌ Could not load checkpoint: {e}")
            return False
    else:
        try:
            model.load_state_dict(ckpt)
            print("[Extract] ✅ Loaded direct checkpoint")
            return True
        except Exception as e:
            print(f"[Extract] ❌ Could not load checkpoint: {e}")
            return False

def make_batched_edge_index(edges: torch.LongTensor, num_nodes: int, batch_size: int, device=None):
    device = device or edges.device
    edge_list = []
    for b in range(batch_size):
        offset = b * num_nodes
        edge_list.append(edges + offset)
    return torch.cat(edge_list, dim=1).to(device)  # [2, batch_size * E]

def collate_sequences(batch):
    """
    batch: list of dicts from HOISequenceDataset
    return stacked tensors:
      obj_imgs: [B,T,C,H,W]
      pose_nodes: [B,T,N,3]
      kin_feats: [B,T,K]
      labels: [B,T]
      session_ids: list
    """
    obj = torch.stack([item["obj_imgs"] for item in batch], dim=0)
    pose = torch.stack([item["pose_nodes"] for item in batch], dim=0)
    kin = torch.stack([item["kin_feats"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    sessions = [item.get("session_id", f"seq_{i}") for i,item in enumerate(batch)]
    return {
        "obj_imgs": obj,
        "pose_nodes": pose,
        "kin_feats": kin,
        "labels": labels,
        "session_ids": sessions
    }

# -------------------------
# Main
# -------------------------
@torch.no_grad()
def extract_features_v3(args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"[Extract] Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    ds = HOISequenceDataset(args.dataset_json, seq_len=args.seq_len, stride=args.stride)
    print(f"[Dataset] {len(ds.windows)} sequences loaded from dataset.")

    # DataLoader to parallelize file I/O and preprocessing
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True, collate_fn=collate_sequences)

    # models
    obj_enc = ObjectEncoder(out_dim=args.obj_dim, pretrained=False).to(device)
    pose_enc = PoseGNN_PyG(in_dim=3, hidden=args.hidden, out_dim=args.pose_dim, model_type=args.gnn_type).to(device)

    if args.obj_ckpt: try_load_state(obj_enc, args.obj_ckpt)
    if args.pose_ckpt: try_load_state(pose_enc, args.pose_ckpt)

    obj_enc.eval(); pose_enc.eval()

    edges = get_hand_edges().to(torch.long).to(device)

    saved_blocks = 0
    buffer = []
    save_block = args.save_block
    t0 = time.time()
    total_sequences = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="[Extract] batches")):
        # batch tensors on CPU/pinned memory
        obj_imgs = batch["obj_imgs"]      # [B, T, C, H, W]
        pose_nodes = batch["pose_nodes"]  # [B, T, N, 3]
        kin_feats = batch["kin_feats"]    # [B, T, K]
        labels = batch["labels"]          # [B, T]
        sessions = batch["session_ids"]   # list length B

        B, T, C, H, W = obj_imgs.shape
        N_nodes = pose_nodes.shape[2]
        K = kin_feats.shape[-1]

        # move to device in a single call for efficiency
        obj_imgs = obj_imgs.to(device, non_blocking=True).view(B*T, C, H, W)     # [B*T, C, H, W]
        pose_nodes = pose_nodes.to(device, non_blocking=True).view(B*T*N_nodes, 3)  # [B*T*N, 3]
        kin_feats = kin_feats.to(device, non_blocking=True).view(B*T, K)       # [B*T, K]
        labels = labels.to(device, non_blocking=True).view(B*T)               # [B*T]

        # ---- Object encoder (batched) ----
        obj_feats = obj_enc(obj_imgs)   # [B*T, obj_dim]
        obj_feats = obj_feats.view(B, T, -1)  # [B, T, obj_dim]

        # ---- Pose encoder (vectorized) ----
        # Build big edge_index of size (B*T) graphs, each with N_nodes nodes
        try:
            edge_big = make_batched_edge_index(edges, N_nodes, B*T, device=device)  # [2, (B*T)*E]
            # We expect pose_enc to have conv1/conv2 attributes (PyG layers)
            conv1 = getattr(pose_enc, "conv1")
            conv2 = getattr(pose_enc, "conv2")
            h = conv1(pose_nodes, edge_big)   # [B*T*N, hidden]
            h = F.relu(h)
            h = conv2(h, edge_big)            # [B*T*N, out_dim]
            h = h.view(B, T, N_nodes, -1)     # [B, T, N, out_dim]
            pose_feats = h.mean(dim=2)        # [B, T, out_dim]
        except Exception as e:
            # Fallback: process per-frame but still in batch over B
            # For t in [0..T): feed (B, N, 3) at once through pose_enc (if it supports batch)
            print(f"[Extract] ⚠️ Pose fast path failed ({e}), falling back to per-frame batched over B.")
            pose_list = []
            for t in range(T):
                # take B graphs for frame t: shape [B, N, 3]
                nodes_t = pose_nodes.view(B, T, N_nodes, 3)[:, t]  # [B, N, 3]
                # If pose_enc supports batch of graphs represented as list, some impls may accept nodes_t directly
                try:
                    out_t = pose_enc(nodes_t, edges)  # expect [B, out_dim] or [B,1,out_dim]
                except Exception:
                    # process each sample individually (slow)
                    out_frames = []
                    for b in range(B):
                        out_b = pose_enc(nodes_t[b], edges)
                        if out_b.ndim == 1: out_b = out_b.unsqueeze(0)
                        out_frames.append(out_b.squeeze(0))
                    out_t = torch.stack(out_frames, dim=0)
                if out_t.ndim == 1:
                    out_t = out_t.unsqueeze(0)
                pose_list.append(out_t)
            pose_feats = torch.stack(pose_list, dim=1)  # [B, T, out_dim]

        # ---- Kin_feats already [B*T, K] -> reshape to [B,T,K] ----
        kin_feats = kin_feats.view(B, T, K)

        # ---- Build per-sequence fused features and add to buffer ----
        for b in range(B):
            fused = torch.cat([obj_feats[b].cpu(), pose_feats[b].cpu(), kin_feats[b].cpu()], dim=-1)  # [T, D]
            lbl = labels.view(B, T)[b].cpu()
            sess = sessions[b]
            buffer.append({
                "features": fused,   # [T, D]
                "labels": lbl,       # [T]
                "session": sess
            })
            total_sequences += 1

        # cleanup GPU memory references
        del obj_imgs, obj_feats, pose_nodes, pose_feats, kin_feats, labels
        torch.cuda.empty_cache()

        # save block if buffer large enough
        if len(buffer) >= save_block or total_sequences == len(ds.windows):
            block_id = saved_blocks
            out_path = os.path.join(args.out_dir, f"block_{block_id:03d}.pt")
            torch.save(buffer, out_path)
            print(f"[Extract] Saved block {block_id} ({len(buffer)} sequences) -> {out_path}")
            buffer = []
            saved_blocks += 1

    elapsed = time.time() - t0
    print(f"[Extract] Done. Total sequences: {total_sequences}. Time: {elapsed/60:.2f} min. Blocks saved: {saved_blocks}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/features_stageC_v3")
    p.add_argument("--obj_ckpt", type=str, default=None)
    p.add_argument("--pose_ckpt", type=str, default=None)
    p.add_argument("--obj_dim", type=int, default=128)
    p.add_argument("--pose_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--kin_dim", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--gnn_type", choices=["gcn", "sage"], default="gcn")
    p.add_argument("--batch", type=int, default=4, help="num sequences per batch for DataLoader")
    p.add_argument("--workers", type=int, default=4, help="num workers for DataLoader")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--save_block", type=int, default=20, help="how many sequences to buffer before saving a single file")
    args = p.parse_args()

    extract_features_v3(args)
