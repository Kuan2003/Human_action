#!/usr/bin/env python3
"""
Stage D — Fine-tune end-to-end (unfreeze last K encoder layers)
"""
import os, argparse
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import datetime
from datasets import HOISequenceDataset
import models as models_module
from environment_config import setup_environment

# Expect HOI_FullModel or HOIModel or components
HOIModel = getattr(models_module, "HOI_FullModel", None) or getattr(models_module, "HOIModel", None)

if HOIModel is None:
    # fallback: check components exist
    raise ImportError("Please provide HOI_FullModel or HOIModel in src.models for Stage D")

def session_split(ds, val_ratio=0.2, seed=123):
    sess2idx={}
    for i,w in enumerate(ds.windows):
        sess=w[0]; sess2idx.setdefault(sess, []).append(i)
    sessions=list(sess2idx.keys()); import random; random.Random(seed).shuffle(sessions)
    total=len(ds.windows); val_idx=[]; cur=0
    for s in sessions:
        if cur/total >= val_ratio: break
        val_idx+=sess2idx[s]; cur+=len(sess2idx[s])
    train_idx=[i for i in range(total) if i not in val_idx]
    return train_idx, val_idx

def train(args):
    setup_environment()
    
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    
    # Setup TensorBoard
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[D] TensorBoard logs will be saved to: {log_dir}")
    
    ds = HOISequenceDataset(args.dataset_json, seq_len=args.seq_len, stride=args.stride)
    train_idx, val_idx = session_split(ds, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = DataLoader(ds, batch_size=args.batch, sampler=SubsetRandomSampler(train_idx),
                              collate_fn=lambda x:x, num_workers=args.workers)
    val_loader = DataLoader(ds, batch_size=args.batch, sampler=SubsetRandomSampler(val_idx),
                            collate_fn=lambda x:x, num_workers=args.workers)

    model = HOIModel(obj_dim=args.obj_dim, pose_dim=args.pose_dim, kin_dim=args.kin_dim,
                     hidden=args.hidden, num_classes=len(ds.label2id)).to(device)

    # load saved weights if exist
    if args.stageA_ckpt and os.path.exists(args.stageA_ckpt):
        d = torch.load(args.stageA_ckpt, map_location=device)
        # expected dict with "model" or state dict
        if isinstance(d, dict) and ("model" in d or "obj_enc" in d):
            if "model" in d:
                model.obj_enc.load_state_dict(d["model"])
            if "obj_enc" in d:
                model.obj_enc.load_state_dict(d["obj_enc"])
        else:
            try:
                model.obj_enc.load_state_dict(d)
            except Exception:
                pass
    if args.stageB_ckpt and os.path.exists(args.stageB_ckpt):
        d = torch.load(args.stageB_ckpt, map_location=device)
        if isinstance(d, dict) and ("model" in d or "pose_enc" in d):
            if "model" in d:
                model.pose_enc.load_state_dict(d["model"])
            if "pose_enc" in d:
                model.pose_enc.load_state_dict(d["pose_enc"])
        else:
            try:
                model.pose_enc.load_state_dict(d)
            except Exception:
                pass
    if args.stageC_ckpt and os.path.exists(args.stageC_ckpt):
        d = torch.load(args.stageC_ckpt, map_location=device)
        # Handle dictionary format from stageC checkpoint
        if isinstance(d, dict) and "fusion" in d:
            model.temporal.load_state_dict(d["fusion"])
        else:
            try:
                model.temporal.load_state_dict(d)
            except Exception:
                pass

    # unfreeze last K layers: we do simple heuristic: unfreeze final fc of obj_enc and final conv of pose_enc
    for name, p in model.named_parameters():
        p.requires_grad = False
        if ("fc" in name or "conv3" in name or "temporal" in name):
            p.requires_grad = True

    # prepare optimizer with param groups
    params = [
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and ("obj_enc" in n or "pose_enc" in n)], "lr": args.lr_enc},
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and ("temporal" in n)], "lr": args.lr_lstm}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    best=-1.0
    for epoch in range(args.epochs):
        model.train(); total=0; corr=0
        pbar = tqdm(train_loader, desc=f"[D] Epoch {epoch}")
        for batch in pbar:
            B = len(batch); T = args.seq_len
            obj_imgs = torch.stack([item["obj_imgs"] for item in batch], dim=0).to(device)
            pose_nodes = torch.stack([item["pose_nodes"] for item in batch], dim=0).to(device)
            kin = torch.stack([item["kin_feats"] for item in batch], dim=0).to(device)
            labels = torch.stack([item["label"] for item in batch], dim=0).to(device)

            logits = model(obj_imgs, pose_nodes, kin)  # [B,T,C]
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = logits.argmax(-1)
            corr += (preds==labels).sum().item(); total += B*T
            pbar.set_postfix(loss=loss.item(), acc=corr/max(1,total))
        # val
        model.eval(); v_corr=0; v_tot=0; v_loss=0.0
        with torch.no_grad():
            for batch in val_loader:
                B = len(batch)
                obj_imgs = torch.stack([item["obj_imgs"] for item in batch], dim=0).to(device)
                pose_nodes = torch.stack([item["pose_nodes"] for item in batch], dim=0).to(device)
                kin = torch.stack([item["kin_feats"] for item in batch], dim=0).to(device)
                labels = torch.stack([item["label"] for item in batch], dim=0).to(device)
                logits = model(obj_imgs, pose_nodes, kin)
                v_loss += criterion(logits.view(-1, logits.shape[-1]), labels.view(-1)).item() * B
                preds = logits.argmax(-1)
                v_corr += (preds==labels).sum().item(); v_tot += B*args.seq_len
        val_acc = v_corr/max(1,v_tot)
        val_loss_avg = v_loss/v_tot
        train_acc = corr/max(1,total)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', loss.item(), epoch)
        writer.add_scalar('Loss/Validation', val_loss_avg, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate/Encoder', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Learning_Rate/LSTM', optimizer.param_groups[1]['lr'], epoch)
        
        print(f"[D] Epoch {epoch} | Train: loss={loss.item():.4f}, acc={train_acc:.4f} | Val: loss={val_loss_avg:.4f}, acc={val_acc:.4f}")
        
        if val_acc > best:
            best = val_acc
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss_avg,
                "train_acc": train_acc,
                "args": vars(args)
            }, os.path.join(args.out_dir, "finetune_best.pth"))
            print(f"[D] ✅ Saved best model (val_acc={best:.4f})")
            
        # Log to text file
        log_path = os.path.join(args.out_dir, "train_log.txt")
        with open(log_path, "a") as f:
            f.write(f"{datetime.datetime.now()} | epoch={epoch} | train_loss={loss.item():.4f} | train_acc={train_acc:.4f} | val_loss={val_loss_avg:.4f} | val_acc={val_acc:.4f}\n")
    # Close TensorBoard writer
    writer.close()
    print(f"[D] Training completed! Best val acc: {best:.4f}")
    print(f"[D] TensorBoard logs saved to: {log_dir}")
    print(f"[D] To view: tensorboard --logdir {log_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json", default="dataset.json")
    p.add_argument("--out_dir", default="outputs/stageD")
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr_enc", type=float, default=1e-5)
    p.add_argument("--lr_lstm", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--obj_dim", type=int, default=128)
    p.add_argument("--pose_dim", type=int, default=128)
    p.add_argument("--kin_dim", type=int, default=4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--obj_ckpt", default="outputs/stageA/obj_encoder_best.pth")
    p.add_argument("--pose_ckpt", default="outputs/stageB/pose_gnn_best.pth")
    p.add_argument("--stageA_ckpt", default="outputs/stageA/obj_encoder_best.pth")
    p.add_argument("--stageB_ckpt", default="outputs/stageB/pose_gnn_best.pth")
    p.add_argument("--stageC_ckpt", default="outputs/stageC/fusion_best.pth")   
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    args = p.parse_args()
    train(args)
