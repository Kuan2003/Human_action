# src/train_stageA.py
import os
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms

from models import ObjectEncoder  # dùng model bạn có sẵn
from preprocess_ultils import load_rgb, load_mask
from path_utils import SmartPathResolver
from environment_config import setup_environment


# ----------------------------
# Dataset cho Stage A (frame-level)
# ----------------------------
class ObjectFrameDataset(Dataset):
    def __init__(self, dataset_json, transform=None, drop_no_mask=True):
        # Smart path resolution
        self.path_resolver = SmartPathResolver(anchor_file=dataset_json)
        print(f"[StageA] Detected project root: {self.path_resolver.get_project_root()}")
        
        with open(dataset_json) as f:
            items = json.load(f)

        self.items = []
        for it in items:
            if drop_no_mask and not it.get("has_mask", True):
                continue
            # Resolve paths
            for path_key in ["rgb_path", "mask_path"]:
                if path_key in it:
                    it[path_key] = self.path_resolver.resolve(it[path_key], must_exist=False)
            base_dir = os.path.dirname(os.path.abspath(dataset_json))

            rgb_path = it["rgb_path"]
            mask_path = it["mask_path"]

            # Nếu path là tương đối nhưng bắt đầu bằng "processed_data_auto", chỉ giữ phần sau
            if not os.path.isabs(rgb_path):
                if rgb_path.startswith("processed_data_auto/"):
                    rgb_path = os.path.join(os.path.dirname(base_dir), rgb_path)
                else:
                    rgb_path = os.path.join(base_dir, rgb_path)

            if not os.path.isabs(mask_path):
                if mask_path.startswith("processed_data_auto/"):
                    mask_path = os.path.join(os.path.dirname(base_dir), mask_path)
                else:
                    mask_path = os.path.join(base_dir, mask_path)

            it["rgb_path"] = rgb_path
            it["mask_path"] = mask_path

            # Chỉ giữ frame hợp lệ
            if not os.path.exists(it["rgb_path"]):
                continue
            if not os.path.exists(it["mask_path"]):
                continue

            self.items.append(it)
        labels = sorted({it["action"] for it in self.items})
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
            obj_crop = rgb[y0:y1, x0:x1]
        else:
            obj_crop = rgb

        obj_crop = cv2.resize(obj_crop, (128, 128))
        img = self.transform(obj_crop)

        label = self.label2id.get(it.get("action", "None"), self.label2id.get("None", 0))
        return img, label, it.get("session", "default")


# ----------------------------
# Tách train / val theo session
# ----------------------------
def session_split_indices(dataset, val_ratio=0.2, seed=42):
    sess2idx = {}
    for i, it in enumerate(dataset.items):
        sess = it.get("session", "default")
        sess2idx.setdefault(sess, []).append(i)

    sessions = list(sess2idx.keys())
    random.Random(seed).shuffle(sessions)

    total = len(dataset)
    val_indices = []
    cur = 0
    for s in sessions:
        if cur / total >= val_ratio:
            break
        val_indices.extend(sess2idx[s])
        cur += len(sess2idx[s])

    train_indices = [i for i in range(total) if i not in val_indices]
    return train_indices, val_indices


# ----------------------------
# Training Loop
# ----------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ObjectFrameDataset(args["dataset_json"])
    print(f"[A] Dataset loaded: {len(ds)} frames, classes: {ds.label2id}")

    # split train/val theo session
    train_idx, val_idx = session_split_indices(ds, val_ratio=0.2, seed=123)
    train_loader = DataLoader(ds, batch_size=args["batch"], sampler=SubsetRandomSampler(train_idx),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(ds, batch_size=args["batch"], sampler=SubsetRandomSampler(val_idx),
                            num_workers=4, pin_memory=True)

    # class weight
    all_labels = [ds.label2id[it["action"]] for it in ds.items]
    cnt = Counter(all_labels)
    weights = torch.tensor([1.0 / (cnt.get(i, 1)) for i in range(len(ds.label2id))], dtype=torch.float)
    weights = weights.to(device)

    model = ObjectEncoder(out_dim=args["obj_dim"], pretrained=True).to(device)
    head = nn.Linear(args["obj_dim"], len(ds.label2id)).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs(args["out_dir"], exist_ok=True)
    best_acc = 0.0

    for epoch in range(args["epochs"]):
        model.train()
        head.train()
        total_loss, total_correct, total_frames = 0, 0, 0

        for imgs, labels, _ in tqdm(train_loader, desc=f"[A] Epoch {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device)

            feats = model(imgs)
            logits = head(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_frames += imgs.size(0)

        train_acc = total_correct / total_frames
        scheduler.step()
        print(f"[A][Train] Epoch {epoch} | Loss: {total_loss/total_frames:.4f} | Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        head.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = model(imgs)
                logits = head(feats)
                loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total
        print(f"[A][Val] Epoch {epoch} | Loss: {val_loss/val_total:.4f} | Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "encoder": model.state_dict(),
                "head": head.state_dict(),
                "label2id": ds.label2id
            }, os.path.join(args["out_dir"], "obj_encoder_best.pth"))
            print(f"[A] ✅ Saved best model (val_loss={best_loss:.4f})")
        
        log_path = os.path.join(args["out_dir"], "train_log.txt")
        with open(log_path, "a") as f:
            f.write(f"{datetime.datetime.now()} | epoch={epoch} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | train_loss={total_loss/total_frames:.4f}|val_loss={val_loss/val_total:.4f}\n")

    print(f"[A] Training finished. Best loss = {best_loss:.4f}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--obj_dim", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="outputs_stageA")
    args = vars(parser.parse_args())

    train(args)
