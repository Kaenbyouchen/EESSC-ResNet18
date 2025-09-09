# test.py
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from PIL import Image
import json

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import ResNet18  # 你的ResNet18实现（支持 in_ch 参数）

# ----------------- 参数 -----------------

def get_args():
    with open("config_test.json", "r") as f:
        cfg = json.load(f)
    return argparse.Namespace(**cfg)

# ----------------- 模态与输入通道 -----------------
def get_in_ch(modality: str) -> int:
    return {"RGB": 3, "RGB_RESIZED": 3, "RAW": 1, "PACKED": 4}[modality]

# ----------------- 变换（与训练一致的Val/Test变换） -----------------
pil_val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# ----------------- 支持PNG/JPG/NPY的数据集 -----------------
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".npy"}

class MixedFolder(Dataset):
    """ImageFolder风格：支持 png/jpg 与 npy；支持 类别/模态/文件 结构"""
    def __init__(self, root, is_train: bool, modality: str | None = None):
        self.root = root
        self.is_train = is_train
        self.modality = modality

        self.classes = sorted([d for d in os.listdir(root)
                               if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cdir = os.path.join(root, cls)
            if self.modality is not None:
                cdir = os.path.join(cdir, self.modality)
            if not os.path.isdir(cdir):
                print(f"[警告] 跳过：{cdir} 不存在")
                continue
            for name in os.listdir(cdir):
                p = os.path.join(cdir, name)
                if not os.path.isfile(p):
                    continue
                ext = os.path.splitext(p)[1].lower()
                if ext in VALID_EXTS:
                    self.samples.append((p, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def _load_npy(self, path: str) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"不支持的 .npy 形状: {arr.shape}")

        x = torch.from_numpy(arr)
        x = F.interpolate(x.unsqueeze(0), size=(224, 224),
                          mode="bilinear", align_corners=False).squeeze(0)
        x = (x - 0.5) / 0.5
        return x

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            x = self._load_npy(path)
        else:
            with Image.open(path) as im:
                im = im.convert("RGB")
            x = pil_val_tf(im)
        return x, torch.tensor(y, dtype=torch.long)

# ----------------- 评估（准确率 + 混淆矩阵 + 逐类准确率） -----------------
@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    total = len(loader.dataset)
    correct = 0

    # 混淆矩阵：行=真实标签(gt)，列=预测标签(pred)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                    # [B, num_classes]
        pred = logits.argmax(dim=1)          # [B]
        correct += (pred == y).sum().item()

        # 统计混淆矩阵
        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1

    acc = correct / max(total, 1)

    # 逐类准确率（per-class accuracy）：对角线元素 / 该类总样本数
    support = cm.sum(dim=1).clamp_min(1)     # 每类的样本数（避免除0）
    per_class_acc = (cm.diag().float() / support.float()).cpu().numpy()

    return acc, cm.cpu().numpy(), per_class_acc

# ----------------- 主函数 -----------------
def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    in_ch = get_in_ch(args.modality)

    # 测试集
    test_set = MixedFolder(args.test_dir, is_train=False, modality=args.modality)
    num_classes = len(test_set.classes) if len(test_set.classes) > 0 else 4
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )

    # 模型
    model = ResNet18(classes_num=num_classes, in_ch=in_ch).to(device)

    # 加载权重（兼容 DataParallel）
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    new_state = {}
    for k, v in state.items():
        nk = k.replace('module.', '') if k.startswith('module.') else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)

    # 评估
    acc, cm, per_class_acc = evaluate_full(model, test_loader, device, num_classes)

    # 打印结果
    print(f"\nTest set ({args.modality}) accuracy: {acc:.4f}")
    print(f"Classes ({num_classes}): {test_set.classes}")

    # 逐类准确率
    print("\nPer-class accuracy:")
    for i, cls in enumerate(test_set.classes):
        print(f"  {i:2d} - {cls:15s}: {per_class_acc[i]:.4f}")

    # 混淆矩阵
    print("\nConfusion Matrix (rows=GT, cols=Pred):")
    # 友好打印（对齐）
    with np.printoptions(linewidth=160, formatter={'int': lambda x: f"{x:6d}"}):
        print(cm)

    # 可选：保存混淆矩阵为CSV
    if args.save_cm_csv:
        import csv
        os.makedirs(os.path.dirname(args.save_cm_csv), exist_ok=True) if os.path.dirname(args.save_cm_csv) else None
        with open(args.save_cm_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([''] + [f'pred:{c}' for c in test_set.classes])
            for i, row in enumerate(cm):
                writer.writerow([f'gt:{test_set.classes[i]}'] + list(row))
        print(f"\nConfusion matrix saved to: {args.save_cm_csv}")

if __name__ == '__main__':
    main()
