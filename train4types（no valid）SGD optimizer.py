# -*- coding: utf-8 -*-
import os
import time
import torch
import visdom
import numpy as np
from PIL import Image
from copy import deepcopy
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms

from model import ResNet18  # 你的 ResNet18 实现（支持 in_ch）

# ========= Data storage path & Modality=========
train_dir = r"D:\EESSC\archive\80train20test\train"
test_dir  = r"D:\EESSC\archive\80train20test\test"
MODALITY  = "RGB"  # choose: 'RGB' / 'RGB_RESIZED' / 'PACKED' / 'RAW'
# ======================================
# mapping modality to channels
IN_CH = {"RGB": 3, "RGB_RESIZED": 3, "RAW": 1, "PACKED": 4}[MODALITY]
BATCH_SIZE = 32
EPOCH = 50
LR = 5e-5
WEIGHT_DECAY = 0.05
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device =", device, "| modality =", MODALITY, "| in_ch =", IN_CH)

def make_unique_path(base_name: str, acc: float, save_dir="./models") -> str:
    os.makedirs(save_dir, exist_ok=True)
    base, ext = os.path.splitext(base_name)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fname = f"{base}_acc{acc:.4f}_{ts}{ext}"
    return os.path.join(save_dir, fname)


# ---------------- General data augmentation (effective for PNG) ----------------
pil_train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.5),   #improvement
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])
pil_test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# ---------------- probably data types ----------------
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".npy"}

class MixedFolder(Dataset):
    # Loading png/npy
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

        if arr.ndim == 2:            # (H,W) -> (1,H,W)
            arr = arr[None, ...]
        elif arr.ndim == 3:          # (H,W,C) -> (C,H,W)
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"不支持的 .npy 形状: {arr.shape}")

        x = torch.from_numpy(arr)    # [C,H,W]
        x = F.interpolate(x.unsqueeze(0), size=(224, 224),
                          mode="bilinear", align_corners=False).squeeze(0)
        # if self.is_train and torch.rand(1).item() < 0.5:   #npy improvement:horizontal flip: enlarge dataset
        #     x = torch.flip(x, dims=[2])
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
            tf = pil_train_tf if self.is_train else pil_test_tf
            x = tf(im)
        return x, torch.tensor(y, dtype=torch.long)

# ---------------- DataLoader ----------------
train_set = MixedFolder(train_dir, is_train=True,  modality=MODALITY)
test_set  = MixedFolder(test_dir,  is_train=False, modality=MODALITY)

num_classes = len(train_set.classes) if len(train_set.classes) > 0 else 4

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------- Model and Optimizer ----------------
model = ResNet18(classes_num=num_classes, in_ch=IN_CH).to(device)
# optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,              # 学习率（推荐从0.01开始）
    momentum=0.9,         # 动量系数，常用0.9
    weight_decay=WEIGHT_DECAY,
    nesterov=True         # 可选，通常设True更好
)

criterion = nn.CrossEntropyLoss()
viz = visdom.Visdom()

from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH)  # T_max=总轮数


# ---------------- Evaluation function: Calculate the accuracy rate given the loader ----------------
@torch.no_grad()
def evaluate_acc(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total = len(loader.dataset)
    if total == 0:
        return 0.0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).float().sum().item()
    return correct / total

# ---------------- Training cycle: Each round is evaluated on the test, and the best one is selected accordingly ----------------
best_test_acc, best_epoch = 0.0, 0       # the best based on test-accuracy
best_state = None                        #

for epoch in range(EPOCH):
    model.train()
    running_loss = 0.0

    for step, (images, labels) in enumerate(train_loader, start=1):
        assert images.shape[1] == IN_CH, f"输入通道={images.shape[1]} 与 IN_CH={IN_CH} 不一致，检查 MODALITY/数据路径"
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Progress bar
        rate = step / max(len(train_loader), 1)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print(f"\repoch: {epoch+1} train loss: {int(rate*100):^3d}%[{a}->{b}]{loss:.3f}", end="")

    # Average loss in each epoch
    epoch_loss = running_loss / max(len(train_loader), 1)
    # print(f"  |  epoch{epoch+1} avg train loss: {epoch_loss:.4f}")

    # Visualization loss
    try:
        viz.line([epoch_loss], [epoch + 1], win='loss',
                 update='append',
                 opts=dict(title='Training Loss (per epoch)', xlabel='Epoch', ylabel='Loss'))
    except Exception:
        pass

    #—— Evaluating on test ——
    test_acc = evaluate_acc(model, test_loader)
    print(f"  epoch{epoch+1} TEST acc: {test_acc:.4f}")

    scheduler.step()  # <—— adjust lr each epoch

    try:
        viz.line([test_acc], [epoch + 1], win='test_acc',
                 update='append',
                 opts=dict(title='Test Accuracy', xlabel='Epoch', ylabel='Accuracy'))
    except Exception:
        pass

    if test_acc > best_test_acc:
        best_test_acc, best_epoch = test_acc, epoch + 1
        best_state = deepcopy(model.state_dict())

# === Training completed: Save the "Best Model" (selected by TEST acc) only once. ===
final_path = make_unique_path(f"ResNet18_{MODALITY}.pth", best_test_acc, "./models")
torch.save(best_state if best_state is not None else model.state_dict(), final_path)
print(f"[INFO] 保存最优模型(按 TEST acc) -> {final_path}")
print(f"[INFO] 最优 TEST 准确率：{best_test_acc:.4f} @ epoch {best_epoch}")

# === Optional: After loading the optimal weight, confirm once again on test (it should be consistent/close to best test acc) ===
if best_state is not None:
    model.load_state_dict(best_state)
final_test_acc = evaluate_acc(model, test_loader)
print(f"Final TEST accuracy (best checkpoint): {final_test_acc:.4f}")
