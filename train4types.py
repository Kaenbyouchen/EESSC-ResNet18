# -*- coding: utf-8 -*-
import os
import torch
import visdom
import numpy as np
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from model import ResNet18  # 你的ResNet18(3通道)实现

# ========= 必改：你的数据路径 & 模态 =========
# 目录结构示例：
# train/
#   backpack/  (里边是该模态的 png 或 npy)
#   bicycle/
#   car/
#   person/
# test/
#   backpack/
#
#
#   ...
"""
把 train_dir / test_dir 改成你想跑的模态路径：
把 MODALITY 改成对应字符串，让脚本自动设定 IN_CH（1/3/4）。
其余保持不变，直接运行即可。
运行前在终端运行 conda activate EESSC1
python -m visdom.server -port 8097

"""
train_dir = r"D:\EESSC\ResNet18\subset\train_small"   # 或 /train/RGB /train/PACKED /train/RAW
test_dir  = r"D:\EESSC\ResNet18\subset\test_small"    # 对应的 test 路径
MODALITY  = "RAW"  # 选：'RGB' / 'RGB_RESIZED' / 'PACKED' / 'RAW'



def make_unique_path(base_name: str, acc: float, save_dir="./models") -> str:
    os.makedirs(save_dir, exist_ok=True)
    base, ext = os.path.splitext(base_name)

    # 找到已有的编号
    existing = [f for f in os.listdir(save_dir) if f.startswith(base)]
    max_idx = 0
    for f in existing:
        parts = f.split("_")
        for p in parts:
            if p.isdigit():
                max_idx = max(max_idx, int(p))

    new_idx = max_idx + 1
    fname = f"{base}_{new_idx}_acc{acc:.4f}{ext}"
    return os.path.join(save_dir, fname)



# ==========================================

# 映射模态到输入通道数
IN_CH = {"RGB": 3, "RGB_RESIZED": 3, "RAW": 1, "PACKED": 4}[MODALITY]

BATCH_SIZE = 32                    # 512 太大容易炸显存，先用 128；不够再调
EPOCH = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4    #让参数往 0 方向“收缩”，避免权重无限增大
SAVE_PATH = f"./models/ResNet18_{MODALITY}.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device =", device, "| modality =", MODALITY, "| in_ch =", IN_CH)

# ---------------- 通用数据增强（对PNG有效；NPY走张量增强） ----------------
pil_train_tf = transforms.Compose([  # 用于 训练集，因为加了随机水平翻转（增强数据多样性，防止过拟合）
    transforms.Resize((224, 224)),  # 1. 把图片缩放到 224×224
    # transforms.RandomHorizontalFlip(p=0.5), # 2. improvement：以 50% 概率做水平翻转（数据增强）
    transforms.ToTensor(),  # 3. 转成张量 (C,H,W)，数值范围 [0,1]
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),  # 4. 归一化： (x-0.5)/0.5 => [-1,1]
])
pil_val_tf = transforms.Compose([   # 用于 验证/测试集，没有随机翻转，保证输入固定，结果可复现
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# ---------------- 能同时读取 PNG 和 NPY 的数据集 ----------------
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".npy"}

class MixedFolder(Dataset):
    """ImageFolder 风格的数据集：既支持 png/jpg，也支持 npy (1/3/4通道)；支持 类别/模态/文件 结构"""
    def __init__(self, root, is_train: bool, modality: str | None = None):
        self.root = root
        self.is_train = is_train
        self.modality = modality  # 新增

        # 扫描子文件夹，类别名来自子目录（backpack/bicycle/car/person）
        self.classes = sorted([d for d in os.listdir(root)
                               if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 收集样本：遍历类别，制定模态后进入对应目录，遍历文件过滤目录和非法扩展名，收集文件与标签到samples
        self.samples = []
        for cls in self.classes:
            cdir = os.path.join(root, cls)
            # 如果指定了模态，则进入该模态子目录（如 backpack/RGB_RESIZED）
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


    def __len__(self):  #返回数据集中样本数量
        return len(self.samples)

    def _load_npy(self, path: str) -> torch.Tensor:
        """读取 npy -> [C,H,W]，C∈{1,3,4}，归一化到[-1,1]并resize到224，把它变成模型可以直接吃的 张量"""
        arr = np.load(path).astype(np.float32)
        # min-max到[0,1]
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        # 转 [C,H,W]
        if arr.ndim == 2:                # (H,W) -> (1,H,W)
            arr = arr[None, ...]
        elif arr.ndim == 3:              # (H,W,C) -> (C,H,W)
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"不支持的 .npy 形状: {arr.shape}")

        x = torch.from_numpy(arr)  # [C,H,W], C=1/3/4
        # resize 到 224
        x = F.interpolate(x.unsqueeze(0), size=(224, 224),
                          mode="bilinear", align_corners=False).squeeze(0)
        # improvement：数据增强：训练集随机水平翻转
        # if self.is_train and torch.rand(1).item() < 0.5:
        #     x = torch.flip(x, dims=[2])
        # 归一化到[-1,1]：与 ToTensor()+Normalize(mean=0.5,std=0.5)一致
        x = (x - 0.5) / 0.5
        return x  # [C,224,224]

    def __getitem__(self, idx):
        path, y = self.samples[idx]    #取出当前样本的路径和标签
        ext = os.path.splitext(path)[1].lower()   #拆分文件名和扩展名，比如 "abc.npy" → (".npy")
        if ext == ".npy":
            x = self._load_npy(path)                # [C,224,224], C=1/3/4
        else:
            # png/jpg 用 PIL 流程（固定为3通道）
            with Image.open(path) as im:
                im = im.convert("RGB")
            tf = pil_train_tf if self.is_train else pil_val_tf
            x = tf(im)                               # [3,224,224]
        return x, torch.tensor(y, dtype=torch.long)

# ---------------- 输入通道适配器：C -> 3 ----------------
# class InChAdapter(nn.Module):
#     """把任意通道(C=1/3/4)映射到3通道再喂给你的ResNet18(3ch)"""
#     def __init__(self, in_ch: int):
#         super().__init__()
#         if in_ch == 3:
#             self.map = nn.Identity()
#         else:
#             self.map = nn.Conv2d(in_ch, 3, kernel_size=1, bias=False)
#             nn.init.kaiming_normal_(self.map.weight, mode='fan_out', nonlinearity='relu')
#
#     def forward(self, x):
#         return self.map(x)

# ---------------- DataLoader ----------------
train_set = MixedFolder(train_dir, is_train=True, modality=MODALITY)  #MixedFolder取单个样本，train_set[i] 返回 (x, y)，其中 x=[C,224,224] 的图像张量，y 是类别标签。
test_set  = MixedFolder(test_dir,  is_train=False, modality=MODALITY)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0) #取成批取样本
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------- 模型：适配器 + 你的ResNet18(4类) ----------------
# adapter = InChAdapter(IN_CH)
# backbone = ResNet18(classes_num=4)   # 你的实现，保持3通道输入
# model = nn.Sequential(adapter, backbone).to(device)

model = ResNet18(classes_num=4, in_ch=IN_CH).to(device)

# ---------------- 优化器/损失/可视化 ----------------
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  #优化器 = 根据梯度 + 学习率规则，决定如何更新网络参数
criterion = nn.CrossEntropyLoss()  #损失函数：交叉熵损失
viz = visdom.Visdom()   #启用可视化工具

# ---------------- 评估函数（保留你原逻辑） ----------------
@torch.no_grad()
def evalute(model, loader):
    correct = 0
    total = len(loader.dataset)
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).float().sum().item()
    return correct / max(total, 1)

# ---------------- 训练循环（保留你原逻辑风格） ----------------
best_acc, best_epoch, global_step = 0.0, 0, 0
for epoch in range(EPOCH):
    running_loss = 0.0
    model.train()
    for step, (images, labels) in enumerate(train_loader, start=0):

        assert images.shape[1] == IN_CH, f"输入通道={images.shape[1]} 与 IN_CH={IN_CH} 不一致，检查 MODALITY/数据路径"

        images, labels = images.to(device), labels.to(device)  #把数据搬到 GPU/CPU
        optimizer.zero_grad()  #清空梯度（避免累加）
        outputs = model(images)  #前向计算，得到预测 outputs
        loss = criterion(outputs, labels)   #计算 loss（交叉熵）
        loss.backward()   #反向传播计算梯度
        optimizer.step()   #用优化器更新参数

        # 打印训练进度
        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\repoch: {} train loss: {:^3.0f}%[{}->{}]{:.3f}".format(
            epoch+1, int(rate*100), a, b, loss), end="")
        #可视化训练 loss
        epoch_loss = running_loss / len(train_loader)
    try:
        viz.line([epoch_loss], [epoch + 1], win='loss',
                     update='append', opts=dict(title='Training Loss (per epoch)',
                                                xlabel='Epoch', ylabel='Loss'))
    except Exception:
         pass

    # 每个 epoch 结束在测试集评估
    test_acc = evalute(model, test_loader)
    print("  epoch{} test acc:{}".format(epoch+1, test_acc))
    # 此处注释内容为仅保留accuracy上升的数据画折线图
    # if test_acc > best_acc:
    #     best_acc, best_epoch = test_acc, epoch + 1
    #     torch.save(model.state_dict(), SAVE_PATH)
    #     try:
    #         viz.line([test_acc], [global_step], win='test_acc', update='append', opts=dict(title='Validation Accuracy'))
    #     except Exception:
    #         pass

    # 可视化测试准确率：每个 epoch 都画
    try:
        viz.line([test_acc], [epoch + 1], win='test_acc', update='append',
                 opts=dict(title='Validation Accuracy',
                           xlabel='Epoch', ylabel='Accuracy'))
    except Exception:
        pass

    # 再单独保存最好模型
    if test_acc > best_acc:
        best_acc, best_epoch = test_acc, epoch + 1
        save_path = make_unique_path(f"ResNet18_{MODALITY}.pth", best_acc, "./models")
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] 保存最优模型 -> {save_path}")

print("Finish ! 最优测试准确率：{:.4f} @ epoch {}".format(best_acc, best_epoch))
print("模型已保存到：", SAVE_PATH)
