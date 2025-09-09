# -*- coding: utf-8 -*-
"""
从如下结构的原始数据集中抽取小样本，并按 80/20 划分为 train_small/ 和 test_small/：
ROOT/
  backpack/
    PACKED/*.npy
    RAW/*.npy
    RGB/*.png
    RGB_RESIZED/*.png
  bicycle/...
  car/...
  person/...

输出：
OUT_ROOT/
  train_small/
    backpack/PACKED|RAW|RGB|RGB_RESIZED/...
    bicycle/...
    car/...
    person/...
  test_small/
    backpack/PACKED|RAW|RGB|RGB_RESIZED/...
    ...

注意：
1) “原始数据集根目录（SRC_DIR）” 就是 backpack/bicycle/car/person 的外面一层；
2) 第四种模态名按你的实际：RGB_RESIZED（大写）；
3) 为了避免混淆，本脚本默认输出到 OUT_ROOT 下的 train_small/、test_small/（不会动你的原数据）。

把 SRC_DIR 改成 包含 backpack/bicycle/car/person 的那一层；
把 OUT_ROOT 改成你想放“小数据集”的目录；
设置每个“类别-模态”要抽多少张（N_PER_CLASS_PER_MODALITY），脚本会同时拆成 80/20；
跑：python make_small_and_split_nested.py；
在训练脚本里把 train_dir 指到 OUT_ROOT/train_small，test_dir 指到 OUT_ROOT/test_small 就能快速测试。
"""

import os
import random
import shutil
from pathlib import Path

# =================== 需要你修改的参数 ===================
SRC_DIR   = r"D:\EESSC\archive\dataset_v5\train"   # 例如：D:/datasets/PASCAL_RAW
OUT_ROOT  = r"D:\EESSC\ResNet18\subset"        # 例如：D:/datasets/PASCAL_RAW_SUBSET
CLASSES   = ["backpack", "bicycle", "car", "person"]
MODALITIES= ["PACKED", "RAW", "RGB", "RGB_RESIZED"]  # 注意第四种是 RGB_RESIZED（大写）
N_PER_CLASS_PER_MODALITY = 300   # 每个“类别-模态”子目录抽取的样本数量；总样本≈ 4类×4模态×N
TRAIN_RATIO = 0.8               # 80/20 划分
SEED = 42
# 允许的扩展名（RGB/RGB_RESIZED 为图片；PACKED/RAW 为 npy。为稳妥都放进来）
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".npy"}
# =======================================================

random.seed(SEED)

def list_files(dir_path: Path):
    """列出一个目录中的所有“有效文件”（按扩展名过滤）"""
    if not dir_path.exists():
        return []
    return [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]

def ensure_clean_dir(p: Path):
    """清空并重建一个目录（谨慎：会删掉已有内容）"""
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def main():
    src = Path(SRC_DIR)
    assert src.exists(), f"SRC_DIR 不存在：{src}"

    out_root = Path(OUT_ROOT)
    out_train = out_root / "train_small"
    out_test  = out_root / "test_small"
    ensure_clean_dir(out_train)
    ensure_clean_dir(out_test)

    total_train = 0
    total_test  = 0

    for cls in CLASSES:
        for mod in MODALITIES:
            src_dir = src / cls / mod
            files = list_files(src_dir)
            if not files:
                print(f"[跳过] {cls}/{mod} 无文件或目录不存在：{src_dir}")
                continue

            random.shuffle(files)
            # 抽样数量 = min(配置N, 实际可用数量)
            k = min(N_PER_CLASS_PER_MODALITY, len(files))
            subset = files[:k]

            # 按 80/20 划分
            n_train = int(round(k * TRAIN_RATIO))
            train_files = subset[:n_train]
            test_files  = subset[n_train:]

            # 目标目录：保持 类别/模态 层级
            dst_train = out_train / cls / mod
            dst_test  = out_test  / cls / mod
            dst_train.mkdir(parents=True, exist_ok=True)
            dst_test.mkdir(parents=True,  exist_ok=True)

            # 复制文件
            for p in train_files:
                shutil.copy2(p, dst_train / p.name)
            for p in test_files:
                shutil.copy2(p, dst_test / p.name)

            total_train += len(train_files)
            total_test  += len(test_files)
            print(f"[完成] {cls}/{mod}: 抽样 {k} -> 训练 {len(train_files)} | 测试 {len(test_files)}")

    print("\n===== 全部完成 =====")
    print(f"训练集总样本：{total_train}")
    print(f"测试集总样本：{total_test}")
    print(f"训练集目录：{out_train.resolve()}")
    print(f"测试集目录：{out_test.resolve()}")
    print("提示：把训练代码里的 train_dir/test_dir 分别指向以上两个目录即可。")

if __name__ == "__main__":
    main()
