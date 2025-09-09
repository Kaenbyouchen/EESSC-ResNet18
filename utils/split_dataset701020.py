# -*- coding: utf-8 -*-
"""
按 70/10/20 划分数据集：
假设数据结构为：
ROOT/
  backpack/
    PACKED/ RAW/ RGB/ RGB_RESIZED/
  bicycle/
    PACKED/ RAW/ RGB/ RGB_RESIZED/
  car/
    PACKED/ RAW/ RGB/ RGB_RESIZED/
  person/
    PACKED/ RAW/ RGB/ RGB_RESIZED/

输出：
ROOT/train/...
ROOT/val/...
ROOT/test/...
"""

import os
import shutil
import random

# ========== 参数设置 ==========
ROOT = r"D:\EESSC\archive\dataset_v5"   # 👉 改成你的绝对路径
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
TEST_RATIO  = 0.2
SEED = 42

CLASSES = ["backpack", "bicycle", "car", "person"]
MODALITIES = ["PACKED", "RAW", "RGB", "RGB_RESIZED"]
VALID_EXTS = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

OUT_TRAIN = os.path.join(ROOT, "train")
OUT_VAL   = os.path.join(ROOT, "val")
OUT_TEST  = os.path.join(ROOT, "test")
# ==============================

random.seed(SEED)

def is_valid_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in VALID_EXTS

def ensure_clean_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def main():
    # 简单校验比例
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例之和应为1，但当前为 {total_ratio}")

    # 清空并创建 train/val/test 根目录
    ensure_clean_dir(OUT_TRAIN)
    ensure_clean_dir(OUT_VAL)
    ensure_clean_dir(OUT_TEST)

    total_train = 0
    total_val   = 0
    total_test  = 0

    for cls in CLASSES:
        cls_dir = os.path.join(ROOT, cls)
        if not os.path.isdir(cls_dir):
            print(f"[跳过] 找不到类别目录: {cls_dir}")
            continue

        for mod in MODALITIES:
            src_dir = os.path.join(cls_dir, mod)
            if not os.path.isdir(src_dir):
                print(f"[跳过] {cls}/{mod} 不存在")
                continue

            files = [f for f in os.listdir(src_dir) if is_valid_file(os.path.join(src_dir, f))]
            if not files:
                print(f"[跳过] {cls}/{mod} 无有效文件")
                continue

            random.shuffle(files)

            n = len(files)
            n_train = int(n * TRAIN_RATIO)
            n_val   = int(n * VAL_RATIO)
            n_test  = n - n_train - n_val   # 剩余给 test，避免四舍五入丢文件

            train_files = files[:n_train]
            val_files   = files[n_train:n_train + n_val]
            test_files  = files[n_train + n_val:]

            # 目标目录：保持 类别/模态 层级
            dst_train_dir = os.path.join(OUT_TRAIN, cls, mod)
            dst_val_dir   = os.path.join(OUT_VAL,   cls, mod)
            dst_test_dir  = os.path.join(OUT_TEST,  cls, mod)
            os.makedirs(dst_train_dir, exist_ok=True)
            os.makedirs(dst_val_dir,   exist_ok=True)
            os.makedirs(dst_test_dir,  exist_ok=True)

            # 复制文件
            for f in train_files:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_train_dir, f))
            for f in val_files:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_val_dir, f))
            for f in test_files:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_test_dir, f))

            print(f"[完成] {cls}/{mod}: 总 {n} -> 训练 {len(train_files)}, 验证 {len(val_files)}, 测试 {len(test_files)}")
            total_train += len(train_files)
            total_val   += len(val_files)
            total_test  += len(test_files)

    print("\n========== 划分完成 ==========")
    print(f"训练集总文件数: {total_train}")
    print(f"验证集总文件数: {total_val}")
    print(f"测试集总文件数: {total_test}")
    print(f"训练目录: {OUT_TRAIN}")
    print(f"验证目录: {OUT_VAL}")
    print(f"测试目录: {OUT_TEST}")

if __name__ == "__main__":
    main()
