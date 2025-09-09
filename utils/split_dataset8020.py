# -*- coding: utf-8 -*-
"""
按 80/20 划分数据集：
假设数据结构为：
ROOT/
  backpack/
    PACKED/ RAW/ RGB/ RGB_size/
  bicycle/
    PACKED/ RAW/ RGB/ RGB_size/
  car/
    PACKED/ RAW/ RGB/ RGB_size/
  person/
    PACKED/ RAW/ RGB/ RGB_size/

输出：
ROOT/train/...
ROOT/test/...
"""

import os
import shutil
import random

# ========== 参数设置 ==========
ROOT = r"D:\EESSC\archive\dataset_v5"   # 👉 在这里写绝对路径，例如 "D:/datasets/mydata"
TRAIN_RATIO = 0.8
SEED = 42
CLASSES = ["backpack", "bicycle", "car", "person"]
MODALITIES = ["PACKED", "RAW", "RGB", "RGB_RESIZED"]
VALID_EXTS = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
OUT_TRAIN = os.path.join(ROOT, "train")
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
    # 清空并创建 train/test 根目录
    ensure_clean_dir(OUT_TRAIN)
    ensure_clean_dir(OUT_TEST)

    total_train = 0
    total_test = 0

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
            random.shuffle(files)

            k = int(len(files) * TRAIN_RATIO)
            train_files = files[:k]
            test_files  = files[k:]

            dst_train_dir = os.path.join(OUT_TRAIN, cls, mod)
            dst_test_dir  = os.path.join(OUT_TEST,  cls, mod)
            os.makedirs(dst_train_dir, exist_ok=True)
            os.makedirs(dst_test_dir, exist_ok=True)

            for f in train_files:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_train_dir, f))
            for f in test_files:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_test_dir, f))

            print(f"[完成] {cls}/{mod}: 总数 {len(files)} -> 训练 {len(train_files)}, 测试 {len(test_files)}")
            total_train += len(train_files)
            total_test  += len(test_files)

    print("\n========== 划分完成 ==========")
    print(f"训练集总文件数: {total_train}")
    print(f"测试集总文件数: {total_test}")
    print(f"训练目录: {OUT_TRAIN}")
    print(f"测试目录: {OUT_TEST}")

if __name__ == "__main__":
    main()
