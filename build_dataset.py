import os
import shutil
from pathlib import Path

# ====================== 路径配置 ======================
# 你的原始数据集路径
INPUT_FOLDER = "datasets/Market-1501-v15.09.15/bounding_box_train"
# 生成的YOLO格式数据集
OUTPUT_ROOT = "datasets/market_yolo"
# ======================================================

# 1. 删除旧的错误数据（防止干扰）
if os.path.exists(OUTPUT_ROOT):
    shutil.rmtree(OUTPUT_ROOT)
    print("已删除旧的错误数据集")

# 2. 创建正确的目录结构
Path(f"{OUTPUT_ROOT}/images/train").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_ROOT}/labels/train").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_ROOT}/images/val").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_ROOT}/labels/val").mkdir(parents=True, exist_ok=True)

# 3. 获取所有图片文件
all_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".jpg")]
print(f"发现图片总数：{len(all_files)} 张")

# 4. 划分训练集和验证集（8:2）
split_idx = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

def process_files(file_list, split):
    count = 0
    for fname in file_list:
        src_img = os.path.join(INPUT_FOLDER, fname)
        dst_img = os.path.join(f"{OUTPUT_ROOT}/images/{split}", fname)
        dst_lbl = os.path.join(f"{OUTPUT_ROOT}/labels/{split}", fname.replace(".jpg", ".txt"))

        # 复制图片
        shutil.copy(src_img, dst_img)

        # ✅ 关键：生成正确的行人框标签（不是全屏！）
        # 格式：类别 中心x 中心y 宽度 高度（都归一化到0-1）
        # Market-1501的行人在图片中间，所以框设为：中心0.5, 0.5，宽0.6，高0.85
        label = "0 0.5 0.5 0.6 0.85\n"

        # 写入标签文件
        with open(dst_lbl, "w") as f:
            f.write(label)
        count += 1
    print(f"{split}集处理完成：{count} 张")

# 处理训练集和验证集
print("正在生成训练集...")
process_files(train_files, "train")
print("正在生成验证集...")
process_files(val_files, "val")

print("\n✅ 数据集生成完毕！")
print("结构：")
print(f"{OUTPUT_ROOT}/")
print("├── images/")
print("│   ├── train/")
print("│   └── val/")
print("└── labels/")
print("    ├── train/")
print("    └── val/")