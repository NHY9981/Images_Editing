import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import random
from pathlib import Path
import albumentations as A
import sys

# 设置随机种子以确保可重复性
random.seed(42)
np.random.seed(42)

# 配置
INPUT_CSV = '/root/autodl-tmp/csv/dfgc_train.csv'
AUGMENTED_ROOT = 'augmented'  # 增强数据的根目录名
OUTPUT_CSV = '/root/autodl-tmp/csv/dfgc_train_balanced.csv'
AUGMENTATION_FACTOR = 7  # 每个真实样本生成的增强样本数量

# 读取原始CSV
df = pd.read_csv(INPUT_CSV, header=None)
df.columns = ['path', 'label']

# 分离类别
real_samples = df[df['label'] == 0]
fake_samples = df[df['label'] == 1]

print(f"原始数据集统计:")
print(f"真实样本(类别0): {len(real_samples)}")
print(f"伪造样本(类别1): {len(fake_samples)}")
print(f"不平衡比例: 1:{len(fake_samples)/len(real_samples):.2f}")

# 定义增强管道
augmentation_pipeline = A.Compose([
    # 空间变换
    A.OneOf([
        A.HorizontalFlip(p=0.8),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
        A.Perspective(scale=(0.01, 0.05), p=0.5),
    ], p=1.0),
    
    # 颜色变换
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.5),
    ], p=0.8),
    
    # 噪声和模糊
    A.OneOf([
        A.GaussianBlur(blur_limit=(1, 3), p=0.5),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.5),
    ], p=0.5),
])

# 高级增强管道
advanced_augmentation = A.Compose([
    A.ElasticTransform(alpha=1, sigma=10, alpha_affine=10, p=0.5),
    A.RandomShadow(p=0.3),
])

def get_augmented_path(original_path):
    """根据原始路径生成增强图像的保存路径，格式为：
    /root/autodl-tmp/Celeb-DF-v2-crop/augmented/Celeb-real/id0_0000/aug_0_12345.png
    """
    path = Path(original_path)
    
    # 解析原始路径结构：/root/autodl-tmp/Celeb-DF-v2-crop/子文件夹/ID/文件名
    parts = path.parts
    if len(parts) < 6:
        raise ValueError(f"无效的路径格式: {original_path}")
    
    # 构建增强路径
    root_dir = Path(parts[0]) / parts[1] / parts[2] # /root/autodl-tmp
    base_dir = parts[3]  # Celeb-DF-v2-crop
    sub_dir = parts[4]   # Celeb-real 或 YouTube-real
    id_dir = parts[5]    # id0_0000 或 00000

    # 创建增强目录结构：/root/autodl-tmp/Celeb-DF-v2-crop/augmented/子文件夹/ID/
    aug_dir = root_dir / base_dir / AUGMENTED_ROOT / sub_dir / id_dir
    aug_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成唯一文件名
    aug_filename = f"aug_{path.stem}_{random.randint(0, 99999):05d}{path.suffix}"
    return str(aug_dir / aug_filename)

def augment_image(img_path, output_path, use_advanced=False):
    """加载、增强并保存图像"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}")
            return False
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if use_advanced:
            augmented = advanced_augmentation(image=img)
        else:
            augmented = augmentation_pipeline(image=img)
            
        augmented_img = augmented['image']
        augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, augmented_img)
        return True
    except Exception as e:
        print(f"处理 {img_path} 时出错: {e}")
        return False

# 初始化新的数据列表 - 先包含原始数据
new_data = df.values.tolist()

# 处理并增强少数类(真实人脸)
for idx, row in tqdm(real_samples.iterrows(), total=len(real_samples), desc="增强真实样本"):
    img_path = row['path']
    
    # 创建增强版本
    for aug_idx in range(AUGMENTATION_FACTOR):
        use_advanced = (aug_idx % 3 == 0)  # 30%使用高级增强
        
        # 生成增强图像的保存路径
        augmented_path = get_augmented_path(img_path)
        
        # 执行增强并保存
        if augment_image(img_path, augmented_path, use_advanced):
            new_data.append([augmented_path, 0])

# 创建新的DataFrame (保持与原始CSV相同的格式)
new_df = pd.DataFrame(new_data, columns=['path', 'label'])

# 保存新的CSV文件 (不包含header和index，与原始格式一致)
new_df.to_csv(OUTPUT_CSV, header=False, index=False)

# 打印统计信息
real_count = len(new_df[new_df['label'] == 0])
fake_count = len(new_df[new_df['label'] == 1])

print(f"\n新数据集统计:")
print(f"真实样本(类别0): {real_count}")
print(f"伪造样本(类别1): {fake_count}")
print(f"新比例: 1:{fake_count/real_count:.2f}")
print(f"增强数据集已保存到 {OUTPUT_CSV}")
print(f"增强图片存储在: {Path(df.iloc[0]['path']).parents[2]}/{AUGMENTED_ROOT}/")