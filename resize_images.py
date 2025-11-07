"""
批量调整图片尺寸为 640x640
"""
import os
from PIL import Image
import cv2

# 设置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAT_TEST_DIR = os.path.join(SCRIPT_DIR, "cat_test")
TARGET_SIZE = (640, 640)

print("=" * 70)
print("批量调整图片尺寸")
print("=" * 70)

print(f"\n目标资料夹: {CAT_TEST_DIR}")
print(f"目标尺寸: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

# 获取所有图片文件
image_files = [f for f in os.listdir(CAT_TEST_DIR) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

print(f"\n找到 {len(image_files)} 张图片")

if len(image_files) == 0:
    print("❌ 没有找到图片文件")
    exit(1)

# 处理每张图片
success_count = 0
for i, filename in enumerate(image_files, 1):
    file_path = os.path.join(CAT_TEST_DIR, filename)
    
    try:
        # 使用 PIL 读取图片
        img = Image.open(file_path)
        original_size = img.size
        
        # 调整尺寸（使用 LANCZOS 高质量重采样）
        img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        # 保存（覆盖原文件）
        img_resized.save(file_path, quality=95)
        
        print(f"[{i}/{len(image_files)}] ✅ {filename}")
        print(f"        {original_size[0]}x{original_size[1]} → {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
        
        success_count += 1
        
    except Exception as e:
        print(f"[{i}/{len(image_files)}] ❌ {filename}: {e}")

print("\n" + "=" * 70)
print(f"处理完成: {success_count}/{len(image_files)} 张图片成功调整")
print("=" * 70)

# 验证结果
print("\n验证调整结果:")
for filename in image_files[:3]:  # 验证前3张
    file_path = os.path.join(CAT_TEST_DIR, filename)
    img = Image.open(file_path)
    print(f"  {filename}: {img.size[0]}x{img.size[1]}")
