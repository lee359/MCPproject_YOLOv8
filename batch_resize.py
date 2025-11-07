from PIL import Image
import os

folders = ['dog_B', 'dog_NB', 'cat_NB']
total_success = 0
total_files = 0

for folder in folders:
    if not os.path.exists(folder):
        print(f"❌ 資料夾不存在: {folder}")
        continue
    
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"\n📁 {folder}: 找到 {len(images)} 張圖片")
    
    for img_name in images:
        img_path = os.path.join(folder, img_name)
        try:
            img = Image.open(img_path)
            original_size = img.size
            img_resized = img.resize((640, 640), Image.Resampling.LANCZOS)
            img_resized.save(img_path, quality=95)
            total_success += 1
            total_files += 1
        except Exception as e:
            print(f"  ❌ {img_name}: {e}")
            total_files += 1

print(f"\n{'='*60}")
print(f"✅ 處理完成: {total_success}/{total_files} 張圖片成功調整為 640x640")
print(f"{'='*60}")

# 驗證結果
print("\n驗證結果:")
for folder in folders:
    if os.path.exists(folder):
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if images:
            sample = images[0]
            img = Image.open(os.path.join(folder, sample))
            print(f"  {folder}/{sample}: {img.size[0]}x{img.size[1]}")
