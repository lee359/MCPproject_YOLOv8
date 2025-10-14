import os
from ultralytics import YOLO
from PIL import Image

# ✅ 參數設定
source_image = "123.jpg"               # 輸入圖片路徑
output_dir = "C:\\Users\\user\\runs\\detect"             # 自訂輸出資料夾
imgsz = 640                            # 圖片大小

# ✅ 確保輸出資料夾存在
os.makedirs(output_dir, exist_ok=True)

# ✅ 計算當前已有多少張圖片（用來命名下一張）
existing_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
next_id = len(existing_files) + 1      # 自動從1開始編號

# ✅ 載入模型
model = YOLO("best.pt")

# ✅ 執行預測，不讓YOLO自動儲存
results = model.predict(source=source_image, save=False, imgsz=imgsz)

# ✅ 將推論後的結果圖像儲存下來
for r in results:
    # 將推論結果畫到圖上
    im_array = r.plot()  # numpy array
    im = Image.fromarray(im_array)
    
    # 依序命名為 1.png、2.png...
    save_path = os.path.join(output_dir, f"{next_id}.png")
    im.save(save_path)
    print(f"儲存推論結果至：{save_path}")
