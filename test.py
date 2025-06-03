from ultralytics import YOLO
import cv2
import numpy as np

# === 1. 載入 YOLOv8 模型 ===
model = YOLO('yolov8n.pt')  # 可替換成 yolov8s.pt 或其他

# === 2. 指定圖片路徑 ===
image_path = 'michael-sum-LEpfefQf4rU-unsplash.jpg'  # <<== 替換為你的圖片檔案名稱
img = cv2.imread(image_path)

# === 3. 推論圖片，降低信心門檻 ===
results = model.predict(source=img, conf=0.25, verbose=False)

# === 4. 印出原始模型預測框（除錯用） ===
print("Detection results (boxes raw data):")
print(results[0].boxes)

# === 5. 類別名稱參照：COCO 資料集中 cat=16, dog=17 ===
target_classes = [16, 17]  # cat=16, dog=17
found_cat = False
found_dog = False
cat_count = 0
dog_count = 0

# 自訂類別名稱，避免亂碼問題
custom_names = {
    16: "Cat",
    17: "Dog"
}

# === 6. 畫出所有被偵測的物件 ===
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # 使用自訂類別名稱，避免亂碼
        if cls in custom_names:
            label = custom_names[cls]
        else:
            label = model.names[cls]  # 其他類別使用模型預設名稱
            
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 畫框與標籤 (為貓和狗使用不同顏色)
        if cls == 16:  # cat
            color = (255, 97, 0)  # 藍綠色 (BGR格式)
            found_cat = True
            cat_count += 1
        elif cls == 17:  # dog
            color = (0, 255, 255)  # 黃色 (BGR格式)
            found_dog = True
            dog_count += 1
        else:
            color = (0, 255, 0)  # 其他物件使用綠色
            
        # 畫出邊界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 畫出標籤背景
        text = f'{label} {conf:.2f}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img, (x1, y1-25), (x1+text_size[0], y1), color, -1)
        
        # 畫出標籤文字 (白色)
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# === 7. 印出偵測摘要 (英文) ===
if found_cat and found_dog:
    print(f"✅ Detected: {cat_count} cat(s) and {dog_count} dog(s)")
elif found_cat:
    print(f"✅ Detected: {cat_count} cat(s)")
elif found_dog:
    print(f"✅ Detected: {dog_count} dog(s)")
else:
    print("❌ No cats or dogs detected")

# === 8. 調整圖片大小並顯示 ===
resized_img = cv2.resize(img, (1000, 800))

# === 9. 顯示結果圖像 ===
cv2.imshow("YOLOv8 Cat & Dog Detection (1000x800)", resized_img)  # 英文標題
cv2.waitKey(0)
cv2.destroyAllWindows()