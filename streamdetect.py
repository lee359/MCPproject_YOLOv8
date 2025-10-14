import cv2
from ultralytics import YOLO

# ❗請替換為你的 ESP32-CAM 實際 IP
ESP32_URL = 'http://192.168.0.102:81/stream'

# 讀取 ESP32 串流
cap = cv2.VideoCapture(ESP32_URL)

# 載入模型：可以用 yolov8n.pt 或你訓練好的 best.pt
model = YOLO('best.pt')  # 或 'best.pt'

if not cap.isOpened():
    print("❌ 無法連接到 ESP32-CAM 串流")
    exit()

print("✅ 成功連線，開始辨識...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 畫面讀取失敗")
        break

    # 將畫面送入模型進行推論
    results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)

    # 將辨識結果繪製在原圖上
    annotated_frame = results[0].plot()

    # 顯示畫面
    cv2.imshow("ESP32-CAM + YOLOv8", annotated_frame)

    # 按下 q 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
