import cv2
from ultralytics import YOLO
import time

# ❗請替換為你的 ESP32-CAM 實際 IP
ESP32_URL = 'http://192.168.0.102:81/stream'

# 讀取 ESP32 串流
cap = cv2.VideoCapture(ESP32_URL)

# 設定緩衝區大小為 1，避免累積延遲
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 載入模型：可以用 yolov8n.pt 或你訓練好的 best.pt
model = YOLO('best.pt')  # 或 'best.pt'

if not cap.isOpened():
    print("❌ 無法連接到 ESP32-CAM 串流")
    exit()

print("✅ 成功連線，開始辨識...")

# 跳幀計數器 - 每 N 幀處理一次
frame_skip = 2  # 每 2 幀處理 1 幀，可調整為 3, 4 等
frame_count = 0

# FPS 控制
prev_time = time.time()
fps_display = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 畫面讀取失敗")
        break

    frame_count += 1
    
    # 跳幀處理：只處理特定幀
    if frame_count % frame_skip != 0:
        # 顯示原始畫面（不辨識）
        cv2.imshow("ESP32-CAM + YOLOv8", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # 清空緩衝區，確保取得最新畫面
    # 讀取並丟棄多餘的幀
    for _ in range(cap.get(cv2.CAP_PROP_BUFFERSIZE)):
        cap.grab()
    
    # 將畫面送入模型進行推論
    results = model.predict(source=frame, imgsz=416, conf=0.3, verbose=False)  # 降低解析度從 640 到 416

    # 將辨識結果繪製在原圖上
    annotated_frame = results[0].plot()
    
    # 計算並顯示 FPS
    curr_time = time.time()
    fps_display = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # 在畫面上顯示 FPS
    cv2.putText(annotated_frame, f"FPS: {fps_display:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 顯示畫面
    cv2.imshow("ESP32-CAM + YOLOv8", annotated_frame)

    # 按下 q 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
