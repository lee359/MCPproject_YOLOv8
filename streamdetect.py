import cv2
from ultralytics import YOLO
import time
from collections import deque
import numpy as np

# ❗請替換為你的 ESP32-CAM 實際 IP
ESP32_URL = 'http://192.168.0.102:81/stream'

# 讀取 ESP32 串流
cap = cv2.VideoCapture(ESP32_URL)

# 設定緩衝區大小為 1，避免累積延遲
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 載入模型：可以用 yolov8n.pt 或你訓練好的 best.pt
model = YOLO('best.pt')  # 或 'best.pt'

# 用於平滑檢測結果，減少閃爍
detection_history = deque(maxlen=3)  # 保存最近3幀的檢測結果

if not cap.isOpened():
    print("❌ 無法連接到 ESP32-CAM 串流")
    exit()

print("✅ 成功連線，開始辨識...")

# 跳幀計數器 - 每 N 幀處理一次
frame_skip = 10  # 每 2 幀處理 1 幀，可調整為 3, 4 等
frame_count = 0

# FPS 控制（已禁用显示）
# prev_time = time.time()
# fps_display = 0

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
    for _ in range(int(cap.get(cv2.CAP_PROP_BUFFERSIZE))):
        cap.grab()
    
    # 將畫面送入模型進行推論
    results = model.predict(source=frame, imgsz=416, conf=confidence_threshold, iou=0.5, verbose=False)
    
    # 獲取當前幀的檢測結果
    current_detections = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # 提取邊界框座標、信心度和類別
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            current_detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': conf,
                'cls': cls
            })
    
    # 將當前檢測添加到歷史記錄
    detection_history.append(current_detections)
    
    # 平滑處理：只顯示在多幀中都出現的檢測
    stable_detections = []
    if len(detection_history) >= 2:  # 至少有2幀數據
        for det in current_detections:
            # 檢查這個檢測是否在前一幀也出現過（位置相近）
            is_stable = False
            for prev_frame_dets in list(detection_history)[:-1]:  # 檢查歷史幀
                for prev_det in prev_frame_dets:
                    if prev_det['cls'] == det['cls']:  # 同類別
                        # 計算邊界框中心點距離
                        curr_center = [(det['bbox'][0] + det['bbox'][2])/2, 
                                     (det['bbox'][1] + det['bbox'][3])/2]
                        prev_center = [(prev_det['bbox'][0] + prev_det['bbox'][2])/2,
                                     (prev_det['bbox'][1] + prev_det['bbox'][3])/2]
                        distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                         (curr_center[1] - prev_center[1])**2)
                        
                        # 如果中心點距離小於50像素，認為是同一物體
                        if distance < 50:
                            is_stable = True
                            break
                if is_stable:
                    break
            
            # 如果是穩定的檢測或信心度很高，則添加
            if is_stable or det['conf'] > 0.6:
                stable_detections.append(det)
    else:
        # 初始幀，使用高信心度的檢測
        stable_detections = [det for det in current_detections if det['conf'] > 0.5]
    
    # 手動繪製穩定的檢測結果
    annotated_frame = frame.copy()
    for det in stable_detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        conf = det['conf']
        cls = det['cls']
        
        # 獲取類別名稱
        class_name = model.names[cls]
        
        # 繪製邊界框（使用較粗的線條）
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 繪製標籤背景
        label = f'{class_name} {conf:.2f}'
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
        
        # 繪製標籤文字
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # 顯示畫面
    cv2.imshow("ESP32-CAM + YOLOv8", annotated_frame)

    # 按下 q 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
