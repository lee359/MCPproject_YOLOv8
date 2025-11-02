import cv2
from ultralytics import YOLO
from collections import deque
import numpy as np

# ESP32-CAM 設定
ESP32_URL = 'http://192.168.0.102:81/stream'
cap = cv2.VideoCapture(ESP32_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 載入模型
model = YOLO('best.pt')

# 追踪系統初始化
tracked_objects = {}
object_id_counter = [0]

if not cap.isOpened():
    print("❌ 無法連接到 ESP32-CAM 串流")
    exit()

print("✅ 成功連線，開始辨識...")

frame_skip, frame_count = 5, 0

def get_center(bbox):
    """計算邊界框中心點"""
    return [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]

def calculate_distance(center1, center2):
    """計算兩點間距離"""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 畫面讀取失敗")
        break

    frame_count += 1
    
    # 跳幀處理
    if frame_count % frame_skip != 0:
        cv2.imshow("ESP32-CAM + YOLOv8", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # 清空緩衝區
    for _ in range(int(cap.get(cv2.CAP_PROP_BUFFERSIZE))):
        cap.grab()
    
    # 模型推論
    results = model.predict(source=frame, imgsz=416, conf=0.4, iou=0.45, verbose=False)
    
    # 提取檢測結果
    current_detections = [
        {
            'bbox': box.xyxy[0].cpu().numpy().tolist(),
            'conf': float(box.conf[0]),
            'cls': int(box.cls[0])
        }
        for box in results[0].boxes
    ]
    
    # 更新追踪物體年齡，刪除過期物體
    for obj_id in list(tracked_objects.keys()):
        tracked_objects[obj_id]['age'] += 1
        if tracked_objects[obj_id]['age'] > 3:
            del tracked_objects[obj_id]
    
    # 匹配與追踪
    for det in current_detections:
        curr_center = get_center(det['bbox'])
        best_match_id, best_similarity = None, 0
        
        # 尋找最佳匹配
        for obj_id, tracked in tracked_objects.items():
            if tracked['cls'] == det['cls'] and tracked['bbox']:
                distance = calculate_distance(curr_center, get_center(tracked['bbox'][-1]))
                if distance < 60:
                    similarity = 1 / (1 + distance / 50)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = obj_id
        
        # 更新或創建物體追踪
        if best_match_id and best_similarity > 0.5:
            tracked_objects[best_match_id]['bbox'].append(det['bbox'])
            tracked_objects[best_match_id]['conf'].append(det['conf'])
            tracked_objects[best_match_id]['age'] = 0
            # 保持最近5幀
            if len(tracked_objects[best_match_id]['bbox']) > 5:
                tracked_objects[best_match_id]['bbox'].pop(0)
                tracked_objects[best_match_id]['conf'].pop(0)
        elif det['conf'] > 0.5:
            # 新物體
            tracked_objects[object_id_counter[0]] = {
                'bbox': [det['bbox']],
                'conf': [det['conf']],
                'cls': det['cls'],
                'age': 0
            }
            object_id_counter[0] += 1
    
    # 生成穩定檢測（至少2幀且當前幀有匹配）
    stable_detections = [
        {
            'bbox': np.mean(tracked['bbox'], axis=0).tolist(),
            'conf': np.mean(tracked['conf']),
            'cls': tracked['cls']
        }
        for tracked in tracked_objects.values()
        if len(tracked['bbox']) >= 2 and tracked['age'] == 0
    ]
    
    # 繪製檢測結果
    annotated_frame = frame.copy()
    for det in stable_detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{model.names[det['cls']]} {det['conf']:.2f}"
        
        # 繪製邊界框
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 繪製標籤
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    cv2.imshow("ESP32-CAM + YOLOv8", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
