import cv2
from ultralytics import YOLO
import numpy as np
from filterpy.kalman import KalmanFilter
import torch

# ESP32-CAM 設定
ESP32_URL = 'http://192.168.0.102:81/stream'
cap = cv2.VideoCapture(ESP32_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 載入模型
model = YOLO('best.pt')

# 檢查是否有可用的 GPU 並將模型轉移
if torch.cuda.is_available():
    print("偵測到 CUDA，正在將模型轉移到 GPU...")
    model.to('cuda')
    print("模型已成功轉移到 GPU。")
else:
    print("未偵測到 CUDA，模型將在 CPU 上運行。")

# 追踪系統：存儲每個物體的 Kalman Filter
tracked_objects = {}
object_id_counter = [0]

def create_kalman_filter():
    """創建 Kalman Filter (追踪中心點 x, y)"""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    # 狀態轉移矩陣 [x, y, vx, vy]
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    # 測量矩陣
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    # 測量噪聲
    kf.R *= 10
    # 過程噪聲
    kf.Q *= 0.1
    # 初始協方差
    kf.P *= 1000
    return kf

def get_center(bbox):
    """計算邊界框中心點"""
    return [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]

def calculate_distance(center1, center2):
    """計算兩點間距離"""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

if not cap.isOpened():
    print("❌ 無法連接到 ESP32-CAM 串流")
    exit()

print("✅ 成功連線，開始辨識...")

# 設定視窗大小（可調整窗口顯示）
cv2.namedWindow("ESP32-CAM + YOLOv8", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ESP32-CAM + YOLOv8", 640, 480)  # 設定視窗大小為 640x480

frame_skip, frame_count = 1, 0  # frame_skip: 調整檢測頻率 (越大越慢，如 30 = 1 FPS)
max_age = 5  # 物體消失前的最大幀數 (增加此值可減少閃爍)

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
    
    # 模型推論（調整參數以提高準確度）
    # imgsz 改為 640（YOLOv8 標準尺寸，提高準確度）
    # conf 維持 0.25（信心度閾值）
    # iou 改為 0.45（標準 NMS 閾值，減少重疊框）
    # half=False 強制使用全精度（FP32），避免 GPU 半精度導致的準確度下降
    results = model.predict(source=frame, imgsz=640, conf=0.60, iou=0.3, verbose=False, half=False)
    
    # 提取檢測結果
    current_detections = [
        {
            'bbox': box.xyxy[0].cpu().numpy().tolist(),
            'conf': float(box.conf[0]),
            'cls': int(box.cls[0])
        }
        for box in results[0].boxes
    ]
    
    # 更新所有追踪物體的年齡
    for obj_id in list(tracked_objects.keys()):
        tracked_objects[obj_id]['age'] += 1
        if tracked_objects[obj_id]['age'] > max_age:  # 使用可調整的 max_age
            del tracked_objects[obj_id]
    
    # 匹配當前檢測與已追踪物體
    for det in current_detections:
        curr_center = get_center(det['bbox'])
        best_match_id, best_similarity = None, 0
        
        # 尋找最佳匹配（使用預測位置）
        for obj_id, tracked in tracked_objects.items():
            if tracked['cls'] == det['cls']:
                # 使用 Kalman Filter 預測的位置
                predicted_center = tracked['kf'].x[:2].flatten()
                distance = calculate_distance(curr_center, predicted_center)
                
                if distance < 80:  # 匹配閾值
                    similarity = 1 / (1 + distance / 50)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = obj_id
        
        # 更新或創建物體追踪
        if best_match_id and best_similarity > 0.4:
            # 更新 Kalman Filter
            tracked_objects[best_match_id]['kf'].predict()
            tracked_objects[best_match_id]['kf'].update(curr_center)
            
            # 保存原始邊界框和信心度
            tracked_objects[best_match_id]['bbox'] = det['bbox']
            tracked_objects[best_match_id]['conf'] = det['conf']
            tracked_objects[best_match_id]['age'] = 0
        elif det['conf'] > 0.4:
            # 創建新物體追踪
            kf = create_kalman_filter()
            kf.x[:2] = np.array(curr_center).reshape(2, 1)
            
            tracked_objects[object_id_counter[0]] = {
                'kf': kf,
                'bbox': det['bbox'],
                'conf': det['conf'],
                'cls': det['cls'],
                'age': 0
            }
            object_id_counter[0] += 1
    
    # 繪製穩定的追踪結果（顯示最近檢測到的物體，包括暫時未匹配的）
    annotated_frame = frame.copy()
    for obj_id, tracked in tracked_objects.items():
        if tracked['age'] <= 3:  # 顯示最近 3 幀內有活動的物體（減少閃爍）
            # 使用 Kalman 平滑後的中心點重建邊界框
            smoothed_center = tracked['kf'].x[:2].flatten()
            bbox = tracked['bbox']
            
            # 計算邊界框寬高
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            # 使用平滑的中心點重建邊界框
            x1 = int(smoothed_center[0] - w/2)
            y1 = int(smoothed_center[1] - h/2)
            x2 = int(smoothed_center[0] + w/2)
            y2 = int(smoothed_center[1] + h/2)
            
            label = f"{model.names[tracked['cls']]} {tracked['conf']:.2f}"
            
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
