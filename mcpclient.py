# server.py
from mcp.server.fastmcp import FastMCP, Image
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from io import BytesIO
import time
import requests
import os
from filterpy.kalman import KalmanFilter
import torch
import sys

# 獲取腳本所在目錄的絕對路徑
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")

# ESP32-CAM 串流 URL
DEFAULT_STREAM_URL = 'http://192.168.0.103:81/stream'  # 更新為 .103

# 加載訓練好的模型（使用絕對路徑）
model = YOLO(MODEL_PATH)

# 檢查是否有可用的 GPU 並將模型轉移
if torch.cuda.is_available():
    model.to('cuda')

# Create an MCP server
mcp = FastMCP("YOLOv8 Detection Server")

# Kalman Filter 輔助函數
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

@mcp.tool()
def detect_esp32_stream(
    stream_url: str = None,
    imgsz: int = 416, 
    conf: float = 0.25, 
    frame_skip: int = 1,
    max_age: int = 5,
    display_tolerance: int = 3,
    use_kalman: bool = True
) -> dict:
    """
    從預設的 ESP32-CAM 串流進行單幀 YOLO 物體偵測（支援 FPS 和閃爍控制）
    
    預設串流 URL: http://192.168.0.103:81/stream
    
    Args:
        stream_url: 串流 URL（可選，預設使用 http://192.168.0.103:81/stream）
        imgsz: 圖像大小，預設 416
        conf: 信心閾值，預設 0.25
        frame_skip: 跳幀數量（越大 FPS 越低，如 30 = 1 FPS），預設 1
        max_age: 物體最大存活幀數（預設 5）
        display_tolerance: 顯示容忍幀數（age <= 此值時顯示，預設 3）
        use_kalman: 是否使用 Kalman Filter 平滑追踪（預設 True）
    
    Returns:
        dict: 包含偵測結果和註釋圖像的字典
        
    FPS 調整建議:
        - frame_skip=1: 全速檢測（30 FPS 攝像頭 → 30 FPS）
        - frame_skip=5: 平衡模式（30 FPS → 6 FPS）
        - frame_skip=15: 節能模式（30 FPS → 2 FPS）
        - frame_skip=30: 低速模式（30 FPS → 1 FPS）✅
        - frame_skip=60: 超低速（30 FPS → 0.5 FPS）✅
        
    閃爍控制:
        - max_age: 控制物體保留時間（建議 5-20）
        - display_tolerance: 控制顯示容忍度（建議 3-7）
    """
    start_time = time.time()
    
    # 使用自定義 URL 或預設 URL
    target_url = stream_url if stream_url else DEFAULT_STREAM_URL
    
    # 追踪系統初始化
    tracked_objects = {}
    object_id_counter = [0]
    
    try:
        # 連接到串流
        cap = cv2.VideoCapture(target_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 設定連接超時
        connect_timeout = 5
        wait_start = time.time()
        
        while not cap.isOpened() and (time.time() - wait_start) < connect_timeout:
            time.sleep(0.1)
        
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"無法在 {connect_timeout} 秒內連接到串流: {target_url}",
                "elapsed_time": round(time.time() - start_time, 2)
            }
        
        # 多幀檢測循環（用於 Kalman Filter 追踪）
        frame_count = 0
        frames_to_process = max(frame_skip * 3, 5)  # 至少處理 5 幀或 3 個檢測週期
        
        for _ in range(frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 跳幀處理
            if frame_count % frame_skip != 0:
                continue
            
            # 清空緩衝區
            for _ in range(min(2, int(cap.get(cv2.CAP_PROP_BUFFERSIZE)))):
                cap.grab()
            
            # 執行 YOLO 偵測
            results = model.predict(source=frame, imgsz=imgsz, conf=conf, iou=0.15, verbose=False)
            
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
                if tracked_objects[obj_id]['age'] > max_age:
                    del tracked_objects[obj_id]
            
            if use_kalman:
                # Kalman Filter 追踪模式
                for det in current_detections:
                    curr_center = get_center(det['bbox'])
                    best_match_id, best_similarity = None, 0
                    
                    # 尋找最佳匹配
                    for obj_id, tracked in tracked_objects.items():
                        if tracked['cls'] == det['cls']:
                            predicted_center = tracked['kf'].x[:2].flatten()
                            distance = calculate_distance(curr_center, predicted_center)
                            
                            if distance < 80:
                                similarity = 1 / (1 + distance / 50)
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match_id = obj_id
                    
                    # 更新或創建物體追踪
                    if best_match_id and best_similarity > 0.4:
                        tracked_objects[best_match_id]['kf'].predict()
                        tracked_objects[best_match_id]['kf'].update(curr_center)
                        tracked_objects[best_match_id]['bbox'] = det['bbox']
                        tracked_objects[best_match_id]['conf'] = det['conf']
                        tracked_objects[best_match_id]['age'] = 0
                    elif det['conf'] > 0.4:
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
            else:
                # 簡單模式：直接使用當前檢測
                for det in current_detections:
                    tracked_objects[object_id_counter[0]] = {
                        'bbox': det['bbox'],
                        'conf': det['conf'],
                        'cls': det['cls'],
                        'age': 0
                    }
                    object_id_counter[0] += 1
        
        # 讀取最後一幀用於繪圖
        read_start = time.time()
        ret, frame = cap.read()
        read_time = time.time() - read_start
        
        if not ret:
            cap.release()
            return {
                "success": False,
                "error": "無法讀取畫面",
                "elapsed_time": round(time.time() - start_time, 2)
            }
        
        # 繪製穩定的追踪結果
        predict_start = time.time()
        detections = []
        annotated_frame = frame.copy()
        
        for obj_id, tracked in tracked_objects.items():
            if tracked['age'] <= display_tolerance:  # 使用可調整的顯示容忍度
                bbox = tracked['bbox']
                
                if use_kalman and 'kf' in tracked:
                    # 使用 Kalman 平滑的中心點
                    smoothed_center = tracked['kf'].x[:2].flatten()
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    x1 = int(smoothed_center[0] - w/2)
                    y1 = int(smoothed_center[1] - h/2)
                    x2 = int(smoothed_center[0] + w/2)
                    y2 = int(smoothed_center[1] + h/2)
                else:
                    # 使用原始邊界框
                    x1, y1, x2, y2 = map(int, bbox)
                
                class_name = model.names[tracked['cls']]
                confidence = tracked['conf']
                
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "age": tracked['age']
                })
                
                # 繪製邊界框
                label = f"{class_name} {confidence:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 繪製標籤
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        predict_time = time.time() - predict_start
        
        # 將圖像轉換為 base64
        encode_start = time.time()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        encode_time = time.time() - encode_start
        
        cap.release()
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "tracked_objects_count": len(tracked_objects),
            "annotated_image_base64": img_base64,
            "frame_size": {"height": frame.shape[0], "width": frame.shape[1]},
            "stream_url": target_url,
            "parameters": {
                "imgsz": imgsz,
                "conf": conf,
                "frame_skip": frame_skip,
                "max_age": max_age,
                "display_tolerance": display_tolerance,
                "use_kalman": use_kalman
            },
            "performance": {
                "total_time": round(total_time, 2),
                "read_time": round(read_time, 2),
                "predict_time": round(predict_time, 2),
                "encode_time": round(encode_time, 2),
                "frames_processed": frame_count
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "elapsed_time": round(time.time() - start_time, 2)
        }

@mcp.tool()
def detect_stream_frame_simple(stream_url: str, imgsz: int = 416, conf: float = 0.3) -> dict:
    """
    從串流 URL 捕獲一幀並進行 YOLO 物體偵測（簡化版，不返回圖像）
    
    Args:
        stream_url: 串流 URL (例如: http://192.168.0.103:81/stream)
        imgsz: 圖像大小，預設 416
        conf: 信心閾值，預設 0.3
    
    Returns:
        dict: 只包含偵測結果（不含圖像，回應更快）
    """
    start_time = time.time()
    
    try:
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"無法連接到串流: {stream_url}"
            }
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {
                "success": False,
                "error": "無法讀取畫面"
            }
        
        # 執行 YOLO 偵測
        results = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)
        
        # 取得偵測結果
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                detection = {
                    "class": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                }
                detections.append(detection)
        
        cap.release()
        
        return {
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "frame_size": {"height": frame.shape[0], "width": frame.shape[1]},
            "elapsed_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def check_stream_health(stream_url: str) -> dict:
    """
    快速檢查串流健康狀態（用於診斷連接問題）
    
    Args:
        stream_url: 串流 URL
    
    Returns:
        dict: 串流健康狀態和診斷資訊
    """
    result = {
        "url": stream_url,
        "timestamp": time.time()
    }
    
    try:
        # HTTP 測試
        start = time.time()
        response = requests.get(stream_url, timeout=3, stream=True)
        http_time = time.time() - start
        
        result["http_status"] = response.status_code
        result["http_time"] = round(http_time, 3)
        result["content_type"] = response.headers.get('Content-Type', 'Unknown')
        
        # 讀取少量數據測試
        chunk = next(response.iter_content(100), None)
        result["can_receive_data"] = chunk is not None
        if chunk:
            result["data_size"] = len(chunk)
        
        response.close()
        
        # OpenCV 測試
        start = time.time()
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        opencv_connect_time = time.time() - start
        
        result["opencv_opened"] = cap.isOpened()
        result["opencv_connect_time"] = round(opencv_connect_time, 3)
        
        if cap.isOpened():
            start = time.time()
            ret, frame = cap.read()
            read_time = time.time() - start
            
            result["can_read_frame"] = ret
            result["frame_read_time"] = round(read_time, 3)
            
            if ret:
                result["frame_size"] = {
                    "height": frame.shape[0],
                    "width": frame.shape[1],
                    "channels": frame.shape[2]
                }
        
        cap.release()
        result["success"] = True
        result["overall_status"] = "健康" if result.get("can_read_frame") else "異常"
        
    except requests.exceptions.Timeout:
        result["success"] = False
        result["error"] = "HTTP 請求超時"
        result["overall_status"] = "超時"
    except requests.exceptions.ConnectionError as e:
        result["success"] = False
        result["error"] = f"連接錯誤: {str(e)}"
        result["overall_status"] = "連接失敗"
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
        result["overall_status"] = "錯誤"
    
    return result

@mcp.tool()
def detect_image(image_path: str, imgsz: int = 640, conf: float = 0.3) -> dict:
    """
    對單張圖片進行 YOLO 物體偵測
    
    Args:
        image_path: 圖片路徑
        imgsz: 圖像大小，預設 640
        conf: 信心閾值，預設 0.3
    
    Returns:
        dict: 包含偵測結果的字典
    """
    try:
        # 執行 YOLO 偵測
        results = model.predict(source=image_path, imgsz=imgsz, conf=conf, verbose=False)
        
        # 取得偵測結果
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                detection = {
                    "class": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        # 將辨識結果繪製在原圖上
        annotated_frame = results[0].plot()
        
        # 將圖像轉換為 base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "image_path": image_path,
            "detections": detections,
            "detection_count": len(detections),
            "annotated_image_base64": img_base64,
            "parameters": {
                "imgsz": imgsz,
                "conf": conf
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# 添加這行代碼以確保服務器可以運行
if __name__ == "__main__":
   mcp.run()