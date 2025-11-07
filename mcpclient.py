# server.py
from mcp.server.fastmcp import FastMCP, Image
from datetime import datetime
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
import csv
import json

# 重新加入 url_is_accessible 輔助函數
def url_is_accessible(url, timeout=5):
    """檢查 URL 是否可訪問（使用 GET 方法以支援 ESP32-CAM）"""
    try:
        # ESP32-CAM 不支援 HEAD 請求，改用 GET 但只讀取少量數據
        response = requests.get(url, timeout=timeout, stream=True)
        # 接受 200 (OK) 和 405 (Method Not Allowed) 都視為可訪問
        # 因為某些 ESP32-CAM 對 stream endpoint 會回傳 405 給 HEAD 請求
        accessible = response.status_code in [200, 405]
        response.close()
        return accessible
    except requests.RequestException:
        return False

# 獲取腳本所在目錄的絕對路徑
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")
CSV_LOG_PATH = os.path.join(SCRIPT_DIR, "detection_logs.csv")
CSV_LOG_MULTI_PATH = os.path.join(SCRIPT_DIR, "detection_logs_multi.csv")

# ESP32-CAM 串流 URL
DEFAULT_STREAM_URL = 'http://192.168.0.102:81/stream'

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

def log_detection_to_csv(result_data):
    """
    將偵測結果記錄到 CSV 檔案
    
    Args:
        result_data: detect_stream_frame_simple 的返回值
    """
    # CSV 欄位名稱（移除 success）
    fieldnames = ['timestamp', 'detection_count', 'class', 'confidence', 'bbox', 'total_time', 'yolo_inference_time']
    
    # 檢查檔案是否存在
    file_exists = os.path.isfile(CSV_LOG_PATH)
    
    try:
        with open(CSV_LOG_PATH, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 如果檔案不存在，寫入標題列
            if not file_exists:
                writer.writeheader()
            
            # 準備要寫入的資料（移除 success）
            row_data = {
                'timestamp': result_data.get('timestamp', ''),
                'detection_count': result_data.get('detection_count', 0),
                'class': result_data.get('class', ''),
                'confidence': result_data.get('confidence', 0),
                'bbox': result_data.get('bbox', ''),
                'total_time': result_data.get('total_time', 0),
                'yolo_inference_time': result_data.get('yolo_inference_time', 0)
            }
            
            writer.writerow(row_data)
            
    except Exception as e:
        print(f"❌ CSV 記錄失敗: {e}", file=sys.stderr)

def log_multi_detection_to_csv(result_data):
    """
    將多物體偵測結果記錄到 CSV 檔案（每個物體一行）
    
    Args:
        result_data: detect_stream_frame_multi 的返回值
    """
    # CSV 欄位名稱
    fieldnames = ['timestamp', 'detection_count', 'class', 'confidence', 'bbox', 'total_time', 'yolo_inference_time']
    
    # 檢查檔案是否存在
    file_exists = os.path.isfile(CSV_LOG_MULTI_PATH)
    
    try:
        with open(CSV_LOG_MULTI_PATH, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 如果檔案不存在，寫入標題列
            if not file_exists:
                writer.writeheader()
            
            # 獲取所有偵測到的物體
            all_detections = result_data.get('all_detections', [])
            
            if all_detections:
                # 有檢測到物體：為每個物體寫入一行
                for det in all_detections:
                    row_data = {
                        'timestamp': result_data.get('timestamp', ''),
                        'detection_count': result_data.get('detection_count', 0),
                        'class': det.get('class', ''),
                        'confidence': det.get('confidence', 0),
                        'bbox': str(det.get('bbox', [])),
                        'total_time': result_data.get('total_time', 0),
                        'yolo_inference_time': result_data.get('yolo_inference_time', 0)
                    }
                    writer.writerow(row_data)
            else:
                # 沒有檢測到物體：寫入一行空記錄
                row_data = {
                    'timestamp': result_data.get('timestamp', ''),
                    'detection_count': 0,
                    'class': '',
                    'confidence': 0,
                    'bbox': '',
                    'total_time': result_data.get('total_time', 0),
                    'yolo_inference_time': result_data.get('yolo_inference_time', 0)
                }
                writer.writerow(row_data)
            
    except Exception as e:
        print(f"❌ CSV 記錄失敗: {e}", file=sys.stderr)

@mcp.tool()
def detect_esp32_stream(
        url: str = None,
        imgsz: int = 640,
        conf: float = 0.5,
        iou: float = 0.3,
        frame_skip: int = 3,
        max_age: int = 10,
        display_tolerance: int = 2,
        use_kalman: bool = True
) -> dict:
    """
    從 ESP32-CAM 的串流中偵測物體，並返回帶有標註的圖像和偵測結果。
    """
    start_time = time.time()
    target_url = url if url else DEFAULT_STREAM_URL

    # --- 步驟 1: 初始化所有變數，確保它們在任何路徑下都有定義 ---
    yolo_time, read_time, predict_time, encode_time = 0, 0, 0, 0
    total_frames = 0
    success = False
    error_message = ""
    detections, tracked_objects = [], {}
    img_base64 = ""
    annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap = None
    object_id_counter = [0]  # <--- 修正：將遺失的 object_id_counter 加回來

    try:
        # --- 步驟 2: 執行主要偵測邏輯 ---
        if not url_is_accessible(target_url):
            raise ConnectionError(f"無法訪問串流 URL: {target_url}")
            
        cap = cv2.VideoCapture(target_url)
        if not cap.isOpened():
            raise IOError(f"無法打開串流: {target_url}")

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
            yolo_start = time.time()
            results = model.predict(source=frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
            yolo_time += time.time() - yolo_start
            
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
            # 如果讀不到最後一幀，也視為一個可處理的狀況，而非讓程式崩潰
            annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(annotated_frame, "Stream ended unexpectedly", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            annotated_frame = frame.copy()

        # 繪製穩定的追踪結果
        predict_start = time.time()
        
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
        
        success = True

    except Exception as e:
        # --- 步驟 3: 處理所有可能的錯誤 ---
        error_message = str(e)
        success = False
        # 在黑色畫面上顯示錯誤訊息
        cv2.putText(annotated_frame, f"Error: {error_message[:50]}", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

    finally:
        # --- 步驟 4: 無論成功或失敗，都執行清理和記錄 ---
        if cap is not None and cap.isOpened():
            cap.release()
        
        total_time = time.time() - start_time

        params = {
            "target_url": target_url, "imgsz": imgsz, "conf": conf, 
            "frame_skip": frame_skip, "max_age": max_age, 
            "display_tolerance": display_tolerance, "use_kalman": use_kalman
        }
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "detection_count": len(detections),
            "detections": detections,
            "total_time": round(total_time, 3),
            "yolo_inference_time": round(yolo_time, 3)
                        
        }
        
        return result


@mcp.tool()
def detect_stream_frame_simple(stream_url: str=None, imgsz: int = 640, conf: float = 0.5, iou: float = 0.3) -> dict:
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
    yolo_time = 0  # 初始化 yolo_time
    target_url = stream_url if stream_url else DEFAULT_STREAM_URL
    
    try:
        cap = cv2.VideoCapture(target_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"無法連接到串流: {target_url}"
            }
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {
                "success": False,
                "error": "無法讀取畫面"
            }
        
        # 執行 YOLO 偵測
        yolo_start = time.time()
        results = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)
        yolo_time = time.time() - yolo_start

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
        
        total_time = time.time() - start_time
        
        # 提取第一个检测物体的信息（如果有）
        first_class = detections[0]['class'] if detections else ''
        first_confidence = round(detections[0]['confidence'], 3) if detections else 0
        # 将 bbox 中的每个坐标都限制到小数点后第三位
        first_bbox = str([round(coord, 3) for coord in detections[0]['bbox']]) if detections else ''
        
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_count": len(detections),
            "class": first_class,
            "confidence": first_confidence,
            "bbox": first_bbox,
            "total_time": round(total_time, 3),
            "yolo_inference_time": round(yolo_time, 3)
        }
        
        # 記錄到 CSV
        log_detection_to_csv(result)
        
        return result
        
    except Exception as e:
        error_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_count": 0,
            "class": '',
            "confidence": 0,
            "bbox": '',
            "total_time": round(time.time() - start_time, 3),
            "yolo_inference_time": 0,
            "error": str(e),
            "error_type": type(e).__name__
        }
        
        # 記錄錯誤到 CSV
        log_detection_to_csv(error_result)
        
        return error_result

@mcp.tool()
def detect_stream_frame_multi(stream_url: str=None, imgsz: int = 640, conf: float = 0.5, iou: float = 0.3) -> dict:
    """
    從串流 URL 捕獲一幀並進行 YOLO 物體偵測（多物體版本，返回所有檢測物體）
    
    適用場景：
    - 同時存在多個物體（例如：貓+狗）
    - 需要記錄每個檢測物體的完整資訊
    - 實驗場景 C 和 D（多物體偵測）
    
    Args:
        stream_url: 串流 URL (例如: http://192.168.0.103:81/stream)
        imgsz: 圖像大小，預設 640
        conf: 信心閾值，預設 0.5
        iou: IOU 閾值，預設 0.3
    
    Returns:
        dict: 包含所有偵測物體的詳細資訊
            - timestamp: 偵測時間戳記（格式：YYYY-MM-DD HH:MM:SS）
            - detection_count: 檢測到的物體數量
            - all_detections: 所有物體的列表，每個物體包含 class, confidence, bbox
            - total_time: 總處理時間（秒）
            - yolo_inference_time: YOLO 推理時間（秒）
    """
    start_time = time.time()
    target_url = stream_url if stream_url else DEFAULT_STREAM_URL
    yolo_time = 0
    
    try:
        cap = cv2.VideoCapture(target_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "detection_count": 0,
                "all_detections": [],
                "error": f"無法連接到串流: {target_url}",
                "total_time": 0,
                "yolo_inference_time": 0
            }
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "detection_count": 0,
                "all_detections": [],
                "error": "無法讀取畫面",
                "total_time": 0,
                "yolo_inference_time": 0
            }
        
        # 執行 YOLO 偵測
        yolo_start = time.time()
        results = model.predict(source=frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        yolo_time = time.time() - yolo_start

        # 取得所有偵測結果
        all_detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                detection = {
                    "class": r.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": [round(coord, 3) for coord in box.xyxy[0].tolist()]
                }
                all_detections.append(detection)
        
        cap.release()
        
        total_time = time.time() - start_time
        
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_count": len(all_detections),
            "all_detections": all_detections,
            "total_time": round(total_time, 3),
            "yolo_inference_time": round(yolo_time, 3)
        }
        
        # 記錄到 CSV（每個物體一行）
        log_multi_detection_to_csv(result)
        
        return result
        
    except Exception as e:
        error_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_count": 0,
            "all_detections": [],
            "total_time": round(time.time() - start_time, 3),
            "yolo_inference_time": 0,
            "error": str(e),
            "error_type": type(e).__name__
        }
        
        # 記錄錯誤到 CSV
        log_multi_detection_to_csv(error_result)
        
        return error_result

@mcp.tool()
def check_stream_health(url: str = DEFAULT_STREAM_URL, timeout: int = 5) -> dict:
    """
    檢查 ESP32-CAM 串流是否可訪問（透過 HTTP 和 OpenCV 測試連接性）
    
    Args:
        url: 串流 URL
        timeout: 超時時間（秒），預設 5 秒
    
    Returns:
        dict: 健康檢查結果
    """
    result = {
        "url": url,
        "timestamp": time.time()
    }
    
    try:
        # HTTP 測试
        start = time.time()
        response = requests.get(url, timeout=timeout, stream=True)
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
        cap = cv2.VideoCapture(url)
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