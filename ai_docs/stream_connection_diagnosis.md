# ESP32-CAM 串流連接問題診斷報告

**問題**: MCP Client 調用 `detect_stream_frame` 時返回 `Failed to fetch`  
**測試 URL**: `http://192.168.0.103:81/stream`  
**日期**: 2025-10-15

---

## 診斷結果總結

### ✅ 正常的部分

1. **網路連通性**: ✅ 正常
   - Ping 192.168.0.103 成功
   - 平均延遲: 2-3ms

2. **ESP32-CAM 串流**: ✅ 正常
   - HTTP 狀態碼: 200
   - Content-Type: `multipart/x-mixed-replace`
   - 可以正常接收數據

3. **OpenCV 串流讀取**: ✅ 正常
   - `cap.isOpened()`: True
   - 可以成功讀取畫面 (240x320)
   - 連續讀取無問題

4. **MCP 函數本身**: ✅ 正常
   - 直接調用 `detect_stream_frame()` 成功
   - 返回正確的結果結構
   - Base64 圖像編碼正常

### ❌ 問題所在

**問題不在服務器端程式碼，而在 MCP Client 端！**

---

## 可能的原因

### 1. MCP Client 超時設定過短

MCP client（如 Claude Desktop）可能有預設的請求超時時間，而串流處理需要較長時間：

- 連接串流: ~1-2 秒
- 讀取畫面: ~0.5-1 秒  
- YOLO 推論: ~2-5 秒（取決於硬體）
- 圖像編碼: ~0.5-1 秒

**總計**: 4-10 秒，可能超過 MCP client 的超時限制

### 2. MCP Client 網路限制

某些 MCP client 可能限制工具訪問外部網路資源（安全考量）

### 3. 同步調用阻塞

MCP 協議可能在等待回應時阻塞，導致超時

---

## 解決方案

### 方案 1: 增加超時處理和進度回報（推薦）

修改 `detect_stream_frame` 函數，增加超時控制和快速回應：

```python
@mcp.tool()
def detect_stream_frame(stream_url: str, imgsz: int = 416, conf: float = 0.3, timeout: int = 10) -> dict:
    """
    從串流 URL 捕獲一幀並進行 YOLO 物體偵測
    
    Args:
        stream_url: 串流 URL (例如: http://192.168.0.102:81/stream)
        imgsz: 圖像大小，預設 416
        conf: 信心閾值，預設 0.3
        timeout: 連接超時時間（秒），預設 10
    
    Returns:
        dict: 包含偵測結果和註釋圖像的字典
    """
    import time
    
    start_time = time.time()
    
    try:
        # 連接到串流（增加超時控制）
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 設定連接超時
        connect_timeout = 5  # 5 秒連接超時
        wait_start = time.time()
        
        while not cap.isOpened() and (time.time() - wait_start) < connect_timeout:
            time.sleep(0.1)
        
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"無法在 {connect_timeout} 秒內連接到串流: {stream_url}",
                "elapsed_time": time.time() - start_time
            }
        
        # 讀取一幀（帶超時）
        read_timeout = 3  # 3 秒讀取超時
        read_start = time.time()
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            return {
                "success": False,
                "error": "無法讀取畫面",
                "elapsed_time": time.time() - start_time
            }
        
        if (time.time() - read_start) > read_timeout:
            cap.release()
            return {
                "success": False,
                "error": f"讀取畫面超時（>{read_timeout}秒）",
                "elapsed_time": time.time() - start_time
            }
        
        # 執行 YOLO 偵測
        predict_start = time.time()
        results = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)
        predict_time = time.time() - predict_start
        
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
        
        # 將辨識結果繪製在原圖上
        annotated_frame = results[0].plot()
        
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
            "annotated_image_base64": img_base64,
            "parameters": {
                "imgsz": imgsz,
                "conf": conf
            },
            "performance": {
                "total_time": round(total_time, 2),
                "predict_time": round(predict_time, 2),
                "encode_time": round(encode_time, 2)
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }
```

### 方案 2: 簡化版本（不返回圖像）

如果 base64 圖像導致回應過大，可以提供一個輕量版本：

```python
@mcp.tool()
def detect_stream_frame_simple(stream_url: str, imgsz: int = 416, conf: float = 0.3) -> dict:
    """
    從串流 URL 捕獲一幀並進行 YOLO 物體偵測（簡化版，不返回圖像）
    
    Args:
        stream_url: 串流 URL
        imgsz: 圖像大小，預設 416
        conf: 信心閾值，預設 0.3
    
    Returns:
        dict: 只包含偵測結果（不含圖像）
    """
    try:
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            return {"success": False, "error": f"無法連接到串流: {stream_url}"}
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"success": False, "error": "無法讀取畫面"}
        
        results = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)
        
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
            "frame_size": frame.shape[:2]  # (height, width)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 方案 3: 增加健康檢查工具

添加一個快速檢查工具，用於測試連接：

```python
@mcp.tool()
def check_stream_health(stream_url: str) -> dict:
    """
    快速檢查串流健康狀態
    
    Args:
        stream_url: 串流 URL
    
    Returns:
        dict: 串流健康狀態
    """
    import time
    
    result = {
        "url": stream_url,
        "timestamp": time.time()
    }
    
    try:
        # HTTP 測試
        import requests
        start = time.time()
        response = requests.get(stream_url, timeout=3, stream=True)
        http_time = time.time() - start
        
        result["http_status"] = response.status_code
        result["http_time"] = round(http_time, 2)
        result["content_type"] = response.headers.get('Content-Type', 'Unknown')
        response.close()
        
        # OpenCV 測試
        start = time.time()
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        opencv_connect_time = time.time() - start
        
        result["opencv_opened"] = cap.isOpened()
        result["opencv_connect_time"] = round(opencv_connect_time, 2)
        
        if cap.isOpened():
            ret, frame = cap.read()
            result["can_read_frame"] = ret
            if ret:
                result["frame_size"] = frame.shape
        
        cap.release()
        result["success"] = True
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
    
    return result
```

---

## 建議的修改步驟

### 步驟 1: 更新 MCPclient.py

將上述三個函數都加入到 `MCPclient.py` 中：
- `detect_stream_frame` - 改進版（增加超時和性能監控）
- `detect_stream_frame_simple` - 簡化版（不返回圖像）
- `check_stream_health` - 健康檢查工具

### 步驟 2: 測試新工具

先使用 `check_stream_health` 確認串流可用，再使用 `detect_stream_frame_simple` 測試偵測功能。

### 步驟 3: 檢查 MCP Client 設定

如果問題仍然存在，需要檢查 MCP client（如 Claude Desktop）的配置：
- 超時設定
- 網路權限
- 日誌輸出

---

## 其他可能的解決方案

### 使用本地圖片快照

如果 MCP client 無法處理即時串流，可以改為：
1. 定期從串流捕獲畫面並保存為圖片
2. 使用 `detect_image` 工具處理保存的圖片

```python
# 輔助腳本：定期保存串流快照
import cv2
import time

cap = cv2.VideoCapture("http://192.168.0.103:81/stream")
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("latest_frame.jpg", frame)
    time.sleep(1)  # 每秒更新
```

然後使用 MCP tool:
```json
{
    "tool": "detect_image",
    "arguments": {
        "image_path": "latest_frame.jpg"
    }
}
```

---

## 結論

**根本原因**: MCP client 端的限制（超時、網路權限等），而非服務器端程式碼問題。

**推薦方案**: 
1. 先使用 `check_stream_health` 工具測試連接
2. 使用改進版的 `detect_stream_frame`（帶超時控制）
3. 如果仍有問題，使用 `detect_stream_frame_simple`（輕量版）

**下一步**: 需要更新 `MCPclient.py` 實現上述改進。
