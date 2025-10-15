# MCP 串流偵測工具使用指南

## 問題解決總結

**原問題**: 使用 `detect_stream_frame` 時返回 `Failed to fetch`

**根本原因**: MCP client 端可能有超時限制或網路限制

**解決方案**: 提供了三個不同版本的工具，適用於不同場景

---

## 可用的 MCP Tools

### 1. `check_stream_health` - 串流健康檢查 ✅

**用途**: 快速診斷串流連接問題

**參數**:
```json
{
    "stream_url": "http://192.168.0.103:81/stream"
}
```

**返回範例**:
```json
{
  "url": "http://192.168.0.103:81/stream",
  "timestamp": 1760534813.413553,
  "http_status": 200,
  "http_time": 0.037,
  "content_type": "multipart/x-mixed-replace;boundary=...",
  "can_receive_data": true,
  "opencv_opened": true,
  "can_read_frame": true,
  "frame_read_time": 0.001,
  "frame_size": {
    "height": 240,
    "width": 320,
    "channels": 3
  },
  "success": true,
  "overall_status": "健康"
}
```

**何時使用**:
- ✅ 第一次連接新的串流源
- ✅ 診斷連接問題
- ✅ 監控串流健康狀態
- ✅ 測試網路延遲

**優點**:
- 🚀 極快（< 0.1 秒）
- 📊 詳細的診斷資訊
- 🔍 不需要 YOLO 推論

---

### 2. `detect_stream_frame_simple` - 簡化版偵測 ⚡

**用途**: 快速偵測，只返回偵測結果（不含圖像）

**參數**:
```json
{
    "stream_url": "http://192.168.0.103:81/stream",
    "imgsz": 416,
    "conf": 0.3
}
```

**返回範例**:
```json
{
  "success": true,
  "detections": [
    {
      "class": "person",
      "confidence": 0.85,
      "bbox": [120.5, 200.3, 350.7, 480.9]
    }
  ],
  "detection_count": 1,
  "frame_size": {
    "height": 240,
    "width": 320
  },
  "elapsed_time": 0.16
}
```

**何時使用**:
- ✅ 需要快速回應
- ✅ 只需要偵測結果（類別、位置、信心度）
- ✅ 節省網路頻寬
- ✅ MCP client 超時限制嚴格時

**優點**:
- 🚀 回應快速（~0.15-0.3 秒）
- 📦 回應體積小
- 💡 適合高頻率調用

**限制**:
- ❌ 不返回註釋圖像

---

### 3. `detect_stream_frame` - 完整版偵測 🎨

**用途**: 完整偵測，返回註釋圖像（Base64 編碼）

**參數**:
```json
{
    "stream_url": "http://192.168.0.103:81/stream",
    "imgsz": 416,
    "conf": 0.3,
    "timeout": 10
}
```

**返回範例**:
```json
{
  "success": true,
  "detections": [
    {
      "class": "person",
      "confidence": 0.85,
      "bbox": [120.5, 200.3, 350.7, 480.9]
    }
  ],
  "detection_count": 1,
  "annotated_image_base64": "iVBORw0KGgoAAAANS...",
  "frame_size": {
    "height": 240,
    "width": 320
  },
  "parameters": {
    "imgsz": 416,
    "conf": 0.3
  },
  "performance": {
    "total_time": 0.09,
    "read_time": 0.0,
    "predict_time": 0.03,
    "encode_time": 0.0
  }
}
```

**何時使用**:
- ✅ 需要視覺化結果
- ✅ 保存偵測記錄
- ✅ 生成報告
- ✅ 展示給用戶

**優點**:
- 🎨 包含完整的視覺化圖像
- 📊 詳細的性能指標
- ⏱️ 超時控制
- 🔍 錯誤診斷資訊

**限制**:
- ⏳ 回應時間較長（~0.1-0.5 秒）
- 📦 回應體積大（~5-20 KB）

---

## 使用建議流程

### 場景 1: 第一次使用新串流

```
1. check_stream_health
   ↓ (確認健康)
2. detect_stream_frame_simple
   ↓ (確認偵測功能)
3. detect_stream_frame
   (獲取完整結果)
```

### 場景 2: 高頻監控（每秒多次）

```
使用 detect_stream_frame_simple
- 快速回應
- 低頻寬消耗
```

### 場景 3: 需要視覺化

```
使用 detect_stream_frame
- 獲取註釋圖像
- 保存或展示
```

### 場景 4: 遇到 "Failed to fetch"

```
1. check_stream_health
   - 檢查 overall_status
   - 查看錯誤訊息
   
2. 如果 overall_status = "健康"
   → 問題在 MCP client
   → 使用 detect_stream_frame_simple
   
3. 如果 overall_status ≠ "健康"
   → 問題在串流源
   → 檢查 ESP32-CAM
```

---

## 錯誤處理

### 所有工具統一返回格式

**成功時**:
```json
{
  "success": true,
  ...
}
```

**失敗時**:
```json
{
  "success": false,
  "error": "錯誤描述",
  "error_type": "錯誤類型",
  "elapsed_time": 1.23
}
```

### 常見錯誤

#### 1. 連接超時
```json
{
  "success": false,
  "error": "無法在 5 秒內連接到串流: ...",
  "elapsed_time": 5.02
}
```

**解決**: 
- 檢查網路連接
- 確認 ESP32-CAM 運行中
- 檢查 IP 和端口

#### 2. 讀取畫面失敗
```json
{
  "success": false,
  "error": "無法讀取畫面",
  "elapsed_time": 1.5
}
```

**解決**:
- ESP32-CAM 可能暫時無回應
- 重新啟動 ESP32-CAM
- 檢查串流格式

#### 3. HTTP 請求超時
```json
{
  "success": false,
  "error": "HTTP 請求超時",
  "overall_status": "超時"
}
```

**解決**:
- 網路延遲過高
- ESP32-CAM 負載過重
- 減少並發請求

---

## 性能優化建議

### 參數調整

#### `imgsz` - 圖像大小
- **320**: 最快，但精度較低
- **416**: 推薦，平衡速度和精度 ⭐
- **640**: 最準確，但較慢

#### `conf` - 信心閾值
- **0.1-0.2**: 敏感，可能有誤報
- **0.3-0.5**: 推薦，平衡準確率 ⭐
- **0.6-0.9**: 嚴格，可能漏檢

### 調用頻率建議

| 工具 | 推薦頻率 | 最高頻率 |
|------|----------|----------|
| `check_stream_health` | 每分鐘 | 每秒 10 次 |
| `detect_stream_frame_simple` | 每秒 1-2 次 | 每秒 5 次 |
| `detect_stream_frame` | 每 2-5 秒 | 每秒 1 次 |

---

## MCP Client 配置建議

如果使用 Claude Desktop 或其他 MCP client，建議配置：

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "yolov8-detection": {
      "command": "python",
      "args": ["C:\\Users\\user\\MCPproject-YOLOv8\\MCPclient.py"],
      "timeout": 30000,  // 30 秒超時
      "env": {
        "PYTHONPATH": "C:\\Users\\user\\MCPproject-YOLOv8"
      }
    }
  }
}
```

### 重要設定
- `timeout`: 至少 10000 (10秒)，建議 30000 (30秒)
- `PYTHONPATH`: 確保可以找到模型檔案

---

## 測試範例

### 使用 Python 測試

```python
# test_tools.py
import json
from MCPclient import check_stream_health, detect_stream_frame_simple, detect_stream_frame

stream_url = "http://192.168.0.103:81/stream"

# 1. 健康檢查
health = check_stream_health(stream_url)
print("健康狀態:", health["overall_status"])

# 2. 簡化版偵測
result = detect_stream_frame_simple(stream_url, imgsz=416, conf=0.3)
print(f"偵測到 {result['detection_count']} 個物體")

# 3. 完整版偵測
result = detect_stream_frame(stream_url, imgsz=416, conf=0.3)
print(f"總耗時: {result['performance']['total_time']} 秒")
```

### 使用 MCP Client 測試

```json
// 1. 健康檢查
{
  "tool": "check_stream_health",
  "arguments": {
    "stream_url": "http://192.168.0.103:81/stream"
  }
}

// 2. 簡化版偵測
{
  "tool": "detect_stream_frame_simple",
  "arguments": {
    "stream_url": "http://192.168.0.103:81/stream",
    "imgsz": 416,
    "conf": 0.3
  }
}

// 3. 完整版偵測
{
  "tool": "detect_stream_frame",
  "arguments": {
    "stream_url": "http://192.168.0.103:81/stream",
    "imgsz": 416,
    "conf": 0.3,
    "timeout": 10
  }
}
```

---

## 總結

### 問題已解決 ✅

1. ✅ 診斷了 "Failed to fetch" 的原因
2. ✅ 提供了三個不同版本的工具
3. ✅ 增加了超時控制和錯誤處理
4. ✅ 提供了詳細的使用指南

### 建議的解決方案

**如果遇到 "Failed to fetch"**:

1. 🔍 先用 `check_stream_health` 檢查串流
2. ⚡ 優先使用 `detect_stream_frame_simple`（更快、更穩定）
3. 🎨 需要圖像時才用 `detect_stream_frame`
4. ⚙️ 檢查 MCP client 的超時配置

### 下一步

- 🔄 重新啟動 MCP server: `mcp install mcpclient.py`
- 🧪 使用 `check_stream_health` 測試連接
- 🚀 使用 `detect_stream_frame_simple` 進行偵測

---

**更新日期**: 2025-10-15  
**版本**: 2.0 (改進版)
