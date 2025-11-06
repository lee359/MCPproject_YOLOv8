# MCP 相關程式碼介紹 v1.0

## 📚 概述

本文檔詳細介紹 `mcpclient.py` 中所有與 **MCP (Model Context Protocol)** 相關的程式碼，包括框架導入、服務器創建、工具註冊和服務器啟動。

---

## 🔷 MCP 程式碼完整列表

### **核心程式碼總覽**

| 行數 | 程式碼 | 用途 |
|------|--------|------|
| 2 | `from mcp.server.fastmcp import FastMCP, Image` | 導入 MCP 框架 |
| 29 | `mcp = FastMCP("YOLOv8 Detection Server")` | 創建 MCP 服務器實例 |
| 61 | `@mcp.tool()` | 註冊工具 1 (detect_esp32_stream) |
| 241 | `@mcp.tool()` | 註冊工具 2 (detect_stream_frame_simple) |
| 299 | `@mcp.tool()` | 註冊工具 3 (check_stream_health) |
| 370 | `@mcp.tool()` | 註冊工具 4 (detect_image) |
| 421 | `mcp.run()` | 啟動 MCP 服務器 |

**統計：MCP 相關程式碼僅 7 行，其餘 400+ 行為業務邏輯（YOLO、OpenCV、Kalman Filter）**

---

## 📖 逐段詳細說明

---

### **1. 導入 MCP 框架（第 2 行）**

```python
from mcp.server.fastmcp import FastMCP, Image
```

#### **說明**
這行程式碼從 MCP 的 FastMCP 模組中導入兩個類別：
- **FastMCP**：MCP 服務器的核心類，用於創建服務器實例和管理工具
- **Image**：MCP 的圖像類型（本專案中未使用，改用 base64 編碼字串）

#### **作用**
- 提供創建 MCP 服務器的基礎能力
- 提供 `@mcp.tool()` 裝飾器用於註冊工具函數
- 提供 `mcp.run()` 方法用於啟動服務器

#### **依賴關係**
```python
fastmcp 包
  ├─ mcp.server.fastmcp.FastMCP  # 服務器核心
  └─ mcp.server.fastmcp.Image    # 圖像處理（未使用）
```

---

### **2. 創建 MCP 服務器實例（第 29 行）**

```python
# Create an MCP server
mcp = FastMCP("YOLOv8 Detection Server")
```

#### **說明**
創建一個名為 "YOLOv8 Detection Server" 的 MCP 服務器實例。

#### **參數**
- **"YOLOv8 Detection Server"**：服務器的顯示名稱
  - 這個名稱會顯示在 Claude Desktop 的工具列表中
  - 幫助用戶識別這是一個 YOLOv8 物體偵測服務

#### **在 Claude Desktop 配置中的對應**
```json
// claude_desktop_config.json
{
  "mcpServers": {
    "YOLOv8 Detection Server": {  // ← 與這裡的名稱對應
      "command": "C:\\Users\\user\\MCPproject-YOLOv8\\venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\user\\MCPproject-YOLOv8\\mcpclient.py"]
    }
  }
}
```

#### **實例變數 `mcp` 的用途**
後續程式碼會使用這個 `mcp` 實例來：
1. 註冊工具：`@mcp.tool()`
2. 啟動服務器：`mcp.run()`

---

### **3. 註冊工具 1 - detect_esp32_stream（第 61 行）**

```python
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
        frame_skip: 跳幀數量（越大 FPS 越低），預設 1
        max_age: 物體最大存活幀數（預設 5）
        display_tolerance: 顯示容忍幀數（預設 3）
        use_kalman: 是否使用 Kalman Filter（預設 True）
    
    Returns:
        dict: 包含偵測結果和註釋圖像的字典
    """
```

#### **功能**
這是系統的**主要工具**，提供完整的物體偵測和追蹤功能：
- 連接到 ESP32-CAM 獲取影像串流
- 使用 YOLOv8 進行物體偵測
- 使用 Kalman Filter 進行物體追蹤（可選）
- 返回標註後的圖像（Base64 編碼）

#### **參數設計**
| 參數 | 類型 | 預設值 | 用途 |
|------|------|--------|------|
| stream_url | str | None | 串流 URL（None 時使用預設 URL）|
| imgsz | int | 416 | YOLO 推論時的圖像大小 |
| conf | float | 0.25 | 信心度閾值（過濾低質量檢測）|
| frame_skip | int | 1 | 跳幀控制（調整 FPS）|
| max_age | int | 5 | 物體未匹配時的最大保留幀數 |
| display_tolerance | int | 3 | 顯示物體的年齡容忍度 |
| use_kalman | bool | True | 是否啟用 Kalman Filter 追蹤 |

#### **返回值結構**
```json
{
  "success": true,
  "detections": [
    {
      "class": "person",
      "confidence": 0.87,
      "bbox": [100, 150, 300, 450],
      "age": 0
    }
  ],
  "detection_count": 1,
  "tracked_objects_count": 3,
  "annotated_image_base64": "iVBORw0KG...",
  "frame_size": {"height": 480, "width": 640},
  "parameters": {...},
  "performance": {
    "total_time": 1.23,
    "predict_time": 0.45
  }
}
```

---

### **4. 註冊工具 2 - detect_stream_frame_simple（第 241 行）**

```python
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
```

#### **功能**
**簡化版**的偵測工具，適合需要快速獲取檢測結果的場景：
- 連接到指定串流 URL
- 讀取單幀並進行 YOLO 推論
- **不使用** Kalman Filter 追蹤
- **不返回**標註圖像（減少數據傳輸量）

#### **與工具 1 的差異**
| 特性 | detect_esp32_stream | detect_stream_frame_simple |
|------|---------------------|---------------------------|
| 預設 URL | ✅ 有 | ❌ 必須提供 |
| Kalman 追蹤 | ✅ 支援 | ❌ 不支援 |
| 返回圖像 | ✅ Base64 | ❌ 不返回 |
| 多幀處理 | ✅ 是 | ❌ 單幀 |
| 處理速度 | 較慢 | **更快** |

#### **適用場景**
- 只需要知道「有沒有物體」、「是什麼物體」
- 不需要查看標註圖像
- 需要更快的回應速度

---

### **5. 註冊工具 3 - check_stream_health（第 299 行）**

```python
@mcp.tool()
def check_stream_health(stream_url: str) -> dict:
    """
    快速檢查串流健康狀態（用於診斷連接問題）
    
    Args:
        stream_url: 串流 URL
    
    Returns:
        dict: 串流健康狀態和診斷資訊
    """
```

#### **功能**
**診斷工具**，用於快速檢查串流連接狀態，不進行物體偵測：
- 測試 HTTP 連接狀態
- 測試 OpenCV 是否能成功連接
- 測試是否能成功讀取影像幀
- 返回連接時間統計

#### **返回值結構**
```json
{
  "url": "http://192.168.0.103:81/stream",
  "http_status": 200,
  "http_time": 0.123,
  "content_type": "multipart/x-mixed-replace",
  "can_receive_data": true,
  "opencv_opened": true,
  "opencv_connect_time": 0.234,
  "can_read_frame": true,
  "frame_read_time": 0.045,
  "frame_size": {
    "height": 480,
    "width": 640,
    "channels": 3
  },
  "success": true,
  "overall_status": "健康"
}
```

#### **適用場景**
- ESP32-CAM 連接有問題時進行診斷
- 檢查網路是否通暢
- 驗證串流 URL 是否正確
- 檢查影像解析度

---

### **6. 註冊工具 4 - detect_image（第 370 行）**

```python
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
```

#### **功能**
處理**本地靜態圖片**的偵測工具：
- 讀取本地圖片文件
- 使用 YOLOv8 進行物體偵測
- 返回標註後的圖像（Base64 編碼）
- **不連接** ESP32-CAM
- **不使用** Kalman Filter

#### **參數**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| image_path | str | 必填 | 圖片的絕對或相對路徑 |
| imgsz | int | 640 | YOLO 推論圖像大小（靜態圖片常用更高解析度）|
| conf | float | 0.3 | 信心度閾值 |

#### **適用場景**
- 測試 YOLO 模型在特定圖片上的表現
- 處理已保存的圖片
- 離線批次處理圖片

---

### **7. 啟動 MCP 服務器（第 421 行）**

```python
if __name__ == "__main__":
   mcp.run()
```

#### **說明**

##### **`if __name__ == "__main__":`**
這是 Python 的標準寫法，確保只有在**直接執行此腳本**時才會啟動服務器。

**直接執行（會啟動服務器）：**
```bash
python mcpclient.py
```

**被其他程式導入（不會啟動服務器）：**
```python
# 其他檔案
from mcpclient import detect_image  # 只導入函數，不啟動服務器
```

##### **`mcp.run()`**
啟動 MCP 服務器的主循環，進入 **stdio 監聽模式**。

#### **運行機制**

## 執行流程：
1. 進入無限循環
2. 從 stdin 讀取 JSON-RPC 請求
3. 解析請求並路由到對應的工具函數
4. 執行工具函數
5. 將返回值序列化為 JSON
6. 寫入 stdout 回傳給 Claude Desktop
7. 回到步驟 2（持續監聽）

```python
# 簡化版的 mcp.run() 內部邏輯
def run():
    while True:  # 無限循環
        # 1. 從 stdin 讀取 JSON-RPC 請求
        request = read_from_stdin()
        
        # 2. 解析請求
        method = request.get("method")
        params = request.get("params", {})
        
        # 3. 路由到對應的工具函數
        if method == "tools/call":
            tool_name = params["name"]
            arguments = params["arguments"]
            
            # 呼叫對應的工具函數
            if tool_name == "detect_esp32_stream":
                result = detect_esp32_stream(**arguments)
            elif tool_name == "detect_stream_frame_simple":
                result = detect_stream_frame_simple(**arguments)
            elif tool_name == "check_stream_health":
                result = check_stream_health(**arguments)
            elif tool_name == "detect_image":
                result = detect_image(**arguments)
            
            # 4. 將結果序列化為 JSON
            response = json.dumps({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": result
            })
            
            # 5. 寫入 stdout 回傳
            print(response, flush=True)
```

#### **stdio 通訊模式**

```
┌─────────────────────────────────────────────────────────────┐
│ Claude Desktop (主進程)                                      │
└────────────┬────────────────────────┬───────────────────────┘
             │ stdin (標準輸入)        │ stdout (標準輸出)
             ↓                        ↑
    ┌────────────────┐       ┌────────────────┐
    │ JSON-RPC 請求   │       │ JSON-RPC 回應   │
    │ (序列化)        │       │ (反序列化)      │
    └────────┬───────┘       └───────┬────────┘
             ↓                        ↑
┌────────────┴────────────────────────┴───────────────────────┐
│ MCP Server (mcpclient.py) - 子進程                          │
│  mcp.run()                                                  │
│  • 從 stdin 讀取請求                                         │
│  • 路由到工具函數                                            │
│  • 將結果寫入 stdout                                         │
└─────────────────────────────────────────────────────────────┘
```

**重要規則：**
- ✅ **stdout** 只能用於 MCP JSON 通訊
- ✅ **stderr** 用於日誌和錯誤訊息
- ❌ **不能**在 stdout 輸出任何非 JSON 內容（會破壞協議）

---

## 📊 MCP 程式碼結構圖

```python
mcpclient.py
│
├─ 📦 導入模組 (第 1-13 行)
│   ├─ from mcp.server.fastmcp import FastMCP, Image  ← MCP 框架
│   ├─ from ultralytics import YOLO
│   ├─ import cv2, numpy, torch, ...
│   └─ ...
│
├─ 🔧 初始化 (第 15-29 行)
│   ├─ SCRIPT_DIR, MODEL_PATH
│   ├─ model = YOLO(MODEL_PATH)
│   ├─ GPU 設備選擇
│   └─ mcp = FastMCP("YOLOv8 Detection Server")  ← 創建 MCP 服務器
│
├─ 🛠️ 輔助函數 (第 31-59 行)
│   ├─ create_kalman_filter()
│   ├─ get_center()
│   └─ calculate_distance()
│
├─ 🎯 MCP 工具函數 (第 61-418 行)
│   │
│   ├─ @mcp.tool()  ← MCP 裝飾器
│   │   def detect_esp32_stream(...) -> dict:
│   │       # 主要偵測工具（完整功能）
│   │
│   ├─ @mcp.tool()  ← MCP 裝飾器
│   │   def detect_stream_frame_simple(...) -> dict:
│   │       # 簡化偵測工具（快速回應）
│   │
│   ├─ @mcp.tool()  ← MCP 裝飾器
│   │   def check_stream_health(...) -> dict:
│   │       # 健康檢查工具（診斷用）
│   │
│   └─ @mcp.tool()  ← MCP 裝飾器
│       def detect_image(...) -> dict:
│           # 靜態圖片偵測工具
│
└─ 🚀 啟動服務器 (第 420-421 行)
    if __name__ == "__main__":
        mcp.run()  ← 啟動 MCP 服務器
```
## 結構圖:
┌─────────────────────────────────────────────────────────────┐
│ 1. 導入 MCP 框架                                             │
├─────────────────────────────────────────────────────────────┤
│ from mcp.server.fastmcp import FastMCP, Image               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 創建 MCP 服務器實例                                       │
├─────────────────────────────────────────────────────────────┤
│ mcp = FastMCP("YOLOv8 Detection Server")                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 註冊工具函數（使用裝飾器）                                 │
├─────────────────────────────────────────────────────────────┤
│ @mcp.tool()                                                 │
│ def detect_esp32_stream(...) -> dict:                      │
│     # 主要偵測功能                                           │
│                                                             │
│ @mcp.tool()                                                 │
│ def detect_stream_frame_simple(...) -> dict:               │
│     # 簡化偵測功能                                           │
│                                                             │
│ @mcp.tool()                                                 │
│ def check_stream_health(...) -> dict:                      │
│     # 健康檢查功能                                           │
│                                                             │
│ @mcp.tool()                                                 │
│ def detect_image(...) -> dict:                             │
│     # 靜態圖片偵測功能                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 啟動服務器                                                │
├─────────────────────────────────────────────────────────────┤
│ if __name__ == "__main__":                                  │
│     mcp.run()  # 進入 stdio 監聽模式                        │
└─────────────────────────────────────────────────────────────┘
---

## 🔄 MCP 完整運作流程

### **階段 1：啟動**
```bash
# Claude Desktop 啟動子進程
C:\...\python.exe mcpclient.py
```

### **階段 2：初始化**
```python
# 執行 mcpclient.py
model = YOLO(MODEL_PATH)              # 載入 YOLO 模型
mcp = FastMCP("YOLOv8 Detection Server")  # 創建 MCP 服務器
# ... 註冊 4 個工具（透過 @mcp.tool()）
mcp.run()                              # 進入監聽模式
```

### **階段 3：監聽請求**
```python
# mcp.run() 進入無限循環
while True:
    request = read_from_stdin()  # 等待 Claude Desktop 發送請求
```

### **階段 4：處理請求**
```
Claude Desktop 發送：
┌────────────────────────────────────────┐
│ {                                      │
│   "method": "tools/call",              │
│   "params": {                          │
│     "name": "detect_esp32_stream",     │
│     "arguments": {                     │
│       "frame_skip": 5                  │
│     }                                  │
│   }                                    │
│ }                                      │
└────────────────────────────────────────┘
         ↓
MCP Server 處理：
┌────────────────────────────────────────┐
│ 1. 解析 JSON                           │
│ 2. 識別工具名稱                         │
│ 3. 提取參數                            │
│ 4. 呼叫 detect_esp32_stream(frame_skip=5)
│ 5. 執行 YOLO 推論                      │
│ 6. 返回結果                            │
└────────────────────────────────────────┘
```

### **階段 5：回傳結果**
```
MCP Server 回傳：
┌────────────────────────────────────────┐
│ {                                      │
│   "success": true,                     │
│   "detections": [...],                 │
│   "annotated_image_base64": "..."      │
│ }                                      │
└────────────────────────────────────────┘
         ↓
Claude Desktop 顯示：
┌────────────────────────────────────────┐
│ ✅ 偵測成功                            │
│ 📊 發現 2 個物體                       │
│ 🖼️  [顯示標註圖像]                     │
└────────────────────────────────────────┘
```

### **階段 6：循環**
```
回到階段 3，繼續等待下一個請求...
```

---

## 🎯 MCP 程式碼統計

### **程式碼行數分析**
| 類型 | 行數 | 佔比 | 說明 |
|------|------|------|------|
| **MCP 核心程式碼** | 7 | 1.7% | import, 創建實例, 裝飾器, run() |
| 業務邏輯 | 400+ | 98.3% | YOLO, OpenCV, Kalman Filter |
| **總計** | ~421 | 100% | |

### **MCP 工具統計**
| 工具名稱 | 參數數量 | 返回圖像 | 用途 |
|---------|---------|---------|------|
| detect_esp32_stream | 7 | ✅ | 主要偵測（完整功能）|
| detect_stream_frame_simple | 3 | ❌ | 快速偵測（簡化版）|
| check_stream_health | 1 | ❌ | 健康檢查（診斷用）|
| detect_image | 3 | ✅ | 靜態圖片偵測 |

---

## 💡 關鍵特性

### **1. 極簡設計**
只需 7 行 MCP 相關程式碼就能將 Python 函數轉換為 Claude Desktop 可用的工具。

### **2. 自動化處理**
- ✅ 自動解析函數參數（名稱、類型、預設值）
- ✅ 自動驗證參數類型
- ✅ 自動處理 JSON-RPC 通訊
- ✅ 自動序列化返回值

### **3. 類型安全**
使用 Python 類型提示（type hints），MCP 框架會自動驗證：
```python
stream_url: str      # 必須是字串
imgsz: int = 416     # 必須是整數
conf: float = 0.25   # 必須是浮點數
use_kalman: bool     # 必須是布林值
```

### **4. Docstring 即文檔**
函數的 docstring 自動成為工具說明，Claude 可以理解並向用戶解釋工具用途。

---

## 📚 相關文檔

- [@mcp.tool() 裝飾器詳細介紹](v1_@mcp.tool_intro.md)
- [系統運作流程圖 v1.2](v1.2_workflow.md)
- [MCP 實現技術報告](mcp_implementation_technical_report.md)
- [Claude Desktop 設置指南](claude_desktop_setup_guide.md)

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-06  
**作者**: YOLOv8 MCP Server Team
