# @mcp.tool() 裝飾器完整介紹 v1.0

## 🎯 什麼是 @mcp.tool()？

`@mcp.tool()` 是 FastMCP 框架提供的一個裝飾器（decorator），它能夠**將普通的 Python 函數轉換成 Claude Desktop 可以呼叫的工具**。

---

## 🌟 核心概念

### **裝飾器是什麼？**

在 Python 中，裝飾器是一種特殊的語法，用來修改或增強函數的功能。想像一下，您有一個普通的函數：

```python
def say_hello(name):
    return f"你好，{name}！"
```

這個函數只能在 Python 程式中使用。但當您加上 `@mcp.tool()` 裝飾器：

```python
@mcp.tool()
def say_hello(name: str) -> dict:
    """向某人打招呼"""
    return {"message": f"你好，{name}！"}
```

這個函數就變成了一個**工具**，Claude Desktop 可以發現並呼叫它，就像使用自己的內建功能一樣。

---

## 📝 基本用法

### **最簡單的例子**

```python
from mcp.server.fastmcp import FastMCP

# 創建 MCP 服務器
mcp = FastMCP("我的服務器")

# 使用裝飾器註冊工具
@mcp.tool()
def add_numbers(a: int, b: int) -> dict:
    """將兩個數字相加"""
    result = a + b
    return {"result": result}

# 啟動服務器
if __name__ == "__main__":
    mcp.run()
```

當這個程式運行後，Claude Desktop 就能看到並使用 `add_numbers` 這個工具了。

---

## 🔍 @mcp.tool() 做了什麼？

### **1. 自動註冊工具**

當您在函數前加上 `@mcp.tool()`：

```python
@mcp.tool()
def detect_esp32_stream(...):
    pass
```

FastMCP 框架會自動執行以下步驟：
- 將這個函數加入到服務器的工具列表中
- 分配一個工具 ID（通常就是函數名稱）
- 讓 Claude Desktop 能夠發現這個工具

**類比**：就像在一個工具箱裡新增了一把工具，當 Claude Desktop 打開工具箱時，就能看到這個新工具並知道如何使用它。

---

### **2. 自動解析函數參數**

裝飾器會仔細檢查您的函數定義，理解每個參數的詳細資訊。

#### **參數名稱**
```python
def detect_esp32_stream(stream_url, imgsz, conf):
    #                    ↑         ↑     ↑
    #                    這些名稱會被記錄下來
```

#### **參數類型**
```python
def detect_esp32_stream(
    stream_url: str,      # ← 字串類型
    imgsz: int,           # ← 整數類型
    conf: float,          # ← 浮點數類型
    use_kalman: bool      # ← 布林值類型
):
```

MCP 會理解這些類型，並在接收到錯誤類型時自動拒絕。

#### **預設值**
```python
def detect_esp32_stream(
    stream_url: str = None,     # ← 預設是 None（可選參數）
    imgsz: int = 416,           # ← 預設是 416
    conf: float = 0.25,         # ← 預設是 0.25
    use_kalman: bool = True     # ← 預設是 True
):
```

當用戶呼叫工具時，如果沒有提供某個參數，MCP 會自動使用預設值。

**實際例子**：
```python
# 用戶只提供 conf 參數
detect_esp32_stream(conf=0.5)

# MCP 自動補齊其他參數
detect_esp32_stream(
    stream_url=None,     # 使用預設值
    imgsz=416,           # 使用預設值
    conf=0.5,            # 用戶提供的值
    frame_skip=1,        # 使用預設值
    use_kalman=True      # 使用預設值
)
```

---

### **3. 讀取並使用函數說明文件**

函數開頭的三引號文字（docstring）會成為工具的使用說明書：

```python
@mcp.tool()
def detect_esp32_stream(...) -> dict:
    """
    從預設的 ESP32-CAM 串流進行單幀 YOLO 物體偵測
    
    這個工具可以連接到 ESP32-CAM，捕獲影像並使用 YOLOv8 
    模型進行物體偵測。支援 Kalman Filter 追蹤以減少閃爍。
    
    Args:
        stream_url: 串流 URL（可選，預設使用預設 URL）
        imgsz: 圖像大小，預設 416
        conf: 信心閾值，預設 0.25
        ...
    
    Returns:
        dict: 包含偵測結果和標註圖像的字典
    """
```

**Claude Desktop 的使用**：
- Claude 會讀取這段說明
- 理解工具的功能和用途
- 知道每個參數的意義
- 向用戶解釋如何使用這個工具

**類比**：就像產品說明書，告訴使用者這個工具是做什麼的、怎麼用、有哪些選項。

---

### **4. 處理 JSON-RPC 通訊**

這是 `@mcp.tool()` 最重要的功能之一。它完全自動化了複雜的通訊過程。

#### **接收請求**

當 Claude Desktop 想要使用您的工具時，它會發送一個 JSON 格式的請求：

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "detect_esp32_stream",
    "arguments": {
      "frame_skip": 5,
      "conf": 0.3
    }
  }
}
```

#### **自動處理流程**

`@mcp.tool()` 裝飾器會自動完成以下步驟：

```
1. 從 stdin 讀取 JSON 請求
   ↓
2. 解析 JSON 字串
   ↓
3. 識別要呼叫的工具名稱（detect_esp32_stream）
   ↓
4. 提取參數（frame_skip=5, conf=0.3）
   ↓
5. 將 JSON 值轉換為 Python 類型
   - "5" (JSON 數字) → 5 (Python int)
   - "0.3" (JSON 數字) → 0.3 (Python float)
   ↓
6. 補齊未提供的參數（使用預設值）
   ↓
7. 呼叫您的 Python 函數
   result = detect_esp32_stream(
       stream_url=None,
       imgsz=416,
       conf=0.3,
       frame_skip=5,
       max_age=5,
       display_tolerance=3,
       use_kalman=True
   )
   ↓
8. 接收函數返回值（Python dict）
   ↓
9. 將 Python dict 轉換為 JSON 字串
   ↓
10. 寫入 stdout 發送回 Claude Desktop
```

**您不需要寫任何程式碼來處理這些步驟！** 全部由裝飾器自動完成。

---

### **5. 自動驗證參數**

裝飾器會在呼叫函數之前檢查參數是否正確。

#### **類型檢查**

如果您定義了參數類型：
```python
def detect_esp32_stream(imgsz: int, conf: float):
    pass
```

當用戶傳入錯誤類型時，MCP 會自動拒絕：

```python
# 錯誤示範
detect_esp32_stream(imgsz="abc", conf="xyz")

# MCP 自動檢測錯誤並回應
{
  "error": {
    "code": -32602,
    "message": "Invalid params: imgsz must be int, got str"
  }
}
```

#### **必填參數檢查**

如果參數沒有預設值，表示它是必填的：

```python
def check_stream_health(stream_url: str):  # stream_url 是必填的
    pass
```

當用戶沒有提供時：
```python
# 錯誤示範
check_stream_health()  # 忘記提供 stream_url

# MCP 自動回應錯誤
{
  "error": {
    "code": -32602,
    "message": "Missing required parameter: stream_url"
  }
}
```

---

### **6. 自動序列化返回值**

您的函數只需要返回一個 Python 字典（dict）：

```python
@mcp.tool()
def my_function():
    return {
        "success": True,
        "result": "完成",
        "count": 42,
        "items": ["item1", "item2"]
    }
```

裝飾器會自動將這個字典轉換成 JSON 格式：

```json
{
  "success": true,
  "result": "完成",
  "count": 42,
  "items": ["item1", "item2"]
}
```

然後通過 stdout 發送給 Claude Desktop。

**支援的數據類型轉換**：
| Python 類型 | JSON 類型 |
|------------|----------|
| dict | object |
| list | array |
| str | string |
| int, float | number |
| True, False | true, false |
| None | null |

---

## 🎬 完整運作流程示範

讓我們通過一個完整的例子，看看 `@mcp.tool()` 如何運作：

### **場景：用戶想要進行物體偵測**

#### **步驟 1：用戶在 Claude Desktop 輸入**
```
「使用 detect_esp32_stream 工具，設定 frame_skip 為 10，confidence 為 0.4」
```

#### **步驟 2：Claude Desktop 生成請求**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "detect_esp32_stream",
    "arguments": {
      "frame_skip": 10,
      "conf": 0.4
    }
  }
}
```

#### **步驟 3：@mcp.tool() 接收並處理**
```python
# 裝飾器內部處理（自動）
request = read_from_stdin()  # 讀取 JSON
params = parse_json(request)  # 解析 JSON

tool_name = params["params"]["name"]  # "detect_esp32_stream"
arguments = params["params"]["arguments"]  # {"frame_skip": 10, "conf": 0.4}

# 補齊預設值
full_arguments = {
    "stream_url": None,      # 預設值
    "imgsz": 416,            # 預設值
    "conf": 0.4,             # 用戶提供
    "frame_skip": 10,        # 用戶提供
    "max_age": 5,            # 預設值
    "display_tolerance": 3,  # 預設值
    "use_kalman": True       # 預設值
}
```

#### **步驟 4：呼叫實際的 Python 函數**
```python
# 裝飾器呼叫您的函數
result = detect_esp32_stream(**full_arguments)

# 您的函數執行
def detect_esp32_stream(stream_url, imgsz, conf, ...):
    # 連接 ESP32-CAM
    cap = cv2.VideoCapture(DEFAULT_STREAM_URL)
    
    # YOLO 推論
    results = model.predict(frame, imgsz=imgsz, conf=conf)
    
    # Kalman Filter 追蹤
    # ... 
    
    # 返回結果
    return {
        "success": True,
        "detections": [
            {
                "class": "person",
                "confidence": 0.87,
                "bbox": [100, 150, 300, 450]
            }
        ],
        "annotated_image_base64": "iVBORw0KG..."
    }
```

#### **步驟 5：@mcp.tool() 序列化並發送**
```python
# 裝飾器序列化返回值（自動）
response = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "success": True,
        "detections": [...],
        "annotated_image_base64": "..."
    }
}

json_response = json.dumps(response)
print(json_response, flush=True)  # 寫入 stdout
```

#### **步驟 6：Claude Desktop 接收並顯示**
```
✅ 偵測成功！

發現 1 個物體：
- person (信心度: 0.87)

[顯示標註後的圖像]
```

---

## 💡 為什麼需要 @mcp.tool()？

### **沒有裝飾器的情況（傳統方式）**

如果沒有 `@mcp.tool()`，您需要手動處理所有事情：

```python
# 傳統方式（需要大量額外程式碼）
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_endpoint():
    # 1. 手動解析 JSON
    data = request.get_json()
    
    # 2. 手動提取參數
    stream_url = data.get('stream_url')
    imgsz = data.get('imgsz', 416)  # 手動處理預設值
    conf = data.get('conf', 0.25)
    
    # 3. 手動驗證參數
    if not isinstance(imgsz, int):
        return jsonify({"error": "imgsz must be integer"}), 400
    
    # 4. 呼叫業務邏輯
    result = do_detection(stream_url, imgsz, conf)
    
    # 5. 手動序列化
    return jsonify(result)

# 6. 手動啟動 HTTP 伺服器
if __name__ == '__main__':
    app.run(port=5000)

# 7. 還需要處理：
#    - 錯誤處理
#    - 認證授權
#    - CORS
#    - 日誌記錄
#    - ...
```

**估計需要 200-300 行額外程式碼！**

---

### **使用 @mcp.tool() 之後**

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool()
def detect_esp32_stream(
    stream_url: str = None,
    imgsz: int = 416,
    conf: float = 0.25
) -> dict:
    """進行物體偵測"""
    # 只需要寫業務邏輯
    result = do_detection(stream_url, imgsz, conf)
    return result

mcp.run()
```

**只需要 10 行程式碼！** 

所有複雜的通訊、解析、驗證、序列化都由裝飾器自動處理。

---

## 🎓 裝飾器的運作原理

### **Python 裝飾器基礎**

```python
# 這兩種寫法是等價的

# 寫法 1：使用 @ 語法
@mcp.tool()
def my_function():
    pass

# 寫法 2：手動包裝
def my_function():
    pass
my_function = mcp.tool()(my_function)
```

裝飾器實際上是一個**包裝函數**，它接收原始函數，然後返回一個增強版的函數。

### **簡化版的 @mcp.tool() 實現**

```python
def tool():
    def decorator(func):
        # 1. 記錄這個函數
        registered_tools[func.__name__] = func
        
        # 2. 解析函數簽名
        signature = inspect.signature(func)
        tool_metadata = {
            "name": func.__name__,
            "description": func.__doc__,
            "parameters": {}
        }
        
        for param_name, param in signature.parameters.items():
            tool_metadata["parameters"][param_name] = {
                "type": param.annotation,
                "default": param.default
            }
        
        # 3. 創建包裝函數
        def wrapper(*args, **kwargs):
            # 驗證參數
            validate_parameters(kwargs, tool_metadata)
            
            # 呼叫原始函數
            result = func(*args, **kwargs)
            
            # 序列化結果
            return serialize_result(result)
        
        return wrapper
    
    return decorator
```

---

## 🔑 關鍵特性總結

### **1. 聲明式設計**

您只需要聲明函數的介面（參數、類型、預設值、說明），不需要關心實現細節：

```python
@mcp.tool()
def my_tool(param1: str, param2: int = 10) -> dict:
    """工具說明"""
    # 只需要寫業務邏輯
    return {"result": "success"}
```

### **2. 類型安全**

使用 Python 的類型提示，MCP 會自動驗證：

```python
imgsz: int        # 必須是整數
conf: float       # 必須是浮點數
use_kalman: bool  # 必須是布林值
```

### **3. 自動化一切**

- ✅ 自動註冊工具
- ✅ 自動解析參數
- ✅ 自動驗證類型
- ✅ 自動處理 JSON-RPC
- ✅ 自動序列化結果
- ✅ 自動錯誤處理

### **4. 零配置**

不需要編寫任何配置文件，函數本身就是完整的配置：

```python
@mcp.tool()  # ← 就這麼簡單！
def my_function(param: str) -> dict:
    """說明"""
    return {"result": param}
```

---

## 📊 實際應用案例

### **案例 1：完整功能工具**

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
    從預設的 ESP32-CAM 串流進行單幀 YOLO 物體偵測
    支援 FPS 控制和閃爍控制
    """
    # 業務邏輯...
    return {
        "success": True,
        "detections": [...],
        "annotated_image_base64": "..."
    }
```

**特點**：
- 7 個參數（1 個必填，6 個可選）
- 豐富的功能選項
- 詳細的說明文件

---

### **案例 2：簡化工具**

```python
@mcp.tool()
def detect_stream_frame_simple(
    stream_url: str, 
    imgsz: int = 416, 
    conf: float = 0.3
) -> dict:
    """
    簡化版串流偵測（不返回圖像，回應更快）
    """
    # 簡化的業務邏輯...
    return {
        "success": True,
        "detections": [...]
    }
```

**特點**：
- 只有 3 個參數
- 更快的執行速度
- 適合快速查詢

---

### **案例 3：診斷工具**

```python
@mcp.tool()
def check_stream_health(stream_url: str) -> dict:
    """
    快速檢查串流健康狀態（用於診斷連接問題）
    """
    # 診斷邏輯...
    return {
        "http_status": 200,
        "can_read_frame": True,
        "overall_status": "健康"
    }
```

**特點**：
- 單一必填參數
- 專注於診斷
- 不進行 YOLO 推論

---

## 🌟 最佳實踐

### **1. 清晰的參數命名**

```python
# ✅ 好的命名
def detect_stream(
    stream_url: str,
    image_size: int,
    confidence_threshold: float
):
    pass

# ❌ 不好的命名
def detect_stream(
    url: str,
    sz: int,
    th: float
):
    pass
```

### **2. 完整的類型提示**

```python
# ✅ 完整的類型提示
def my_tool(
    param1: str,
    param2: int = 10,
    param3: bool = True
) -> dict:
    pass

# ❌ 缺少類型提示
def my_tool(param1, param2=10, param3=True):
    pass
```

### **3. 詳細的 Docstring**

```python
@mcp.tool()
def my_tool(param1: str, param2: int) -> dict:
    """
    工具的簡短描述
    
    這裡可以寫更詳細的說明，解釋工具的用途、
    使用場景、注意事項等。
    
    Args:
        param1: 參數1的說明
        param2: 參數2的說明
    
    Returns:
        dict: 返回值的結構說明
        {
            "result": "結果",
            "status": "狀態"
        }
    """
    pass
```

### **4. 合理的預設值**

```python
@mcp.tool()
def detect_stream(
    stream_url: str = None,     # None 表示使用預設 URL
    imgsz: int = 416,            # 常用的 YOLO 輸入大小
    conf: float = 0.25,          # 平衡精度和召回率
    use_kalman: bool = True      # 預設啟用追蹤
) -> dict:
    pass
```

---

## 🎯 總結

### **@mcp.tool() 裝飾器是什麼？**

一個**智能翻譯機**，它能夠：
- 理解您的 Python 函數（參數、類型、說明）
- 將函數「翻譯」成 Claude Desktop 能理解的格式
- 自動處理所有複雜的通訊細節
- 讓您專注於業務邏輯，而不用擔心如何與 Claude Desktop 溝通

### **核心價值**

**用一句話總結**：
> `@mcp.tool()` 讓您的 Python 函數變成 Claude Desktop 的超能力！只要加上這個裝飾器，Claude 就能呼叫您的函數，就像使用自己的內建工具一樣簡單。

### **為什麼它很重要？**

1. **極簡設計**：只需一行裝飾器，就能註冊工具
2. **自動化**：處理所有複雜的通訊和驗證
3. **類型安全**：自動驗證參數類型
4. **零配置**：函數本身就是完整的配置
5. **專注業務**：讓您專注於實現功能，而不是處理技術細節

---

## 📚 相關文檔

- [MCP 相關程式碼介紹](v1_mcp_intro.md)
- [系統運作流程圖 v1.2](v1.2_workflow.md)
- [MCP 實現技術報告](mcp_implementation_technical_report.md)

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-06  
**作者**: YOLOv8 MCP Server Team
