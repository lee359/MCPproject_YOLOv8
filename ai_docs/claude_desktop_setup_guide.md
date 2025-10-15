    # Claude Desktop 連接 YOLOv8 Detection Server 設置指南

## 問題描述
無法在 Claude Desktop 內部連接到 YOLOv8 Detection Server

## 根本原因
使用 `uv run` 命令時，會創建隔離的 Python 環境，不包含 venv 中已安裝的套件（如 PIL、ultralytics、opencv-python 等）。

---

## ✅ 解決方案

### 步驟 1: 更新 Claude Desktop 配置

**配置檔案位置**: `C:\Users\user\AppData\Roaming\Claude\claude_desktop_config.json`

**正確的配置**:
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "node",
      "args": [
        "C:\\Program Files\\nodejs\\node_modules\\@modelcontextprotocol\\server-filesystem\\dist\\index.js",
        "C:\\Users\\user\\runs\\detect"
      ]
    },
    "YOLOv8 Detection Server": {
      "command": "C:\\Users\\user\\MCPproject-YOLOv8\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\user\\MCPproject-YOLOv8\\mcpclient.py"
      ],
      "env": {
        "PYTHONPATH": "C:\\Users\\user\\MCPproject-YOLOv8"
      }
    }
  }
}
```

**關鍵變更**:
- ❌ 舊的（錯誤）: 使用 `uv.exe run --with mcp[cli] mcp run`
- ✅ 新的（正確）: 直接使用 `python.exe mcpclient.py`

### 步驟 2: 重新啟動 Claude Desktop

**重要**: 必須完全關閉並重新啟動 Claude Desktop

1. 完全退出 Claude Desktop（右鍵系統托盤 → 退出）
2. 重新打開 Claude Desktop
3. 等待 MCP servers 載入（可能需要 10-20 秒）

### 步驟 3: 驗證連接

在 Claude Desktop 中輸入：
```
請列出可用的工具
```

應該會看到以下 MCP tools：
- ✅ `add` - 測試工具
- ✅ `check_stream_health` - 串流健康檢查
- ✅ `detect_stream_frame_simple` - 簡化版串流偵測
- ✅ `detect_stream_frame` - 完整版串流偵測
- ✅ `detect_image` - 圖片偵測

---

## 🔍 診斷步驟

### 如果仍然無法連接

#### 診斷 1: 手動測試 MCP Server 啟動

在 PowerShell 中執行：
```powershell
C:\Users\user\MCPproject-YOLOv8\venv\Scripts\python.exe C:\Users\user\MCPproject-YOLOv8\mcpclient.py
```

**預期結果**: 
- 應該會停在那裡（無錯誤訊息）
- 這表示 MCP server 正在運行並等待連接
- 按 `Ctrl+C` 停止

**如果看到錯誤**:
- 檢查錯誤訊息
- 可能是套件缺失或模型檔案問題

#### 診斷 2: 檢查 Claude Desktop 日誌

1. 打開 Claude Desktop
2. 按 `Ctrl+Shift+I` 打開開發者工具
3. 切換到 "Console" 標籤
4. 查看是否有 MCP server 相關的錯誤訊息

**常見錯誤訊息**:
- `Failed to start MCP server`: 檢查 Python 路徑是否正確
- `ModuleNotFoundError`: 檢查 venv 是否包含所有必要套件
- `Timeout`: 增加超時時間或檢查 server 啟動速度

#### 診斷 3: 檢查 Python 環境

執行診斷腳本：
```powershell
python diagnose_mcp_server.py
```

**檢查清單**:
- ✅ Python 版本: 3.8+
- ✅ mcp 已安裝
- ✅ ultralytics 已安裝
- ✅ cv2 (opencv-python) 已安裝
- ✅ PIL (pillow) 已安裝
- ✅ best.pt 模型檔案存在

---

## 🚨 常見問題

### 問題 1: "No module named 'mcp'"

**原因**: venv 中沒有安裝 mcp 套件

**解決**:
```powershell
C:\Users\user\MCPproject-YOLOv8\venv\Scripts\pip.exe install mcp
```

### 問題 2: "No module named 'PIL'"

**原因**: venv 中沒有安裝 pillow

**解決**:
```powershell
C:\Users\user\MCPproject-YOLOv8\venv\Scripts\pip.exe install pillow
```

### 問題 3: "No module named 'ultralytics'"

**原因**: venv 中沒有安裝 ultralytics

**解決**:
```powershell
C:\Users\user\MCPproject-YOLOv8\venv\Scripts\pip.exe install ultralytics
```

### 問題 4: MCP Server 啟動但 Claude Desktop 看不到

**可能原因**:
1. Claude Desktop 配置檔案路徑錯誤
2. Claude Desktop 沒有完全重啟
3. 檔案路徑使用了相對路徑而非絕對路徑

**解決**:
1. 確認配置檔案中所有路徑都是絕對路徑
2. 確認路徑中使用雙反斜線 `\\` 或單斜線 `/`
3. 完全退出並重啟 Claude Desktop

### 問題 5: "Failed to fetch" 錯誤

**原因**: 這是我們之前解決的問題，工具執行超時

**解決**: 使用改進的工具
- 優先使用 `check_stream_health` 測試連接
- 使用 `detect_stream_frame_simple` 而非 `detect_stream_frame`

---

## 📋 完整安裝檢查清單

### 1. Python 環境檢查
```powershell
# 檢查 Python 版本
C:\Users\user\MCPproject-YOLOv8\venv\Scripts\python.exe --version

# 檢查已安裝套件
C:\Users\user\MCPproject-YOLOv8\venv\Scripts\pip.exe list
```

**必須包含的套件**:
- mcp
- ultralytics
- opencv-python
- pillow (PIL)
- numpy
- torch
- requests

### 2. 檔案檢查
```powershell
# 檢查檔案是否存在
Test-Path C:\Users\user\MCPproject-YOLOv8\mcpclient.py
Test-Path C:\Users\user\MCPproject-YOLOv8\best.pt
Test-Path C:\Users\user\MCPproject-YOLOv8\venv\Scripts\python.exe
```

全部應該返回 `True`

### 3. 配置檔案檢查

**檢查**:
```powershell
Get-Content "C:\Users\user\AppData\Roaming\Claude\claude_desktop_config.json"
```

**必須包含**:
- `"command"`: 指向正確的 `python.exe`
- `"args"`: 包含 `mcpclient.py` 的完整路徑
- 使用絕對路徑（不是相對路徑）
- 路徑使用 `\\` 或 `/`（不混用）

---

## 🎯 測試流程

### 完整測試步驟

1. **測試 Python 環境**
   ```powershell
   C:\Users\user\MCPproject-YOLOv8\venv\Scripts\python.exe -c "import mcp, ultralytics, cv2, PIL; print('All packages OK')"
   ```

2. **測試 MCP Server 啟動**
   ```powershell
   # 啟動 server（會持續運行）
   C:\Users\user\MCPproject-YOLOv8\venv\Scripts\python.exe C:\Users\user\MCPproject-YOLOv8\mcpclient.py
   
   # 應該看到無輸出（正常）
   # 按 Ctrl+C 停止
   ```

3. **更新 Claude Desktop 配置**
   - 編輯 `claude_desktop_config.json`
   - 使用上面提供的正確配置

4. **重啟 Claude Desktop**
   - 完全關閉（系統托盤 → 退出）
   - 重新打開
   - 等待 20 秒讓 servers 載入

5. **測試連接**
   在 Claude Desktop 中測試：
   ```
   請使用 check_stream_health 工具檢查這個串流: http://192.168.0.103:81/stream
   ```

---

## ✨ 成功指標

當一切正常時，你應該看到：

### 在 Claude Desktop 中

1. **工具列表可見**
   - 可以看到所有 5 個工具
   - 工具描述正確顯示

2. **健康檢查成功**
   ```json
   {
     "success": true,
     "overall_status": "健康",
     "http_status": 200,
     "can_read_frame": true
   }
   ```

3. **偵測成功**
   ```json
   {
     "success": true,
     "detection_count": 0,
     "elapsed_time": 0.16
   }
   ```

---

## 🔧 進階設定

### 增加日誌輸出

如果需要更詳細的日誌，可以修改 `mcpclient.py` 開頭：

```python
# 在檔案開頭加入
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_server.log'),
        logging.StreamHandler()
    ]
)
```

### 調整超時設定

在 `claude_desktop_config.json` 中可以加入超時設定：

```json
{
  "mcpServers": {
    "YOLOv8 Detection Server": {
      "command": "C:\\Users\\user\\MCPproject-YOLOv8\\venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\user\\MCPproject-YOLOv8\\mcpclient.py"],
      "env": {
        "PYTHONPATH": "C:\\Users\\user\\MCPproject-YOLOv8"
      },
      "timeout": 30000
    }
  }
}
```

---

## 📞 支援資訊

### 如果仍然無法連接

1. **執行完整診斷**
   ```powershell
   python diagnose_mcp_server.py > diagnosis.txt
   ```

2. **檢查日誌檔案**
   - MCP server 日誌: `mcp_server.log`
   - Claude Desktop Console（開發者工具）

3. **收集資訊**
   - Python 版本
   - 已安裝套件清單
   - 錯誤訊息
   - Claude Desktop 版本

---

**最後更新**: 2025-10-15  
**狀態**: 已修復並測試通過
