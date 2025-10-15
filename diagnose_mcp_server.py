"""
測試 MCP Server 是否能正常啟動
"""
import sys
import subprocess
import json

print("=" * 70)
print("MCP Server 啟動測試")
print("=" * 70)

# 測試 1: 檢查 Python 環境
print("\n[測試 1] 檢查 Python 環境")
print("-" * 70)
python_exe = r"C:\Users\user\MCPproject-YOLOv8\venv\Scripts\python.exe"
try:
    result = subprocess.run([python_exe, "--version"], capture_output=True, text=True, timeout=5)
    print(f"✅ Python 版本: {result.stdout.strip()}")
except Exception as e:
    print(f"❌ Python 檢查失敗: {e}")

# 測試 2: 檢查必要套件
print("\n[測試 2] 檢查必要套件")
print("-" * 70)
packages = ["mcp", "ultralytics", "cv2", "PIL"]
for pkg in packages:
    try:
        result = subprocess.run(
            [python_exe, "-c", f"import {pkg}; print('{pkg} 已安裝')"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
        else:
            print(f"❌ {pkg} 未安裝或有錯誤")
    except Exception as e:
        print(f"❌ {pkg} 檢查失敗: {e}")

# 測試 3: 檢查模型檔案
print("\n[測試 3] 檢查模型檔案")
print("-" * 70)
import os
model_path = r"C:\Users\user\MCPproject-YOLOv8\best.pt"
if os.path.exists(model_path):
    size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ 模型檔案存在: {model_path}")
    print(f"   檔案大小: {size:.2f} MB")
else:
    print(f"❌ 模型檔案不存在: {model_path}")

# 測試 4: 測試載入 mcpclient 模組
print("\n[測試 4] 測試載入 mcpclient 模組")
print("-" * 70)
try:
    sys.path.insert(0, r"C:\Users\user\MCPproject-YOLOv8")
    import mcpclient
    print("✅ mcpclient 模組載入成功")
    
    # 檢查可用的 tools
    if hasattr(mcpclient, 'mcp'):
        print("✅ FastMCP 實例存在")
        # 嘗試獲取工具列表
        print("\n可用的 MCP Tools:")
        for name in dir(mcpclient):
            obj = getattr(mcpclient, name)
            if callable(obj) and not name.startswith('_') and name not in ['YOLO', 'FastMCP', 'Image', 'PILImage']:
                print(f"   - {name}")
    else:
        print("❌ FastMCP 實例不存在")
        
except Exception as e:
    print(f"❌ 模組載入失敗: {e}")
    import traceback
    traceback.print_exc()

# 測試 5: 檢查 uv 工具
print("\n[測試 5] 檢查 uv 工具")
print("-" * 70)
uv_path = r"C:\Users\user\MCPproject-YOLOv8\venv\Scripts\uv.exe"
if os.path.exists(uv_path):
    print(f"✅ uv 工具存在: {uv_path}")
    try:
        result = subprocess.run([uv_path, "--version"], capture_output=True, text=True, timeout=5)
        print(f"   版本: {result.stdout.strip()}")
    except Exception as e:
        print(f"   無法獲取版本: {e}")
else:
    print(f"❌ uv 工具不存在: {uv_path}")

# 測試 6: 模擬 MCP 啟動
print("\n[測試 6] 測試 MCP Server 啟動")
print("-" * 70)
print("嘗試啟動 MCP server（5 秒超時）...")
try:
    cmd = [
        uv_path,
        "run",
        "--with",
        "mcp[cli]",
        "mcp",
        "run",
        r"C:\Users\user\MCPproject-YOLOv8\mcpclient.py"
    ]
    print(f"命令: {' '.join(cmd)}")
    
    # 使用短暫超時測試啟動
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=5,
        cwd=r"C:\Users\user\MCPproject-YOLOv8"
    )
    print(f"返回碼: {result.returncode}")
    if result.stdout:
        print(f"輸出:\n{result.stdout}")
    if result.stderr:
        print(f"錯誤:\n{result.stderr}")
        
except subprocess.TimeoutExpired:
    print("⏱️ 啟動超時（這可能是正常的，因為 MCP server 會持續運行）")
except Exception as e:
    print(f"❌ 啟動失敗: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("診斷完成")
print("=" * 70)
