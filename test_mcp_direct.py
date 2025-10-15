"""
直接測試 MCP detect_stream_frame 函數
"""
import sys
import os
import json

# 設定路徑
sys.path.insert(0, os.path.dirname(__file__))

# 直接載入模組
import importlib.util
spec = importlib.util.spec_from_file_location("mcpclient", "MCPclient.py")
mcpclient = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcpclient)

detect_stream_frame = mcpclient.detect_stream_frame

print("=" * 60)
print("直接測試 detect_stream_frame 函數")
print("=" * 60)

stream_url = "http://192.168.0.103:81/stream"

print(f"\n測試 URL: {stream_url}")
print("開始偵測...\n")

result = detect_stream_frame(stream_url=stream_url, imgsz=416, conf=0.3)

print("=" * 60)
print("執行結果:")
print("=" * 60)

# 不顯示 base64 圖像（太長）
if "annotated_image_base64" in result:
    img_len = len(result["annotated_image_base64"])
    result_copy = result.copy()
    result_copy["annotated_image_base64"] = f"<base64 image, {img_len} chars>"
    print(json.dumps(result_copy, indent=2, ensure_ascii=False))
else:
    print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n" + "=" * 60)
if result.get("success"):
    print("✅ 偵測成功！")
    print(f"偵測到 {result.get('detection_count', 0)} 個物體")
    if result.get('detections'):
        print("\n偵測結果:")
        for i, det in enumerate(result['detections'], 1):
            print(f"  {i}. {det['class']} (信心度: {det['confidence']:.2f})")
else:
    print("❌ 偵測失敗")
    print(f"錯誤: {result.get('error')}")
print("=" * 60)
