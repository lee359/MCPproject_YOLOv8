"""
測試更新後的 MCP tools
"""
import sys
import os
import json
import importlib.util

# 載入更新後的模組
spec = importlib.util.spec_from_file_location("mcpclient", "MCPclient.py")
mcpclient = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcpclient)

stream_url = "http://192.168.0.103:81/stream"

print("=" * 70)
print("測試更新後的 MCP Tools")
print("=" * 70)

# 測試 1: check_stream_health
print("\n[測試 1] check_stream_health - 健康檢查")
print("-" * 70)
result = mcpclient.check_stream_health(stream_url)
print(json.dumps(result, indent=2, ensure_ascii=False))

# 測試 2: detect_stream_frame_simple
print("\n[測試 2] detect_stream_frame_simple - 簡化版偵測（無圖像）")
print("-" * 70)
result = mcpclient.detect_stream_frame_simple(stream_url, imgsz=416, conf=0.3)
print(json.dumps(result, indent=2, ensure_ascii=False))

# 測試 3: detect_stream_frame
print("\n[測試 3] detect_stream_frame - 完整版偵測（含圖像）")
print("-" * 70)
result = mcpclient.detect_stream_frame(stream_url, imgsz=416, conf=0.3, timeout=10)

# 不顯示完整 base64
if "annotated_image_base64" in result:
    img_len = len(result["annotated_image_base64"])
    result_copy = result.copy()
    result_copy["annotated_image_base64"] = f"<base64 image, {img_len} chars>"
    print(json.dumps(result_copy, indent=2, ensure_ascii=False))
else:
    print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n" + "=" * 70)
print("所有測試完成！")
print("=" * 70)
