import cv2
import requests
import time

# 測試 URL
stream_url = "http://192.168.0.103:81/stream"

print("=" * 60)
print("ESP32-CAM 串流連接診斷測試")
print("=" * 60)

# 測試 1: HTTP 請求測試
print("\n[測試 1] HTTP 請求測試...")
try:
    response = requests.get(stream_url, timeout=5, stream=True)
    print(f"✅ HTTP 狀態碼: {response.status_code}")
    print(f"✅ Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
    if response.status_code == 200:
        # 讀取前 100 bytes
        chunk = next(response.iter_content(100))
        print(f"✅ 接收到數據: {len(chunk)} bytes")
    response.close()
except requests.exceptions.Timeout:
    print("❌ 連接超時 - ESP32-CAM 可能沒有回應")
except requests.exceptions.ConnectionError as e:
    print(f"❌ 連接錯誤: {e}")
except Exception as e:
    print(f"❌ HTTP 請求失敗: {e}")

# 測試 2: OpenCV VideoCapture 測試
print("\n[測試 2] OpenCV VideoCapture 測試...")
try:
    cap = cv2.VideoCapture(stream_url)
    print(f"VideoCapture 物件已創建")
    
    # 設定緩衝區
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 檢查是否打開
    is_opened = cap.isOpened()
    print(f"cap.isOpened(): {is_opened}")
    
    if is_opened:
        print("✅ 串流已連接")
        
        # 嘗試讀取一幀
        print("\n[測試 3] 嘗試讀取畫面...")
        for i in range(3):
            ret, frame = cap.read()
            if ret:
                print(f"✅ 第 {i+1} 次讀取成功 - 畫面大小: {frame.shape}")
            else:
                print(f"❌ 第 {i+1} 次讀取失敗")
            time.sleep(0.1)
    else:
        print("❌ 無法打開串流")
        
        # 嘗試獲取更多資訊
        print("\n[額外診斷] 嘗試不同的 OpenCV 後端...")
        backends = [
            (cv2.CAP_FFMPEG, "FFMPEG"),
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
        ]
        
        for backend_id, backend_name in backends:
            try:
                cap2 = cv2.VideoCapture(stream_url, backend_id)
                if cap2.isOpened():
                    print(f"✅ {backend_name} 後端可以連接")
                    ret, frame = cap2.read()
                    if ret:
                        print(f"   畫面大小: {frame.shape}")
                    cap2.release()
                else:
                    print(f"❌ {backend_name} 後端無法連接")
            except Exception as e:
                print(f"❌ {backend_name} 後端錯誤: {e}")
    
    cap.release()
    
except Exception as e:
    print(f"❌ OpenCV 測試失敗: {e}")
    import traceback
    traceback.print_exc()

# 測試 4: 檢查 ESP32-CAM 根路徑
print("\n[測試 4] 檢查 ESP32-CAM Web 介面...")
try:
    root_url = "http://192.168.0.103"
    response = requests.get(root_url, timeout=5)
    print(f"✅ 根路徑狀態碼: {response.status_code}")
    if "stream" in response.text.lower():
        print("✅ 網頁內容包含 'stream' 關鍵字")
except Exception as e:
    print(f"❌ 無法訪問根路徑: {e}")

print("\n" + "=" * 60)
print("診斷完成")
print("=" * 60)
