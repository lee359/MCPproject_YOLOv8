"""
測試 CSV 記錄功能
"""
import os
import csv
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_LOG_PATH = os.path.join(SCRIPT_DIR, "detection_logs.csv")

print("=" * 70)
print("CSV 記錄功能測試")
print("=" * 70)

print(f"\n[1] CSV 檔案位置")
print(f"    {CSV_LOG_PATH}")

if os.path.exists(CSV_LOG_PATH):
    print(f"\n[2] CSV 檔案狀態: ✅ 存在")
    
    # 讀取並顯示 CSV 內容
    try:
        df = pd.read_csv(CSV_LOG_PATH)
        print(f"\n[3] 記錄總數: {len(df)} 筆")
        
        print(f"\n[4] CSV 欄位:")
        for col in df.columns:
            print(f"    - {col}")
        
        print(f"\n[5] 最近 5 筆記錄:")
        print(df.tail(5).to_string(index=False))
        
        print(f"\n[6] 統計摘要:")
        print(f"    - 成功次數: {df['success'].sum()}")
        print(f"    - 失敗次數: {len(df) - df['success'].sum()}")
        print(f"    - 平均偵測數: {df['detection_count'].mean():.2f}")
        print(f"    - 平均總時間: {df['total_time'].mean():.3f} 秒")
        print(f"    - 平均 YOLO 時間: {df['yolo_inference_time'].mean():.3f} 秒")
        
    except Exception as e:
        print(f"❌ 讀取 CSV 失敗: {e}")
else:
    print(f"\n[2] CSV 檔案狀態: ⚠️ 尚未建立")
    print(f"    提示: 執行 detect_stream_frame_simple 工具後會自動建立")

print("\n" + "=" * 70)
print("測試說明")
print("=" * 70)
print("\n1. 在 Claude Desktop 中執行:")
print("   請使用 detect_stream_frame_simple 工具偵測串流")
print("\n2. 每次執行後，會自動記錄到 detection_logs.csv")
print("\n3. 再次執行此腳本查看記錄:")
print("   python test_csv_logging.py")
print("\n4. CSV 格式:")
print("   timestamp,success,detection_count,detections,total_time,yolo_inference_time")
