# CSV 記錄功能使用指南

## 📋 功能說明

已在 `mcpclient.py` 中添加 **自動 CSV 記錄功能**，每次呼叫 `detect_stream_frame_simple` 工具時，會自動將結果記錄到 CSV 檔案。

---

## 📊 CSV 檔案格式

### 檔案位置
```
C:\Users\user\MCPproject-YOLOv8\detection_logs.csv
```

### CSV 欄位

| 欄位名稱 | 資料類型 | 說明 | 範例 |
|---------|---------|------|------|
| `timestamp` | 字串 | 偵測時間 | `2025-11-07 14:30:45` |
| `success` | 布林值 | 是否成功 | `True` / `False` |
| `detection_count` | 整數 | 偵測到的物體數量 | `2` |
| `detections` | JSON 字串 | 詳細偵測結果 | `[{"class":"dog","confidence":0.92,"bbox":[...]}]` |
| `total_time` | 浮點數 | 總處理時間（秒） | `0.123` |
| `yolo_inference_time` | 浮點數 | YOLO 推理時間（秒） | `0.050` |

### CSV 範例內容

```csv
timestamp,success,detection_count,detections,total_time,yolo_inference_time
2025-11-07 14:30:45,True,2,"[{""class"":""dog"",""confidence"":0.92,""bbox"":[100,200,300,400]},{""class"":""dog"",""confidence"":0.88,""bbox"":[350,180,500,380]}]",0.123,0.05
2025-11-07 14:31:12,True,1,"[{""class"":""cat"",""confidence"":0.85,""bbox":[150,220,280,350]}]",0.118,0.048
2025-11-07 14:31:45,True,0,"[]",0.095,0.042
```

---

## 🚀 使用方式

### 1. 在 Claude Desktop 中測試

**執行偵測**:
```
請使用 detect_stream_frame_simple 工具偵測這個串流：
http://192.168.0.104:81/stream
```

**自動記錄**: 每次執行後，結果會自動追加到 `detection_logs.csv`

### 2. 查看記錄

**方法 1: 使用測試腳本**
```powershell
python test_csv_logging.py
```

**方法 2: 直接開啟 CSV**
```powershell
# 用 Excel 開啟
start detection_logs.csv

# 用記事本開啟
notepad detection_logs.csv
```

**方法 3: 使用 Python 分析**
```python
import pandas as pd

# 讀取 CSV
df = pd.read_csv('detection_logs.csv')

# 顯示最近 10 筆
print(df.tail(10))

# 計算統計
print(f"平均偵測數: {df['detection_count'].mean():.2f}")
print(f"平均時間: {df['total_time'].mean():.3f} 秒")
```

---

## 📈 進階分析範例

### 分析偵測效能

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_csv('detection_logs.csv')

# 轉換時間戳記為 datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 繪製偵測數量趨勢圖
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['detection_count'], marker='o')
plt.xlabel('時間')
plt.ylabel('偵測數量')
plt.title('物體偵測數量趨勢')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('detection_trend.png')

# 繪製處理時間分布圖
plt.figure(figsize=(10, 5))
plt.hist(df['total_time'], bins=20, alpha=0.7, label='總時間')
plt.hist(df['yolo_inference_time'], bins=20, alpha=0.7, label='YOLO 時間')
plt.xlabel('時間（秒）')
plt.ylabel('次數')
plt.title('處理時間分布')
plt.legend()
plt.tight_layout()
plt.savefig('time_distribution.png')

print("✅ 圖表已儲存")
```

### 計算成功率

```python
import pandas as pd

df = pd.read_csv('detection_logs.csv')

total = len(df)
success_count = df['success'].sum()
success_rate = (success_count / total) * 100

print(f"總執行次數: {total}")
print(f"成功次數: {success_count}")
print(f"成功率: {success_rate:.2f}%")
```

---

## 🔧 程式碼修改說明

### 1. 新增模組導入

```python
import csv
import json
```

### 2. 定義 CSV 檔案路徑

```python
CSV_LOG_PATH = os.path.join(SCRIPT_DIR, "detection_logs.csv")
```

### 3. 新增 `log_detection_to_csv()` 函數

```python
def log_detection_to_csv(result_data):
    """將偵測結果記錄到 CSV 檔案"""
    fieldnames = ['timestamp', 'success', 'detection_count', 'detections', 'total_time', 'yolo_inference_time']
    file_exists = os.path.isfile(CSV_LOG_PATH)
    
    with open(CSV_LOG_PATH, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row_data = {
            'timestamp': result_data.get('timestamp', ''),
            'success': result_data.get('success', False),
            'detection_count': result_data.get('detection_count', 0),
            'detections': json.dumps(result_data.get('detections', []), ensure_ascii=False),
            'total_time': result_data.get('total_time', 0),
            'yolo_inference_time': result_data.get('yolo_inference_time', 0)
        }
        
        writer.writerow(row_data)
```

### 4. 修改 `detect_stream_frame_simple()` 函數

```python
# 在返回結果前記錄
result = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "success": True,
    "detection_count": len(detections),
    "detections": detections,
    "total_time": round(total_time, 3),
    "yolo_inference_time": round(yolo_time, 3)
}

# ✅ 記錄到 CSV
log_detection_to_csv(result)

return result
```

---

## ⚠️ 注意事項

### 1. CSV 檔案會自動建立
- 首次執行時會自動建立 `detection_logs.csv`
- 會自動寫入欄位標題列

### 2. 資料會持續追加
- 每次執行都會在檔案末尾新增一列
- **不會覆蓋**舊資料

### 3. detections 欄位格式
- 以 JSON 字串儲存
- 可以用 `json.loads()` 解析
```python
import json
detections = json.loads(row['detections'])
```

### 4. 錯誤也會被記錄
- `success = False` 時仍會記錄
- 可用於分析失敗原因

---

## 🧹 維護與管理

### 清空記錄

```powershell
# 刪除 CSV 檔案
Remove-Item detection_logs.csv

# 下次執行時會自動重新建立
```

### 備份記錄

```powershell
# 複製檔案並加上日期
Copy-Item detection_logs.csv "detection_logs_backup_$(Get-Date -Format 'yyyyMMdd').csv"
```

### 檔案大小管理

如果記錄太多，可以定期清理：

```python
import pandas as pd

# 只保留最近 1000 筆記錄
df = pd.read_csv('detection_logs.csv')
df_recent = df.tail(1000)
df_recent.to_csv('detection_logs.csv', index=False)
```

---

## 📝 完整測試流程

### 步驟 1: 重啟 Claude Desktop

```powershell
# 1. 完全關閉 Claude Desktop
taskkill /F /IM "Claude.exe"

# 2. 等待 15 秒
Start-Sleep -Seconds 15

# 3. 重新啟動
Start-Process "C:\Users\user\AppData\Local\Programs\claude\Claude.exe"

# 4. 等待 30 秒
Start-Sleep -Seconds 30
```

### 步驟 2: 執行偵測（在 Claude Desktop 中）

```
請使用 detect_stream_frame_simple 工具偵測串流 http://192.168.0.104:81/stream
```

### 步驟 3: 檢查 CSV 記錄

```powershell
python test_csv_logging.py
```

### 步驟 4: 多次測試

重複執行步驟 2，觀察 CSV 檔案的增長

### 步驟 5: 分析結果

```python
import pandas as pd

df = pd.read_csv('detection_logs.csv')
print(df.describe())
```

---

## ✅ 成功指標

### CSV 記錄功能正常運作時：

1. ✅ `detection_logs.csv` 檔案自動建立
2. ✅ 每次執行偵測後，CSV 增加一列
3. ✅ 所有 6 個欄位都有正確的值
4. ✅ 時間戳記格式正確 (`YYYY-MM-DD HH:MM:SS`)
5. ✅ detections 欄位包含有效的 JSON
6. ✅ 成功和失敗的執行都被記錄

---

## 🎯 使用場景

### 1. 效能測試
- 記錄不同參數下的處理時間
- 分析 YOLO 推理效能

### 2. 準確度評估
- 統計偵測數量分布
- 分析信心度變化

### 3. 系統穩定性監控
- 追蹤成功率
- 記錄錯誤模式

### 4. 資料收集
- 累積實驗數據
- 用於論文或報告

---

**版本**: 1.0  
**最後更新**: 2025-11-07  
**相容**: Python 3.7+  
**依賴**: csv, json, pandas (選用，用於分析)
