from ultralytics import YOLO

# 加載訓練好的模型
model = YOLO("best.pt")

# 測試單張圖像
results = model.predict(source="123.jpg", save=True, imgsz=640, data=None)
print(results)

# 測試文件夾中的多張圖像
#results = model.predict(source="path/to/your/images_folder", save=True, imgsz=640)
#print(results)

# 測試視頻
#results = model.predict(source="D:/YOLO3/黑煙測試影片.mp4", save=True, imgsz=640)
#print(results)
#results = model.predict(source="D:/YOLO3/黑煙測試影片.mp4", save=True, imgsz=640)


