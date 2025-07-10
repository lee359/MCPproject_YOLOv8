import cv2
cap = cv2.VideoCapture('https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=15771')

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()             # 讀取影片的每一幀
    if not ret:
        print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
        # 出現錯誤就再讀取一次，避免程式到此處就停止
        cap = cv2.VideoCapture('https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=15771')
        continue
    cv2.imshow('traffic', frame)     # 如果讀取成功，顯示該幀的畫面
    if cv2.waitKey(1) == ord('q'):      # 每一毫秒更新一次，直到按下 q 結束
        break
cap.release()                           # 所有作業都完成後，釋放資源
cv2.destroyAllWindows()                 # 結束所有視窗