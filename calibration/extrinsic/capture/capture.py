import cv2
import os

scale = 1.46
chessboardSize = (8, 5)

cap1 = cv2.VideoCapture(3)
cap2 = cv2.VideoCapture(1)

saveDir = 'saved_img'
os.makedirs(saveDir+"/cam1", exist_ok=True)
os.makedirs(saveDir+"/cam2", exist_ok=True)

imgCount = 0

while True:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        raise SystemExit("Unable to capture new frame.")

    # Crop to fit FOV of IR camera
    h, w = frame1.shape[:2]
    cropW = int(w / scale)
    cropH = int(h / scale)
    startX = (w - cropW) // 2
    startY = (h - cropH) // 2
    frame1 = cv2.resize(frame1[startY:startY+cropH, startX:startX+cropW], (w, h))
    frame2 = cv2.resize(frame2[startY:startY+cropH, startX:startX+cropW], (w, h))
    
    display1 = frame1.copy()
    display2 = frame2.copy()
    
    # Detect chessboard
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    retC1, corners1 = cv2.findChessboardCorners(gray1, chessboardSize)
    retC2, corners2 = cv2.findChessboardCorners(gray2, chessboardSize)
    
    if retC1:
        cv2.drawChessboardCorners(display1, chessboardSize, corners1, retC1)
    if retC2:
        cv2.drawChessboardCorners(display2, chessboardSize, corners2, retC2)
    
    # Show preview
    status = "READY - Press S to save" if retC1 and retC2 else "NOT READY"
    cv2.putText(display1, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(display2, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(display1, "%d saved"%imgCount, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(display2, "%d saved"%imgCount, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('Camera 1', display1)
    cv2.imshow('Camera 2', display2)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and retC1 and retC2:
        cv2.imwrite("%s/cam1/img_%03d.png"%(saveDir, imgCount), frame1)
        cv2.imwrite("%s/cam2/img_%03d.png"%(saveDir, imgCount), frame2)
        imgCount += 1
        print("%d saved."%imgCount)
    elif key == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

print("%d pairs saved in total."%imgCount)