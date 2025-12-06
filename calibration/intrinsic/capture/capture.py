import cv2
import os

chessboardSize = (8, 5)

cap = cv2.VideoCapture(0)
imgCount = 0
scale = 1.46

saveDir = 'saved_img'
os.makedirs(saveDir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        raise SystemExit("Unable to capture new frame.")
    
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Crop to fit FOV of IR camera
    h, w = frame.shape[:2]
    cropW = int(w / scale)
    cropH = int(h / scale)
    startX = (w - cropW) // 2
    startY = (h - cropH) // 2
    cropped = frame[startY:startY+cropH, startX:startX+cropW]
    # cropped = cv2.resize(cropped, (w, h))
    frame = cropped

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cornerFound, corners = cv2.findChessboardCorners(gray, chessboardSize)
    
    display = frame.copy()
    if cornerFound:
        cv2.drawChessboardCorners(display, chessboardSize, corners, cornerFound)
        cv2.putText(display, "READY - Press S to save", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(display, "NOT READY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display, "%d saved"%imgCount, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('Camera', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and cornerFound:
        cv2.imwrite("%s/img_%03d.png"%(saveDir, imgCount), frame)
        imgCount += 1
        print("%d saved."%imgCount)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()