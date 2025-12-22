import cv2
import os

# Scaling factor used to match the field of view of the IR camera
scale = 1.46
# Chessboard inner corner configuration for extrinsic calibration
chessboardSize = (6, 5)

# Open camera devices
cap1 = cv2.VideoCapture(3)
cap2 = cv2.VideoCapture(1)

# Directories for saving synchronized image pairs
saveDir = 'saved_img'
os.makedirs(saveDir+"/cam1", exist_ok=True)
os.makedirs(saveDir+"/cam2", exist_ok=True)

imgCount = 0

while True:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        # Exit if either camera fails to provide a frame
        raise SystemExit("Unable to capture new frame.")
    
    # Rotate frames to correct camera mounting orientation
    frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
    frame2 = cv2.rotate(frame2, cv2.ROTATE_180)

    # Crop images to match the field of view of the IR camera
    h, w = frame1.shape[:2]
    cropW = int(w / scale)
    cropH = int(h / scale)
    startX = (w - cropW) // 2
    startY = (h - cropH) // 2
    frame1 = frame1[startY:startY+cropH, startX:startX+cropW]
    frame2 = frame2[startY:startY+cropH, startX:startX+cropW]
    
    # Create copies for visualization
    display1 = frame1.copy()
    display2 = frame2.copy()
    
    # Convert images to grayscale for chessboard detection
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Detect chessboard corners in both camera images
    retC1, corners1 = cv2.findChessboardCorners(gray1, chessboardSize)
    retC2, corners2 = cv2.findChessboardCorners(gray2, chessboardSize)
    
    if retC1:
        # Draw detected corners for camera 1
        cv2.drawChessboardCorners(display1, chessboardSize, corners1, retC1)
    if retC2:
        # Draw detected corners for camera 2
        cv2.drawChessboardCorners(display2, chessboardSize, corners2, retC2)
    
    # Show capture status and preview
    status = "READY - Press S to save" if retC1 and retC2 else "NOT READY"
    cv2.putText(display1, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(display2, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(display1, "%d saved"%imgCount, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(display2, "%d saved"%imgCount, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('Camera 1', display1)
    cv2.imshow('Camera 2', display2)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and retC1 and retC2:
        # Save synchronized image pair only when both chessboards are detected
        cv2.imwrite("%s/cam1/img_%03d.png"%(saveDir, imgCount), frame1)
        cv2.imwrite("%s/cam2/img_%03d.png"%(saveDir, imgCount), frame2)
        imgCount += 1
        print("%d saved."%imgCount)
    elif key == ord('q'):
        # Exit capture loop
        break

# Release camera devices and close all windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()

# Report total number of saved image pairs
print("%d pairs saved in total."%imgCount)
