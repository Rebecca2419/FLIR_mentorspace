import cv2
import os

# Chessboard inner corner configuration used for intrinsic calibration
chessboardSize = (8, 5)

# Open designated camera device
cap = cv2.VideoCapture(0)
imgCount = 0
scale = 1.46

# Directory for saving captured calibration images
saveDir = 'saved_img'
os.makedirs(saveDir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        # Exit program if camera frame cannot be read
        raise SystemExit("Unable to capture new frame.")
    
    # Rotate image to match camera mounting orientation
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Crop image to match the field of view of the IR camera
    h, w = frame.shape[:2]
    cropW = int(w / scale)
    cropH = int(h / scale)
    startX = (w - cropW) // 2
    startY = (h - cropH) // 2
    cropped = frame[startY:startY+cropH, startX:startX+cropW]
    frame = cropped

    # Convert image to grayscale for corner detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cornerFound, corners = cv2.findChessboardCorners(gray, chessboardSize)
    
    # Prepare image for visualization
    display = frame.copy()
    if cornerFound:
        # Draw detected chessboard corners for visual confirmation
        cv2.drawChessboardCorners(display, chessboardSize, corners, cornerFound)
        cv2.putText(display, "READY - Press S to save", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Indicate that the chessboard is not detected
        cv2.putText(display, "NOT READY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show number of images already saved
    cv2.putText(display, "%d saved"%imgCount, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('Camera', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and cornerFound:
        # Save current frame only if chessboard corners are detected
        cv2.imwrite("%s/img_%03d.png"%(saveDir, imgCount), frame)
        imgCount += 1
        print("%d saved."%imgCount)
    elif key == ord('q'):
        # Exit capture loop
        break

# Release camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
