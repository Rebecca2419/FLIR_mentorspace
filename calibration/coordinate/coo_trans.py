import cv2
import numpy as np

chessboardSize = (8, 5)
chessBlockLen = 46.0

cap = cv2.VideoCapture(3)
scale = 1.46

data = np.load('int_param.npz')
K = data['K']
dist = data['dist']

def calc_trans_param(corners):
    nx, ny = chessboardSize
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= chessBlockLen

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not ok:
        raise RuntimeError("sovlePnP failed")

    R_CW, _ = cv2.Rodrigues(rvec)
    t_CW = tvec.reshape(3, 1)

    R_WC = R_CW.T
    t_WC = - R_CW.T @ t_CW

    T_WC = np.eye(4, dtype=np.float32)
    T_WC[:3, :3] = R_WC
    T_WC[:3, 3] = t_WC.ravel()

    return T_WC


while True:
    ret, frame = cap.read()
    if not ret:
        raise SystemExit("Unable to capture new frame.")

    # Crop to fit FOV of IR camera
    h, w = frame.shape[:2]
    cropW = int(w / scale)
    cropH = int(h / scale)
    startX = (w - cropW) // 2
    startY = (h - cropH) // 2
    cropped = frame[startY:startY+cropH, startX:startX+cropW]
    cropped = cv2.resize(cropped, (w, h))
    frame = cropped

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cornerFound, corners = cv2.findChessboardCorners(gray, chessboardSize)
    
    display = frame.copy()
    if cornerFound:
        cv2.drawChessboardCorners(display, chessboardSize, corners, cornerFound)
        cv2.putText(display, "READY - Press S to save", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(display, "NOT READY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera', display)
    

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and cornerFound:
        T_WC = calc_trans_param(corners)
        np.savez('cood_trans_param.npz', T_WC=T_WC)
        print("Param saved to cood_trans_param.npz")
        print(T_WC)
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
