import cv2
import numpy as np
import glob

chessboardSize = (6, 5)
chessBlockLen = 86.0

# Load image pairs
print("Loading image pairs from file...")

images1 = sorted(glob.glob('../capture/saved_img/cam1/*.png'))
images2 = sorted(glob.glob('../capture/saved_img/cam2/*.png'))
print("%d pairs found."%len(images1))

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2) * chessBlockLen

objPoints = []
imgPoints1 = []
imgPoints2 = []

successCnt = 0

for (fname1, fname2) in zip(images1, images2):
    img1 = cv2.imread(fname1)
    img2 = cv2.imread(fname2)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    ret1, corners1 = cv2.findChessboardCorners(gray1, chessboardSize)
    ret2, corners2 = cv2.findChessboardCorners(gray2, chessboardSize)
    
    if ret1 and ret2:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        
        objPoints.append(objp)
        imgPoints1.append(corners1)
        imgPoints2.append(corners2)
        successCnt += 1
        print("  success.")

    else:
        print("  corner not found in %s - cam1: %s, cam2: %s."%(fname1, str(ret1), str(ret2)))

print("Success: %d of %d."%(successCnt, len(images1)))

if successCnt < 10:
    print("No enough image pairs.")
    exit(1)


# Execute extrinsic calibration
print("Executing extrinsic calibration...")

data1 = np.load('cam1_int_param.npz')
K1 = data1['K']
dist1 = data1['dist']

data2 = np.load('cam2_int_param.npz')
K2 = data2['K']
dist2 = data2['dist']

imgShape = cv2.cvtColor(cv2.imread(images1[0]), cv2.COLOR_BGR2GRAY).shape[::-1]

ret, newK1, newDist1, newK2, newDist2, R, T, E, F = \
    cv2.stereoCalibrate(
        objPoints,
        imgPoints1,
        imgPoints2,
        K1,
        dist1,
        K2,
        dist2,
        imgShape,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

print("Extrinsic calibration finish.")
print("Re-projection error: %.3f px."%ret)

np.savez('ext_param.npz', K1=K1, dist1=dist1, K2=K2, dist2=dist2, R=R, T=T, E=E, F=F)
print("Extrinsic parameters saved to ext_param.npz")