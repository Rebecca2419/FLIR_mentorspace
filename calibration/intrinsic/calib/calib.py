import cv2
import numpy as np
import glob

chessboardSize = (8, 5)
chessBlockLen = 46.0

# Load images
print("Loading images from file...")

images = sorted(glob.glob('../capture/saved_img/*.png'))
print("%d images found."%len(images))

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2) * chessBlockLen

objpoints = []
imgpoints = []

successCnt = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize)
    
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        successCnt += 1
        print("  success.")
    else:
        print("  corner not found in %s."%fname)

print("Success: %d of %d."%(successCnt, len(images)))

if successCnt < 10:
    print("No enough image pairs.")
    exit(1)


# Execute intrinsic calibration
print("Executing intrinsic calibration...")

imgShape = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2GRAY).shape[::-1]

ret, K, dist, rvecs, tvecs = \
    cv2.calibrateCamera(
        objpoints,
        imgpoints,
        imgShape,
        None,
        None
    )

print("Intrinsic calibration finish.")
print(K)
print(dist)
print("Re-projection error: %.3f px."%ret)

np.savez('int_param.npz', K=K, dist=dist)
print("intrinsic parameters saved to int_param.npz")