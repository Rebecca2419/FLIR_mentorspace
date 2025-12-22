import cv2
import numpy as np
import glob

# Chessboard inner corner configuration
chessboardSize = (8, 5)
# Physical length of one chessboard square (in millimeters)
chessBlockLen = 46.0

print("Loading images from file...")

# Load all captured calibration images
images = sorted(glob.glob('../capture/saved_img/*.png'))
print("%d images found."%len(images))

# Prepare 3D object points in the chessboard coordinate system
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2) * chessBlockLen

# Lists for storing 3D object points and corresponding 2D image points
objpoints = []
imgpoints = []

successCnt = 0

for fname in images:
    img = cv2.imread(fname)
    # Convert image to grayscale for corner detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize)
    
    if ret:
        # Store valid 3D–2D point correspondence
        objpoints.append(objp)
        imgpoints.append(corners)
        successCnt += 1
        print("  success.")
    else:
        # Report images where chessboard detection failed
        print("  corner not found in %s."%fname)

# Report number of successful detections
print("Success: %d of %d."%(successCnt, len(images)))

if successCnt < 10:
    # Abort calibration if not enough valid image pairs
    print("No enough image pairs.")
    exit(1)


# Execute intrinsic calibration
print("Executing intrinsic calibration...")

# Image resolution used for calibration
imgShape = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2GRAY).shape[::-1]

# Estimate intrinsic camera matrix and distortion coefficients
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

# Save intrinsic parameters to file
np.savez('int_param.npz', K=K, dist=dist)
print("intrinsic parameters saved to int_param.npz")
