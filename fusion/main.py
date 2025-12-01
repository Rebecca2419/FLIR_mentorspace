import cv2
import numpy as np
from detect import DetectCamera
from matcher import StereoMatcher

cam1 = DetectCamera(3)
cam2 = DetectCamera(1)
matcher = StereoMatcher('ext_param.npz')

cam1.update_background()
cam2.update_background()

cood_param = np.load('cood_trans_param.npz')
T_WC = cood_param['T_WC']

def add_detail(frame, result):
    cv2.putText(frame, "Press B to update background, Q to quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    for ROI, massCenter, areaSize in result:
        cv2.rectangle(frame, (ROI[0],ROI[1]), (ROI[0] + ROI[2], ROI[1] + ROI[3]), (0,255,0), 2)
        if massCenter != None:
            cv2.circle(frame, massCenter, 5, (0,0,255), -1)
            cv2.putText(frame, str(areaSize), (ROI[0], ROI[1]+ROI[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

def trans_cood(origCoodList):
    resultList = []
    for origCood in origCoodList:
        P_C = np.array([origCood[0], origCood[1], origCood[2], 1.0], dtype=np.float32).reshape(4, 1)
        P_W = T_WC @ P_C
        X, Y, Z, _ = P_W.ravel()
        resultList.append((int(X), int(Y), int(Z)))
    return resultList

while True:
    frame1, result1 = cam1.detect()
    frame2, result2 = cam2.detect()

    massCenter1 = [item[1] for item in result1 if item[1] != None]
    massCenter2 = [item[1] for item in result2 if item[1] != None]

    results = matcher.match_and_locate(massCenter1, massCenter2, maxDist=100)
    actualCood = trans_cood(results)
    cv2.putText(frame1, str(actualCood), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    
    add_detail(frame1, result1)
    cv2.imshow("camera1", frame1)

    add_detail(frame2, result2)
    cv2.imshow("camera2", frame2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('b'):
        cam1.update_background()
        cam2.update_background()

cam1.release()