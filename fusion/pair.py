import numpy as np
from detect import DetectCamera
from matcher import StereoMatcher

class PairDetecter:

    def __init__(self, camIndex1, camIndex2, coodParamFilename, extParamFilename):
        self.cam1 = DetectCamera(camIndex1)
        self.cam2 = DetectCamera(camIndex2)
        coodParam = np.load(coodParamFilename)
        self.T_WC = coodParam['T_WC']
        self.matcher = StereoMatcher(extParamFilename)
    
    def update_background(self):
        self.cam1.update_background()
        self.cam2.update_background()

    def get_coordinate(self):
        frame1, result1 = self.cam1.detect()
        frame2, result2 = self.cam2.detect()

        massCenter1 = [item[1] for item in result1 if item[1] != None]
        massCenter2 = [item[1] for item in result2 if item[1] != None]

        origCoodList = self.matcher.match_and_locate(massCenter1, massCenter2, maxDist=100)
        
        # resultList = []
        # for origCood in origCoodList:
        #     P_C = np.array([origCood[0], origCood[1], origCood[2], 1.0], dtype=np.float32).reshape(4, 1)
        #     P_W = self.T_WC @ P_C
        #     X, Y, Z, _ = P_W.ravel()
        #     resultList.append((int(X), int(Y), int(Z)))
        
        # return (frame1, frame2), resultList
        return (frame1, frame2), origCoodList
    
    def release(self):
        self.cam1.release()
        self.cam2.release()