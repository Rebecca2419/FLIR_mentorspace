import numpy as np
from detect import DetectCamera
from matcher import StereoMatcher

class PairDetecter:

    def __init__(self, camIndex1, camIndex2, extParamFilename, factor):
        self.cam1 = DetectCamera(camIndex1)
        self.cam2 = DetectCamera(camIndex2)
        self.matcher = StereoMatcher(extParamFilename, factor)
    
    def update_background(self):
        self.cam1.update_background()
        self.cam2.update_background()

    def get_coordinate(self):
        frame1, result1 = self.cam1.detect()
        frame2, result2 = self.cam2.detect()

        massCenter1 = [item[1] for item in result1 if item[1] != None]
        massCenter2 = [item[1] for item in result2 if item[1] != None]

        origCoodList = self.matcher.match_and_locate(massCenter1, massCenter2, maxDist=100)
        
        return (frame1, frame2), origCoodList
    
    def release(self):
        self.cam1.release()
        self.cam2.release()