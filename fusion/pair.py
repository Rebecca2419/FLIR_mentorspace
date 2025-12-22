import numpy as np
from detect import DetectCamera
from matcher import StereoMatcher

class PairDetecter:

    # Manage two cameras and perform stereo-based coordinate estimation
    def __init__(self, camIndex1, camIndex2, extParamFilename, factor):
        # Initialize foreground detectors for both cameras
        self.cam1 = DetectCamera(camIndex1)
        self.cam2 = DetectCamera(camIndex2)
        # Initialize stereo matcher with extrinsic calibration parameters
        self.matcher = StereoMatcher(extParamFilename, factor)
    
    def update_background(self):
        # Update background models for both cameras
        self.cam1.update_background()
        self.cam2.update_background()

    def get_coordinate(self):
        # Detect foreground objects in both camera views
        frame1, result1 = self.cam1.detect()
        frame2, result2 = self.cam2.detect()

        # Extract valid mass centers from detection results
        massCenter1 = [item[1] for item in result1 if item[1] != None]
        massCenter2 = [item[1] for item in result2 if item[1] != None]

        # Match points between two views and compute real-world coordinates
        origCoodList = self.matcher.match_and_locate(massCenter1, massCenter2, maxDist=50)
        
        return (frame1, frame2), origCoodList
    
    def release(self):
        # Release camera resources
        self.cam1.release()
        self.cam2.release()
