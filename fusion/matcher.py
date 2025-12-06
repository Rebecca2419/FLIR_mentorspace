import cv2
import numpy as np

class StereoMatcher:

    def __init__(self, extParamFilename):
        data = np.load(extParamFilename)
        self.K1 = data['K1']
        self.K2 = data['K2']
        self.dist1 = data['dist1']
        self.dist2 = data['dist2']
        self.R = data['R']
        self.T = data['T']
        self.F = data['F']

        self.P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P2 = self.K2 @ np.hstack([self.R, self.T])

    def _triangulate(self, pt1, pt2):
        """
        (Interal function) Calculate coordinate using triangulation method
        """
        p1 = np.array([[pt1]], dtype=np.float32)
        p2 = np.array([[pt2]], dtype=np.float32)
        
        p1_undist = cv2.undistortPoints(p1, self.K1, self.dist1, P=self.K1)
        p2_undist = cv2.undistortPoints(p2, self.K2, self.dist2, P=self.K2)
        
        points_4d = cv2.triangulatePoints(
            self.P1,
            self.P2,
            p1_undist.reshape(2, 1),
            p2_undist.reshape(2, 1)
        )
        
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d[0, 0], points_3d[1, 0], points_3d[2, 0]
    
    def match_and_locate(self, points1, points2, maxDist=100):
        """
        Calculate coordinates in actural space.
        
        Args:
            points1: List of points in camera1 [(x1, y1), (x2, y2), ...]
            points2: List of points in camera2 [(x1, y1), (x2, y2), ...]
            maxDist: Thershold of polar distance
        
        Returns:
            results: [(X,Y,Z), ...]
        """
        results = []
        
        for pt1 in points1:
            p = np.array([pt1[0], pt1[1], 1.0])
            epiline = self.F @ p
            a, b, c = epiline
            
            bestPt2 = None
            bestDist = maxDist
            
            for pt2 in points2:
                dist = abs(a*pt2[0] + b*pt2[1] + c) / np.sqrt(a**2 + b**2)
                if dist < bestDist:
                    bestDist = dist
                    bestPt2 = pt2
            
            if bestPt2 is not None:
                X, Y, Z = self._triangulate(pt1, bestPt2)
                results.append((int(X * 1.0), int(Z * 1.65)))
        
        return results
