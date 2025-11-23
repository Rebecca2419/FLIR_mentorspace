import cv2


class DetectCamera:

    def __init__(self, camIndex: int, minAreaRatio=0.003, border=5, blur=(15,15), threshold=32):
        """
        Args:
            camIndex: Index of designated camera
            minAreaRatio: Minimal ratio of ROI area under consideration 
            border: Boundary dead zone
            blur: Size used in Gaussian Blur
            threshold: Minial difference between foreground and background
        """
        self.camIndex = camIndex
        self.minAreaRatio = minAreaRatio
        self.border = border
        self.blur = blur
        self.threshold = threshold
        self.cap = cv2.VideoCapture(self.camIndex)
        if not self.cap.isOpened():
            raise SystemExit("Unable to open designated camera.")

    def update_background(self):
        """
        Update background
        """
        while True:
            ok, frame = self.cap.read()
            if not ok:
                raise SystemExit("Unable to capture new frame.")
            tempFrame = frame.copy()
            cv2.putText(tempFrame, "Press ENTER to update background", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow("background"+str(self.camIndex), tempFrame)
            if cv2.waitKey(1) & 0xFF == 13:
                self.base = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.blur, 0)
                cv2.destroyWindow("background"+str(self.camIndex))
                break
    
    
    def detect(self):
        """
        Detect foreground objects.

        Returns:
            (frame, result): 
                - frame: Original frame
                - result: List of (ROI, massCenter, areaSize)
                    - ROI: Bounding box (x, y, w, h)
                    - massCenter: Mass center (x, y) or None
                    - areaSize: Area size or None
        """
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit("Unable to capture new frame.")

        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.blur, 0)
        diff = cv2.absdiff(g, self.base)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)

        h, w = mask.shape
        mask[:self.border,:] = mask[-self.border:,:] = 0
        mask[:,:self.border] = mask[:,-self.border:] = 0

        minArea = int(self.minAreaRatio * w * h)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = []
        for c in cnts:
            if cv2.contourArea(c) < minArea:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            ROI = (x, y, bw, bh)
            roi = mask[y:y+bh, x:x+bw]

            M = cv2.moments(roi, binaryImage=True)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                massCenter = (x+cx, y+cy)
                areaSize = int(M["m00"])
            else:
                massCenter = None
                areaSize = None
            
            result.append((ROI, massCenter, areaSize))    

        return frame, result
    
    def release(self):
        """
        Release camera resource
        """
        self.cap.release()



# Only for testing purpose
if __name__ == "__main__":

    def add_detail(frame, result):
        cv2.putText(frame, "Press B to update background", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, "Press Q to quit", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        for ROI, massCenter, areaSize in result:
            cv2.rectangle(frame, (ROI[0],ROI[1]), (ROI[0] + ROI[2], ROI[1] + ROI[3]), (0,255,0), 2)
            if massCenter != None:
                cv2.circle(frame, massCenter, 5, (0,0,255), -1)
                cv2.putText(frame, str(areaSize), (ROI[0], ROI[1]+ROI[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
    

    cam1 = DetectCamera(1)
    cam2 = DetectCamera(3)

    cam1.update_background()
    cam2.update_background()

    while True:
        frame1, result1 = cam1.detect()
        add_detail(frame1, result1)
        cv2.imshow("camera1", frame1)

        frame2, result2 = cam2.detect()
        add_detail(frame2, result2)
        cv2.imshow("camera2", frame2) 

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('b'):
            cam1.update_background()
            cam2.update_background()

    cam1.release()
    cam2.release()
