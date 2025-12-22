import cv2


class DetectCamera:

    # Camera-based foreground detection using background subtraction
    def __init__(self, camIndex: int, minAreaRatio=0.003, border=5, blur=(15,15), threshold=32):
        """
        Args:
            camIndex: Index of designated camera
            minAreaRatio: Minimal ratio of ROI area under consideration 
            border: Boundary dead zone
            blur: Size used in Gaussian Blur
            threshold: Minial difference between foreground and background
        """
        # Store configuration parameters
        self.camIndex = camIndex
        self.minAreaRatio = minAreaRatio
        self.border = border
        self.blur = blur
        self.threshold = threshold

        # Open camera device
        self.cap = cv2.VideoCapture(self.camIndex)
        if not self.cap.isOpened():
            raise SystemExit("Unable to open designated camera.")

    def update_background(self):
        """
        Update background
        """
        # Continuously capture frames until background is confirmed by user
        while True:
            ok, frame = self.cap.read()
            if not ok:
                raise SystemExit("Unable to capture new frame.")

            # Rotate frame to match camera mounting orientation
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            tempFrame = frame.copy()

            # Display instruction for background capture
            cv2.putText(tempFrame, "Press ENTER to update background", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow("background"+str(self.camIndex), tempFrame)

            # Save current frame as background when ENTER is pressed
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

        # Rotate frame to match camera mounting orientation
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Compute foreground mask using background subtraction
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.blur, 0)
        diff = cv2.absdiff(g, self.base)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to reduce noise
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)

        # Remove detections near image borders
        h, w = mask.shape
        mask[:self.border,:] = mask[-self.border:,:] = 0
        mask[:,:self.border] = mask[:,-self.border:] = 0

        # Compute minimum contour area threshold
        minArea = int(self.minAreaRatio * w * h)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = []
        for c in cnts:
            # Ignore small contours
            if cv2.contourArea(c) < minArea:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            ROI = (x, y, bw, bh)
            roi = mask[y:y+bh, x:x+bw]

            # Compute mass center and area from image moments
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
        
        # Draw detection results on frame
        self.__add_detail(frame, result)
        return frame, result
    
    def release(self):
        """
        Release camera resource
        """
        # Release camera handle
        self.cap.release()

    def __add_detail(self, frame, result):
        # Draw bounding boxes and mass centers for detected objects
        for ROI, massCenter, areaSize in result:
            cv2.rectangle(frame, (ROI[0],ROI[1]), (ROI[0] + ROI[2], ROI[1] + ROI[3]), (0,255,0), 2)
            if massCenter != None:
                cv2.circle(frame, massCenter, 5, (0,0,255), -1)
