import cv2, numpy as np

cam = 1
min_area_ratio = 0.003
border = 12
blur = (15,15)
thershold = 32

cap = cv2.VideoCapture(cam)
if not cap.isOpened():
    raise SystemExit("Unable to open designated camera.")

base = None
print("Press 'b' to update background.")
while True:
    _, frame = cap.read()
    cv2.imshow("background", frame)
    if cv2.waitKey(1) & 0xFF == ord('b'):
        base = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), blur, 0)
        cv2.destroyWindow("background")
        break

print("Press 'q' to quit; press 'b' to update background.")
while True:
    ok, frame = cap.read()
    if not ok: break

    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), blur, 0)
    diff = cv2.absdiff(g, base)
    _, mask = cv2.threshold(diff, thershold, 255, cv2.THRESH_BINARY)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)

    h, w = mask.shape
    mask[:border,:] = mask[-border:,:] = 0
    mask[:,:border] = mask[:,-border:] = 0

    min_area = int(min_area_ratio * w * h)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < min_area: continue
        x,y,bw,bh = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)

    cv2.imshow("Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('b'):
        base = g.copy()

cap.release()
cv2.destroyAllWindows()
