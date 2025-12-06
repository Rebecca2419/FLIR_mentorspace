import cv2
from pair import PairDetecter

pair1 = PairDetecter(3, 4, "pair1_ext_param.npz", (1.0, 1.45))
pair2 = PairDetecter(1, 2, "pair2_ext_param.npz", (1.0, 1.65))

pair1.update_background()
pair2.update_background()

camCood1 = (-660, -3800)
camDire1 = (1, 1)
camCood2 = (1300, 3500)
camDire2 = (-1, -1)

maxCood = 4500
fusionThreshold = 1000

while True:
    (frame11, frame12), cood1 = pair1.get_coordinate()
    (frame21, frame22), cood2 = pair2.get_coordinate()

    roomCood1 = [(camDire1[0] * item[0] + camCood1[0], camDire1[1] * item[1] + camCood1[1]) for item in cood1]
    roomCood2 = [(camDire2[0] * item[0] + camCood2[0], camDire2[1] * item[1] + camCood2[1]) for item in cood2]

    # roomCood1 = [item for item in roomCood1 if abs(item[0]) <= maxCood and abs(item[1]) <= maxCood]
    # roomCood1 = [item for item in roomCood2 if abs(item[0]) <= maxCood and abs(item[1]) <= maxCood]

    cv2.putText(frame11, str(roomCood1), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(frame21, str(roomCood2), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    mergedCood = roomCood1
    for point in roomCood2:
        isDuplicate = False
        for existingPoint in mergedCood:
            distance = ((point[0] - existingPoint[0])**2 + (point[1] - existingPoint[1])**2)**0.5
            if distance < fusionThreshold:
                isDuplicate = True
                break
        if not isDuplicate:
            mergedCood.append(point)
    
    cv2.putText(frame22, str(mergedCood), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.imshow("camera11", frame11)
    cv2.imshow("camera12", frame12)
    cv2.imshow("camera21", frame21)
    cv2.imshow("camera22", frame22)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('b'):
        pair1.update_background()
        pair2.update_background()

pair1.release()
pair2.release()