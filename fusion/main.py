import cv2
from pair import PairDetecter

pair1 = PairDetecter(1, 2, "pair1_ext_param.npz", (1.0, 1.45))
pair2 = PairDetecter(4, 5, "pair2_ext_param.npz", (1.0, 1.65))

pair1.update_background()
pair2.update_background()

camCood1 = (-660, -3800)
camDire1 = (1, 1)
camCood2 = (1300, 3500)
camDire2 = (-1, -1)

while True:
    (frame11, frame12), cood1 = pair1.get_coordinate()
    (frame21, frame22), cood2 = pair2.get_coordinate()

    roomCood1 = [(camDire1[0] * item[0] + camCood1[0], camDire1[1] * item[1] + camCood1[1]) for item in cood1]
    roomCood2 = [(camDire2[0] * item[0] + camCood2[0], camDire2[1] * item[1] + camCood2[1]) for item in cood2]

    cv2.putText(frame11, str(roomCood1), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(frame21, str(roomCood2), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.imshow("camera11", frame11)
    cv2.imshow("camera21", frame21)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('b'):
        pair1.update_background()
        pair2.update_background()

pair1.release()
pair2.release()