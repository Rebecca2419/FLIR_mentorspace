import cv2
from pair import PairDetecter

# pair1 = PairDetecter(2, 3, "pair1_cood_trans_param.npz", "pair1_ext_param.npz")
pair1 = PairDetecter(3, 1, "pair2_cood_trans_param.npz", "pair2_ext_param.npz")

pair1.update_background()
# pair2.update_background()

while True:
    (frame11, frame12), cood1 = pair1.get_coordinate()
    # (frame21, frame22), cood2 = pair2.get_coordinate()
    cv2.putText(frame11, str(cood1), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    # cv2.putText(frame21, str(cood2), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.imshow("camera11", frame11)
    cv2.imshow("camera12", frame12)
    # cv2.imshow("camera21", frame21)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('b'):
        pair1.update_background()
        # pair2.update_background()

pair1.release()
# pair2.release()