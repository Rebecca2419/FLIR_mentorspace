import cv2
import numpy as np
from mystruct import Detection, singleFrameResult

def load_background(load_path):
    data = np.load(load_path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        # npz 
        return {k: data[k] for k in data.files}
    else:
        # npy 
        return {"mean": data}

def subtract_background(image_path, bg_model, threshold=25, min_area=1000, frame_id=0):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    bg_mean = bg_model["mean"].astype(np.float32)
    if bg_mean.shape != img_rgb.shape:
        bg_mean = cv2.resize(bg_mean, (img_rgb.shape[1], img_rgb.shape[0]))
    diff = np.linalg.norm(img_rgb - bg_mean, axis=2)
    mask = (diff > threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    detections = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y, w, h = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        cx, cy = centroids[i]
        detections.append(
            Detection(id=i, centroid=(int(cx), int(cy)), bbox=(x, y, w, h))
        )

        # visualize
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 255, 0), -1)

    frame_result = singleFrameResult(frame_id=frame_id, detections=detections)
    return mask, frame_result, img

