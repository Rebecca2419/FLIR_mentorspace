import cv2
import json
from build_bg import build_background, save_background
from subtract_bg import load_background, subtract_background
import os
from glob import glob

# obtain root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# name & paths
PIC_NAME = "WIN_20251029_13_54_08_Pro"

INPUT_PATH = os.path.join(ROOT_DIR, "back&for-ground/kitchen_corner")

BGIMG_PATH = os.path.join(ROOT_DIR, "back&for-ground/kitchen_corner/background")
BG_PATH = os.path.join(ROOT_DIR, "11_11_bg_model.npz")


# === build background ===
bg = build_background(BGIMG_PATH)
save_background(bg, BG_PATH)

# === load bg & subtract ===
bg_model = load_background(BG_PATH)

input_images = sorted(
    glob(os.path.join(INPUT_PATH, "*.jpg")) +
    glob(os.path.join(INPUT_PATH, "*.png")) +
    glob(os.path.join(INPUT_PATH, "*.jpeg"))
)
for i, img_path in enumerate(input_images):

    mask, frame_result, vis_img = subtract_background(img_path, bg_model,
                                                  threshold=230,
                                                  min_area=1000,
                                                  frame_id=i)
    pic_name = os.path.splitext(os.path.basename(img_path))[0]
    OUTPUT_MASK_PATH = os.path.join(ROOT_DIR, "data", "output", f"{pic_name}_mask.jpg")
    OUTPUT_VIS_PATH = os.path.join(ROOT_DIR, "data", "output", f"{pic_name}_vis.jpg")
    # OUTPUT_JSON_PATH = os.path.join(ROOT_DIR, "data", "output", f"{PIC_NAME}_detections.json")

    # === output mask & vis ===
    cv2.imwrite(OUTPUT_MASK_PATH, mask)
    cv2.imwrite(OUTPUT_VIS_PATH, vis_img)

    print(frame_result.to_dict())

    print("Detection complete;")
    print(f"Mask saved to: {OUTPUT_MASK_PATH}")
    print(f"Visualization saved to: {OUTPUT_VIS_PATH}")
    # print(f"JSON saved to: {OUTPUT_JSON_PATH}")
    


