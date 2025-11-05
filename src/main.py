import cv2
import json
from fake_bg import build_background, save_background
from subtract_bg import load_background, subtract_background
import os

# obtain root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# name & paths
PIC_NAME = "WIN_20251029_13_53_32_Pro"

INPUT_PATH = os.path.join(ROOT_DIR, "pictures for detection", PIC_NAME + ".jpg")
OUTPUT_MASK_PATH = os.path.join(ROOT_DIR, "data", "output", f"bgsb_{PIC_NAME}_mask.jpg")
OUTPUT_VIS_PATH = os.path.join(ROOT_DIR, "data", "output", f"bgsb_{PIC_NAME}_vis.jpg")
OUTPUT_JSON_PATH = os.path.join(ROOT_DIR, "data", "output", f"bgsb_{PIC_NAME}_detections.json")

BGIMG_PATH = os.path.join(ROOT_DIR, "fake_bg.jpg")
BG_PATH = os.path.join(ROOT_DIR, "fake_bg_model.npz")


# === build background ===
bg = build_background([BGIMG_PATH])
save_background(bg, BG_PATH)

# === load bg & subtract ===
bg_model = load_background(BG_PATH)
mask, frame_result, vis_img = subtract_background(INPUT_PATH, bg_model,
                                                  threshold=100,
                                                  min_area=1500,
                                                  frame_id=1)

# === output mask & vis ===
cv2.imwrite(OUTPUT_MASK_PATH, mask)
cv2.imwrite(OUTPUT_VIS_PATH, vis_img)

print(frame_result.to_dict())

print("Detection complete;")
print(f"Mask saved to: {OUTPUT_MASK_PATH}")
print(f"Visualization saved to: {OUTPUT_VIS_PATH}")
print(f"JSON saved to: {OUTPUT_JSON_PATH}")
