import cv2

PIC_NAME = "FLIR0029"
INPUT_PATH = "../data/" + PIC_NAME + ".jpg"
OUTPUT_PATH = "../data/output/bgsb" + PIC_NAME + ".jpg"
BGIMG_PATH = "../fake_bg.jpg"
BG_PATH = "fake_bg_model.npz"

# bg model creation
from fake_bg import build_background, save_background
bg = build_background([BGIMG_PATH])
save_background(bg, BG_PATH)

# foreground detection
from subtract_bg import load_background, subtract_background
bg_model = load_background(BG_PATH)
mask = subtract_background(INPUT_PATH, bg_model, threshold=100)

cv2.imwrite(OUTPUT_PATH, mask)
