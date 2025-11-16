# Demo for FLIR Detection
## How to run this script

### Hardware setup:

#### In main interface:

- Image mode: Switch to "Thermal".

![WIN_20251116_09_34_38_Pro](https://github.com/user-attachments/assets/85c29a59-6100-436e-86d3-33f88d66a0ac)

- Measurement: Uncheck all options.

![WIN_20251116_09_37_36_Pro](https://github.com/user-attachments/assets/5ebaec3e-1678-4982-b243-9ceb068d87db)

- Color: "White hot" is highly recommended.

![WIN_20251116_09_54_59_Pro](https://github.com/user-attachments/assets/3205122c-15d2-4a3c-a955-60222e914012)

- Temperature scale: **Aim the camera at a scene which contains target and background**, and then switch to "Manual".

![WIN_20251116_10_00_12_Pro](https://github.com/user-attachments/assets/0468560b-6432-46fb-a1eb-4ac436c40b69)

#### In camera settings:

Settings -> Device settings -> Show temperature scale: witch to OFF

![WIN_20251116_10_02_36_Pro](https://github.com/user-attachments/assets/8ccd2f0b-05ef-4bd9-a856-d9ce4e1bf682)


After completing all these settings, the camera image should be as shown in the following figure.

![WIN_20251116_10_05_08_Pro](https://github.com/user-attachments/assets/efcbf1d4-648c-4c3d-b259-5c3b9e8db696)



### Software setup:

In python script, please change the index (1 in this demo) to the actual index of your camera (may be 0, 1, 2, ...).
```C
cam1 = DetectCamera(1)
```

### Run script:

```shell
python test.py
```

In background selection window, aim the camera to the background containing no foreground object, and press ENTER.

<img width="1280" height="960" alt=![background_selection] src="https://github.com/user-attachments/assets/e7523f88-2fef-4840-8bd2-828e19ae0282" />

Then in the detection window, the script will continuously detect foreground objects, and mark their center of mass.

<img width="1280" height="960" alt="image" src="https://github.com/user-attachments/assets/785d8b2b-da52-4256-a47c-9eb82963fb9c" />

