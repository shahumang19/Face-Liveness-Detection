import os
import shutil
import cv2

BASE_DIR = "blink-detection\\dataset_B_Eye_Images\\mrlEyes_2018_01\\mrlEyes_2018_01"
DEST_DIR = "blink-detection\\dataset_B_Eye_Images\\merged_dataset"

dirs = [filename for filename in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, filename))]
print(dirs)

for d in dirs:
    current_dir = os.path.join(BASE_DIR, d)
    print(d)
    for f in os.listdir(current_dir):
        fn = f"{current_dir}\\{f}"
        cls = f.split("_")[4]

        if cls == "0":
            cls = "close"
        else:
            cls = "open"
        df = f"{DEST_DIR}\\{cls}\\{f}"

        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (24,24))
        cv2.imwrite(df, img)
        del img
