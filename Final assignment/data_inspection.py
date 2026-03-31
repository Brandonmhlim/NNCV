import os
import numpy as np
from PIL import Image

gt_root = "/home/scur2235/NNCV/Final assignment/data/cityscapes/gtFine/train"

id_to_trainid = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}

class_counts = np.zeros(19, dtype=np.int64)
total_valid_pixels = 0

city_folders = os.listdir(gt_root)
total_cities = len(city_folders)

for city_idx, city_name in enumerate(city_folders, start=1):
    city_path = os.path.join(gt_root, city_name)

    image_files = os.listdir(city_path)
    total_images = len(image_files)

    print(f"[{city_idx}/{total_cities}] Processing city folder: {city_name}")

    for image_idx, file_name in enumerate(image_files, start=1):
        if not file_name.endswith("_gtFine_labelIds.png"):
            continue

        print(f"    [{image_idx}/{total_images}] {file_name}")

        file_path = os.path.join(city_path, file_name)
        label_np = np.array(Image.open(file_path)).flatten()

        for pixel_value in label_np:
            if pixel_value in id_to_trainid:
                train_id = id_to_trainid[pixel_value]
                class_counts[train_id] += 1
                total_valid_pixels += 1

class_percentages = class_counts / total_valid_pixels

print(class_percentages)