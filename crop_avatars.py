import glob
import logging
import math
import os.path
import sys

import cv2
from retinaface import RetinaFace

this_path, data_dir, cropped_dir = sys.argv


def main():
    count = 0
    for sex in ["MALE", "FEMALE"]:
        pattern = os.path.join(data_dir, sex, "*")
        for source_path in glob.glob(f'{pattern}'):
            try:
                target_path = crop_face(source_path, sex)
                count += 1
                print(count, "Cropped", target_path)
            except Exception:
                print("source_path", source_path)
                logging.exception("message")
    pass


def crop_face(source_path: str, sex: str):
    img = cv2.imread(source_path)
    height, width = img.shape[:2]
    face = RetinaFace.detect_faces(img_path=source_path)
    x1, y1, x2, y2 = face['face_1']['facial_area']

    padded_dimension = "x" if x2 - x1 < y2 - y1 else "y"
    padding = abs((x2 - x1) - (y2 - y1))
    if padded_dimension == "x":
        x1 -= math.floor(padding / 2)
        x2 += math.ceil(padding / 2)
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if x2 > width:
            x1 -= (x2 - width)
            x2 = width
    else:
        y1 -= math.floor(padding / 2)
        x2 += math.ceil(padding / 2)
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if y2 > height:
            y1 -= (y2 - height)
            y2 = height

    dir, filename = os.path.split(source_path)
    target_path = os.path.join(cropped_dir, sex, filename)
    cv2.imwrite(target_path, img[y1: y2, x1:x2])
    return target_path


if __name__ == "__main__":
    main()