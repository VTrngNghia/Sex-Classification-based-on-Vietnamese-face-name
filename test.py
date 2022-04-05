import glob
import os
import random
import sys
from typing import List, Tuple

import cv2
import numpy as np
from keras.models import load_model, Model
from keras.utils.np_utils import to_categorical

from train import IMG_WIDTH, IMG_HEIGHT, Image, Label

this_file, test_data_dir, model_dir = sys.argv

paths: List[str] = []


def main():
    model: Model = load_model(model_dir)
    images, labels = load_data(test_data_dir)
    images = np.array(images)
    labels = to_categorical(labels)
    model.evaluate(images, labels)
    predictions = model.predict(images)
    count = 0
    for index, (prediction, label) in enumerate(zip(predictions, labels)):
        if (prediction[0] > prediction[1]) is not (label[0] > label[1]):
            count += 1
            print(index, prediction, label, paths[index])
    print("Error rate", count / len(images))


def load_data(data_dir: str) -> Tuple[List[Image], List[Label]]:
    images: List[Image] = []
    labels: List[Label] = []
    for sex in ["MALE", "FEMALE"]:
        pattern = os.path.join(data_dir, sex, "*")
        for filepath in glob.glob(f'{pattern}'):
            if sex == "FEMALE" and random.random() > 0.4:
                continue
            img = cv2.imread(filepath)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            paths.append(filepath)
            images.append(img)
            labels.append(0 if sex == "MALE" else 1)
    return images, labels


if __name__ == "__main__":
    main()