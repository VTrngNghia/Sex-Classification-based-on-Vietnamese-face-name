import glob
import os
import sys
from typing import Tuple, List

import cv2
import numpy as np
import tensorflow as tf
from keras import models, layers
from keras.callbacks import Callback
from keras.models import load_model
from sklearn.model_selection import train_test_split

IMG_WIDTH = 50
IMG_HEIGHT = 50

ImageShape = IMG_WIDTH, IMG_HEIGHT, 3
Image = np.ndarray
Label = int


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python train.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=0.2
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25
    )

    # Get a compiled neural network
    # model = get_model()
    model = load_model(sys.argv[2])

    # Fit model on training data
    history = model.fit(
        x_train, y_train, epochs=5, batch_size=256,
        callbacks=[TestCallback((x_val, y_val))]
    )
    print("Loss:", history.history['loss'])
    print("Accuracy:", history.history['accuracy'])

    # Evaluate neural network performance
    print('\n\nFinal eval:')
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nValidation loss: {}, acc: {}\n\n'.format(loss, acc))


def load_data(data_dir: str) -> Tuple[List[Image], List[Label]]:
    """
    Load image data from directory `data_dir`.
    """

    images: List[Image] = []
    labels: List[Label] = []
    count = 0
    for age in os.listdir(data_dir):
        # Images in 111 folders are men, and 112 women
        for sex in ["111", "112"]:
            pattern = os.path.join(data_dir, age, sex, "*")
            for filepath in glob.glob(f'{pattern}'):
                try:
                    img = cv2.imread(filepath)
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    images.append(img)
                    labels.append(0 if sex == "111" else 1)
                    count += 1
                    print(count, "Loaded", filepath)
                except Exception:
                    print("Cannot load file:", filepath, Exception)
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    """
    model = models.Sequential()
    # Input
    model.add(layers.InputLayer(input_shape=ImageShape))

    # Simplified version of VGGNet
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation="relu"))
    model.add(layers.Dense(512, activation="relu"))

    # Output
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("Summary", model.summary())
    return model


if __name__ == "__main__":
    main()