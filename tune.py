import sys

import numpy as np
from keras import Model
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

import test

this_file, model_dir, data_dir = sys.argv


def main():
    model: Model = load_model(model_dir)
    images, labels = test.load_data(data_dir)
    labels = to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=0.5
    )
    history = model.fit(x_train, y_train, epochs=10, batch_size=128)
    print('Loss:', history.history['loss'])
    print('Accuracy', history.history['accuracy'])

    print('\n\nFinal eval:')
    model.evaluate(x_test, y_test)


if __name__ == "__main__":
    main()