import pickle
from sklearn.utils import shuffle
from random import randint
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from parameters_nn import *


def load_data(train_p=TRAIN_PICKLE, valid_p=VALID_PICKLE ,test_p=TEST_PICKLE, verbose=True, visualize=False):
    with open(train_p, 'rb') as f:
        train = pickle.load(f)

    with open(test_p, 'rb') as f:
        test = pickle.load(f)

    with open(valid_p, 'rb') as f:
        valid = pickle.load(f)

    x_train, y_train = train['features'], train['labels']
    x_valid, y_valid = valid['features'], valid['labels']
    x_test, y_test = test['features'], test['labels']


    x_train, y_train = shuffle(x_train, y_train)

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    if verbose:
        print("\nLoaded data")
        print("Image shape: {}".format(x_train[0].shape))
        print("Number of Training samples:   {}".format(len(x_train)))
        print("Number of Test samples:   {}".format(len(x_test)))
        print("Number of classes: {}".format(max(max(y_train), max(y_valid), max(y_test))+1))

    if visualize:
        print("\nVisualizing random training data")
        i = randint(0, len(x_train))
        plt.imshow(x_train[i])
        plt.show()

    return x_train, y_train, x_valid, y_valid, x_test, y_test


if __name__ == "__main__":
    load_data(visualize=True, verbose=True)
