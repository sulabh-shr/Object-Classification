import pickle
from sklearn.utils import shuffle

from parameters_nn import *


def load_data(train_p=TRAIN_PICKLE, test_p=TEST_PICKLE):
    with open(train_p, 'rb') as f:
        train = pickle.load(f)

    with open(test_p, 'rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    X_train, y_train = shuffle(X_train, y_train)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    print("\nImage Shape: {}".format(X_train[0].shape))
    print()
    print("Number of Training samples:   {}".format(len(X_train)))
    print("Number of Test samples:   {}".format(len(X_test)))

    return X_train, y_train, X_test, y_test
