import tensorflow as tf
from load_data import load_data
from parameters_nn import *
from sklearn.utils import shuffle
from architecture import LeNet
from time import time
from matplotlib import pyplot as plt
import os

def evaluate(x_data, y_data, logits, one_hot_y, x, y):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = x_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def train(x_train, y_train, x_valid, y_valid, epochs=EPOCHS, batch_size=BATCH_SIZE, tolerance=TOLERANCE,
          save_best=True):
    print("\nRUNNING TRAINING SETUP")

    # Input image where None is used to allow batch of any size
    x = tf.placeholder(tf.float32, (None, 32, 32, IN_DEPTH))

    # Integer labels
    y = tf.placeholder(tf.int32, (None))
    # One hot encoding the y labels
    one_hot_y = tf.one_hot(y, LABELS)

    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    # Mean over cross entropy from all input images
    loss_operation = tf.reduce_mean(cross_entropy)
    # Adam similar to SGD
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE)
    # Minimize the optimization value
    training_operation = optimizer.minimize(loss_operation)

    saver = tf.train.Saver()
    prev_best_accuracy = 0
    print("Starting the Training Session")
    start_time = time()
    epochs_trained = 1      # minimum 1 epoch. Removes division by zero error
    loss = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_train = len(x_train)

        for i in range(epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for offset in range(0, num_train, batch_size):
                end = offset + BATCH_SIZE
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            # Calculating accuracy on validation dataset
            validation_accuracy = evaluate(x_valid, y_valid, logits, one_hot_y, x, y)

            # Adding loss to list for plotting
            loss.append(1-validation_accuracy)
            epochs_trained = i + 1

            print("Epoch: {}".format(i + 1))
            print("Validation Accuracy = {:.2f} %".format(validation_accuracy*100))

            # Checking model improvement
            if validation_accuracy - prev_best_accuracy < -tolerance:
                print("\nModel Accuracy degraded below Tolerance level......")
                print("\nBest Model Accuracy = {:.2f} %".format(prev_best_accuracy*100))
                break

            # Checking if current accuracy is better than best recorded previously
            if validation_accuracy > prev_best_accuracy:
                prev_best_accuracy = validation_accuracy

                # Save if better accuracy when save_best True and continue to next iteration
                if save_best:
                    saver.save(sess, "models/LeNet E" + str(i) + " ACC {:.4f}".format(validation_accuracy))
                    continue

            # Save the model
            saver.save(sess, "models/LeNet E" + str(i) + " ACC {:.4f}".format(validation_accuracy))

    # Calculating total time
    total_time = time() - start_time
    print("Training Time: {:.2f} sec".format(total_time))
    print("Training Time per Epoch: {:.2f} sec".format(total_time/epochs_trained))

    os.system('spd-say -p +35 -r -30 "The training has finished. Thank you"')
    plt.plot(list(range(1, len(loss)+1)), loss, 'ro', linestyle='dashed', dash_joinstyle='round')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.axis([0, epochs_trained, 0, 1])
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(TRAIN_PICKLE, VALID_PICKLE, TEST_PICKLE)
    train(x_train, y_train, x_valid, y_valid, EPOCHS, BATCH_SIZE, TOLERANCE)
