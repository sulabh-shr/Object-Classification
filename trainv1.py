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


def train(x_train, y_train, x_valid, y_valid, epochs, batch_size, tolerance=0.015,
          patience=0, save_best=True):
    print("\nRUNNING TRAINING SETUP")

    x = tf.placeholder(tf.float32, (None, 32, 32, IN_DEPTH))        # Input image. None is used to allow any batch size
    y = tf.placeholder(tf.int32, None)                              # Integer labels
    one_hot_y = tf.one_hot(y, LABELS)                               # One hot encoding the y labels
    logits = LeNet(x)                                               # Output of LeNet

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)                  # Mean over cross entropy from all input images
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE)    # Adam similar to SGD
    training_operation = optimizer.minimize(loss_operation)         # Minimize the optimization value

    saver = tf.train.Saver()        # Initializing saver for later saving model
    prev_best_accuracy = 0          # Holding the value of best yet accuracy
    start_time = time()             # Time at this moment
    epochs_trained = 1              # Minimum 1 epoch required to remove division by zero error
    loss = []                       # Loss data for plotting
    patience_count = 0              # For patience

    # Starting the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_train = len(x_train)

        # Iterating over each Epoch
        for i in range(epochs):
            x_train, y_train = shuffle(x_train, y_train)

            # Iterating over the data in Batch size
            for offset in range(0, num_train, batch_size):
                end = offset + BATCH_SIZE
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            # Calculating accuracy on validation dataset
            validation_accuracy = evaluate(x_valid, y_valid, logits, one_hot_y, x, y)

            loss.append(1-validation_accuracy)          # Adding loss to list for plotting
            epochs_trained = i + 1                      # Used for calculating time per epoch

            print("Epoch: {}".format(i + 1))
            print("Validation Accuracy = {:.2f} %".format(validation_accuracy*100))

            # Checking model improvement
            if validation_accuracy - prev_best_accuracy < -tolerance and patience_count == patience:
                print("\nModel Accuracy degraded below Tolerance level......")
                print("\nBest Model Accuracy = {:.2f} %".format(prev_best_accuracy*100))
                break

            # Checking if current accuracy is better than best recorded previously
            if validation_accuracy > prev_best_accuracy:
                prev_best_accuracy = validation_accuracy
                patience_count = 0

                # Save if better accuracy when save_best True and continue to next iteration
                if save_best:
                    saver.save(sess, "models/LeNet E" + str(i) + " ACC {:.4f}".format(validation_accuracy))
                    continue

            patience_count += 1
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
    train(x_train, y_train, x_valid, y_valid, EPOCHS, BATCH_SIZE, TOLERANCE, PATIENCE)
