import tensorflow as tf
from load_data import load_data
from parameters_nn import *
from sklearn.utils import shuffle
from architecture import LeNet


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


def train(x_train, y_train, x_valid, y_valid, epochs=EPOCHS, batch_size=BATCH_SIZE, tolerance=TOLERANCE):
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
    prev_acc = 0
    print("\nStarting the training session............")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_train = len(x_train)

        for i in range(epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for offset in range(0, num_train, batch_size):
                end = offset + BATCH_SIZE
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = evaluate(x_valid, y_valid, logits, one_hot_y, x, y)
            print("Epoch: {}".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            if validation_accuracy-prev_acc < tolerance:
                print("Tolerance reached")
                break
            saver.save(sess, 'model/LeNet E' + str(i) + ' ACC {.3f}' + str(float(validation_accuracy)))

             # TODO: Best model in past n
             # TODO: Save only if increased accuracy


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(TRAIN_PICKLE, VALID_PICKLE, TEST_PICKLE)
    train(x_train, y_train, x_valid, y_valid, EPOCHS, BATCH_SIZE, TOLERANCE)
