import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import sys


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict


class CifarLearn:
    def __init__(self):
        self.data = []
        self.labels = []
        for i in range(1, 5):
            batch = unpickle('cifar-10-batches-py/data_batch_%d' % i)
            self.data.extend(batch[b'data'])
            labels = []
            for l in batch[b'labels']:
                label = [0] * 10
                label[l] = 1
                labels.append(label)
            self.labels.extend(labels)

    def next_batch(self, size, iteration):
        start = (iteration % (len(self.data) // size)) * size
        stop = start + size
        return self.data[start:stop], self.labels[start:stop]


class CifarTest:
    def __init__(self):
        self.data = []
        self.labels = []
        batch = unpickle('cifar-10-batches-py/test_batch')
        self.data.extend(batch[b'data'])
        labels = []
        for l in batch[b'labels']:
            label = [0] * 10
            label[l] = 1
            labels.append(label)
        self.labels.extend(labels)

    def next_batch(self, size, iteration):
        start = (iteration % (len(self.data) // size)) * size
        stop = start + size
        return self.data[start:stop], self.labels[start:stop]


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main(_):
    cifar = CifarLearn()

    x = tf.placeholder(tf.float32, [None, 32*32*3])
    W = tf.Variable(tf.zeros([32*32*3, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


    test = CifarTest()

    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 32, 32, 3])


    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([64 * 5 * 5, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 64 * 5 * 5])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        batch = cifar.next_batch(1000, i)
        if i % 1 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: test.data, y_: test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
