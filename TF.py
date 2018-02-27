from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

tf.set_random_seed(1234)

FLAGS = None

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  #mnist = input_data.read_data_sets( "MNIST_data/", one_hot=True)

  # Create the model
  
  n = 200
  x = tf.placeholder(tf.float32, [None, 784])
  W1=tf.Variable(tf.truncated_normal([784, n], stddev=0.1))
  b1=tf.Variable(tf.constant(0.1, shape=[n]))
  W2 = tf.Variable(tf.truncated_normal([n, 10], stddev=0.1))
  b2 = tf.Variable(tf.constant(0.1, shape=[10]))
  z = tf.maximum(0.,tf.matmul(x, W1) + b1)

  y= tf.matmul(z,W2)+b2 #output layer: sigmoid

  # Define loss and optimizer
  
  learning_rate = tf.placeholder(tf.float32, shape= [])
  y_ = tf.placeholder(tf.float32, [None, 10])
  
  cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
  cross_entropy_sigmoid = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
  
  train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
  train_step_sigmoid = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy_sigmoid)
  



  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Train
  for _ in range(5000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: 0.5})

  # Test trained model
  pred = tf.nn.softmax(tf.matmul(z,W2)+b2 )
  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

  # Train
  for _ in range(5000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step_sigmoid, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: 0.5})

  # Test trained model
  pred = tf.nn.softmax(tf.matmul(z,W2)+b2 )
  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


#########learning rate 0.05


  # Train
  #  for _ in range(5000):
  #    batch_xs, batch_ys = mnist.train.next_batch(100)
#    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: 0.05})

  # Test trained model
# pred = tf.nn.softmax(tf.matmul(z,W2)+b2 )
#  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
#  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
#                                      y_: mnist.test.labels}))

#############learning rate 0.005
  # Train
  #for _ in range(5000):
  #    batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: 0.005})

  # Test trained model
#  pred = tf.nn.softmax(tf.matmul(z,W2)+b2 )
#  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
#  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
#                                     y_: mnist.test.labels}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  
  #chagnge hidden size
  n = 200
  #n = 2000  #n=20
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)




