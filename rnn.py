from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import random

from read_data import read_data

# random number
seed = 128
rng = np.random.RandomState(seed)

def batch_creator(X_set, y_set, batch_size, dataset_length):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length-1, batch_size)
    
    batch_x = X_set[[batch_mask]]
    batch_x = batch_x.transpose(0, 2, 1)
    #batch_x = batch_x.reshape(-1, batch_x.shape[1]*batch_x.shape[2])

    batch_y = y_set[[batch_mask]]
    batch_y = batch_y.reshape(-1, batch_y.shape[1]*batch_y.shape[2])
    
    return batch_x, batch_y

def model(x, weights, biases):

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, timesteps, 1)
    x = tf.reshape(x, [-1, n_input*stations])
    x = tf.split(x, n_input, 1)

    # 1-layer RNN with n_hidden units.
    rnn_cell = rnn.BasicRNNCell(n_hidden)

    # dropout
    rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=0.3)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # output
    output = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return output

# argument flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'True: training and testing. False: testing only')
flags.DEFINE_integer('epoch', 0, 'Number of epoch to restore')

# training data
stations = 25
T_X = 48 # number of hours to be the historical hours
T_y = 6 # number of hours to be the forecasting hours

dataset = read_data(T_X, T_y)
dataset.read_file()
dataset.split_data()
X, y = dataset.X, dataset.y

# train set = year 2015 + 2016 => 731 days
# val set = year 2017 => 365 days
train_size = 731*24/6
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]
print(X_train.shape)
print(X_val.shape)
print('Reading training data done !')

# Training Parameters
learning_rate = 0.01
training_steps = 3000
batch_size = 128
display_step = 100
val_step = 100

model_path = "model/rnn_epoch{}.ckpt"

# Network Parameters
n_input = T_X # timesteps
n_output = T_y*stations # predict timesteps
n_hidden = 1000 # hidden layer num of features

# tf Graph input
x = tf.placeholder("float", [None, n_input, stations])
y = tf.placeholder("float", [None, n_output])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Prepare for training
pred = model(x, weights, biases)

# Loss and optimizer
cost = tf.losses.absolute_difference(labels=y, predictions=pred)
tf.summary.scalar('rnn_loss', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Merge all the summaries and write them out
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/train_rnn', flush_secs=10)
val_writer = tf.summary.FileWriter('log/val_rnn', flush_secs=10)

# Start training
if FLAGS.train:
    print('\nTraining start ...')
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        for step in range(training_steps):
            # Make the training batch for each step
            batch_x, batch_y = batch_creator(X_train, y_train, batch_size, X_train.shape[0])

            # Run optimization
            _, loss, summary = sess.run([optimizer, cost, merged], feed_dict={x: batch_x, y: batch_y})
            train_writer.add_summary(summary, step)

            # Validation
            if step % val_step == 0:
                #print("Step: {}".format(step+1) + ". Error: {:.6f}".format(loss))

                # compute MSE on validate set
                batch_x, batch_y = batch_creator(X_val, y_val, X_val.shape[0] - 1, X_val.shape[0])
                [validate_loss, summary] = sess.run([cost, merged], feed_dict={x: batch_x, y: batch_y})
                val_writer.add_summary(summary, step)

                #print("Step: {}. Validate Error: {:.2f}".format(step+1, validate_loss))

            if (step+1) % 1000 == 0 and step > 0:
                # Save model weights to disk
                save_path = saver.save(sess, model_path.format(step+1))
                print("Model saved in file: %s" % save_path)

    print("\nTraining completed!")

# Running testing session
print("---------------------------")
print("Starting testing session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    # Restore model weights from previously saved model
    epoch = FLAGS.epoch
    if epoch == 0:
        epoch = training_steps
    saver.restore(sess, model_path.format(epoch))
    print('Checkpoint {} restored!'.format(epoch))

    batch_x, batch_y = batch_creator(X_val, y_val, 1, X_val.shape[0])
    test_loss, pred_out = sess.run([cost, pred], feed_dict={x: batch_x, y: batch_y})
    print("Test error: {:.2f}".format(test_loss))
    
    # Visualizing as image
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title('Prediction')
    X_i = pred_out.reshape(stations, T_y)
    print(X_i)
    ax.imshow(X_i, interpolation='bilinear')

    ax = fig.add_subplot(122)
    ax.set_title('Actual')
    X_i1 = batch_y[0].reshape(stations, T_y)
    print(X_i1)
    ax.imshow(X_i1, interpolation='bilinear')

    plt.show()

