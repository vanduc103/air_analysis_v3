"""
Stacked AutoEncoder
"""

from read_data import read_data

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def batch_creator(X_set, y_set, batch_size, dataset_length):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length-1, batch_size)
    
    batch_x = X_set[[batch_mask]]
    batch_x = batch_x.reshape(-1, batch_x.shape[1]*batch_x.shape[2])

    batch_y = y_set[[batch_mask]]
    batch_y = batch_y.reshape(-1, batch_y.shape[1]*batch_y.shape[2])
    
    return batch_x, batch_y

# random number
seed = 128
rng = np.random.RandomState(seed)

# argument flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'True: training and testing. False: testing only')
flags.DEFINE_integer('epoch', 0, 'Number of epoch to restore')
flags.DEFINE_integer('num_layer', 1, 'Number of layers to train')
flags.DEFINE_integer('size_layer', 100, 'Size of layers to train')

# read train data
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

### define the layers
num_layer = FLAGS.num_layer
layers = [None] * num_layer
for i in range(num_layer):
    layers[i] = FLAGS.size_layer
input_size = stations * T_X
output_size = stations * T_y

model_path = "model/sae_epoch{}.ckpt"

# define placeholders
x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])

# parameters value
epochs = 100
batch_size = 128
dropout_rate = 0.5
learning_rate = 0.001

### weight initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


### define model
out = [None] * num_layer
W = [None] * num_layer

W[0] = tf.get_variable(name='W0', shape=[input_size, layers[0]], 
		initializer=tf.contrib.layers.xavier_initializer())
b0 = tf.Variable(tf.zeros(layers[0]))
out[0] = tf.nn.relu(tf.add(tf.matmul(x, W[0]), b0))

for i in range(1, num_layer):
    W[i] = tf.get_variable(name='W{}'.format(i), shape=[layers[i-1], layers[i]], 
		initializer=tf.contrib.layers.xavier_initializer())
    bi = tf.Variable(tf.zeros(layers[i]))
    out[i] = tf.nn.relu(tf.add(tf.matmul(out[i-1], W[i]), bi))

# drop out layer
dropout = tf.layers.dropout(
    inputs=out[len(layers)-1], rate=dropout_rate, training=True)

# output layer
W_output = tf.get_variable(name='W_output', shape=[layers[num_layer-1], output_size],   
                initializer=tf.contrib.layers.xavier_initializer())
b_output = tf.Variable(tf.zeros(output_size))
output = tf.add(tf.matmul(dropout, W_output), b_output)

### loss function - absolute_difference
loss = tf.losses.absolute_difference(labels=y, predictions=output)
beta = 0.01
regularizer = 0.0
for i in range(num_layer):
    regularizer += tf.nn.l2_loss(W[i])
regularizer += tf.nn.l2_loss(W_output)
loss = tf.reduce_mean(loss + beta * regularizer)
tf.summary.scalar('sae_loss', loss)

## optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# initialize all variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Merge all the summaries and write them out
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/sae_train', flush_secs=10)
val_writer = tf.summary.FileWriter('log/sae_val', flush_secs=10)

# Training process
train = FLAGS.train
if train:
    print('\nTraining start ...')
    with tf.Session() as sess:
      sess.run(init)

      # Restore model weights from previously saved model
      epoch = FLAGS.epoch
      if epoch > 0:
        saver.restore(sess, model_path.format(epoch))
        print('Checkpoint {} restored!'.format(epoch))
      
      ### for each epoch, do:
        ###   for each batch, do:
        ###     create pre-processed batch
        ###     run optimizer by feeding batch
        ###     find cost and reiterate to minimize
      for epoch in range(epochs):
        total_batch = int(X_train.shape[0]/batch_size)
        for i in range(total_batch):
          batch_x, batch_y = batch_creator(X_train, y_train, batch_size, X_train.shape[0])
          _, train_loss, summary = sess.run([optimizer, loss, merged], feed_dict = {x: batch_x, y: batch_y})
                
        #print('Epoch:{}'.format(epoch+1) + '. Cost = {:.5f}'.format(train_loss))
        train_writer.add_summary(summary, epoch)

        # compute error on validate set
        batch_x, batch_y = batch_creator(X_val, y_val, X_val.shape[0], X_val.shape[0])
        [validate_loss, summary] = sess.run([loss, merged], feed_dict={x: batch_x, y: batch_y})

        #print("Epoch:{}. Validate error: {:.2f}".format(epoch+1, validate_loss))
        val_writer.add_summary(summary, epoch)

        # Save model weights to disk
        if (epoch+1) % 100 == 0:
          save_path = saver.save(sess, model_path.format(epoch+1))
          print("Model saved in file: %s" % save_path)

    print("\nTraining complete!")

# Running testing session
print("---------------------------")
print("Starting testing session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    # Restore model weights from previously saved model
    epoch = FLAGS.epoch
    if epoch == 0:
        epoch = epochs
    saver.restore(sess, model_path.format(epoch))
    print('Checkpoint {} restored!'.format(epoch))

    batch_x, batch_y = batch_creator(X_val, y_val, X_val.shape[0], X_val.shape[0])
    test_loss, pred_out = sess.run([loss, output], feed_dict={x: batch_x, y: batch_y})
    print("Test error: {:.2f}".format(test_loss))

    # Visualizing as image
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title('Prediction')
    X_i = pred_out[0].reshape(stations, T_y)
    #print(X_i)
    ax.imshow(X_i, interpolation='bilinear')

    ax = fig.add_subplot(122)
    ax.set_title('Actual')
    X_i1 = batch_y[0].reshape(stations, T_y)
    #print(X_i1)
    ax.imshow(X_i1, interpolation='bilinear')

    plt.show()

