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
    #batch_x = batch_x[..., np.newaxis]

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

# read train data
stations = 25
T_X = 48 # number of hours to be the historical hours
T_y = 6 # number of hours to be the forecasting hours
dataset = read_data(T_X, T_y)
dataset.read_file()
dataset.split_data()
X, y = dataset.X, dataset.y
X = dataset.convert2D_to_3D()
print(X.shape)

# train set = year 2015 + 2016 => 731 days
# val set = year 2017 => 365 days
train_size = 731*24/6
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]
print(X_train.shape)
print(X_val.shape)
print('Reading training data done !')

### define the layers
image_height = stations
image_width = T_X/3

output_T = T_y
output_size = stations*output_T

filters = [128, 64, 64]
kernels = [5, 3, 1]
fc_size = output_size

model_path = "model/cnn_new_epoch{}.ckpt"

# define placeholders
x = tf.placeholder(tf.float32, [None, image_height, image_width, 3])
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
# convolution-pooling layer define
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# convolution-pooling layer #1
W_conv1 = weight_variable([kernels[0], kernels[0], 3, filters[0]])
b_conv1 = bias_variable([filters[0]])
conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
pool1 = max_pool_2x2(conv1)

# convolution-pooling layer #2
W_conv2 = weight_variable([kernels[1], kernels[1], filters[0], filters[1]])
b_conv2 = bias_variable([filters[1]])
conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)
pool2 = max_pool_2x2(conv2)

# convolution-pooling layer #3
W_conv3 = weight_variable([kernels[2], kernels[2], filters[1], filters[2]])
b_conv3 = bias_variable([filters[2]])
conv3 = tf.nn.relu(conv2d(pool2, W_conv3) + b_conv3)
pool3 = max_pool_2x2(conv3)

# fully connected
flatten = tf.contrib.layers.flatten(pool3)
flatten_dim = flatten.get_shape()[1].value
W_fc = tf.get_variable(name='W_fc', shape=[flatten_dim, fc_size], 
		initializer=tf.contrib.layers.xavier_initializer())
b_fc = tf.Variable(tf.zeros(fc_size))
fc = tf.nn.relu(tf.add(tf.matmul(flatten, W_fc), b_fc))

# drop out layer
dropout = tf.layers.dropout(
    inputs=fc, rate=dropout_rate, training=True)

# output layer
W_output = tf.get_variable(name='W_output', shape=[fc_size, output_size],   
                initializer=tf.contrib.layers.xavier_initializer())
b_output = tf.Variable(tf.zeros(output_size))
output = tf.add(tf.matmul(dropout, W_output), b_output)

### loss function - absolute_difference
loss = tf.reduce_mean(tf.abs(tf.subtract(y, output)))
beta = 0.01
regularizer = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_fc) + tf.nn.l2_loss(W_output)
loss = tf.reduce_mean(loss + beta * regularizer)
tf.summary.scalar('cnn_new_loss', loss)

## optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# initialize all variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Merge all the summaries and write them out
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/new_train', flush_secs=10)
val_writer = tf.summary.FileWriter('log/new_val', flush_secs=10)

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
    X_i = pred_out[0].reshape(stations, output_T)
    #print(X_i)
    ax.imshow(X_i, interpolation='bilinear')

    ax = fig.add_subplot(122)
    ax.set_title('Actual')
    X_i1 = batch_y[0].reshape(image_height, output_T)
    #print(X_i1)
    ax.imshow(X_i1, interpolation='bilinear')

    plt.show()

