import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    #temp_batch = unclean_batch_x / unclean_batch_x.max()
    temp_batch = unclean_batch_x / 150.0
    
    return temp_batch

def batch_creator(X_set, y_set, batch_size, dataset_length):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length-1, batch_size)
    
    batch_x = X_set[[batch_mask]]
    batch_x = batch_x[..., np.newaxis]

    batch_y = y_set[[batch_mask]]
    batch_y = batch_y.reshape(-1, batch_y.shape[1]*batch_y.shape[2])
    
    return batch_x, batch_y

def read_data(T_X, T_y):
    # Number of stations in Seoul
    all_stations = 33

    # Number of used stations in Seoul data
    stations = 25

    # read pm2_5 values
    all_data = list()
    X = list()
    idx = 0
    for line in open('data/aqi2_5_seoul_stations_join_2015_2017'):
        idx += 1 
        if idx == all_stations:
            idx = 0 # reset
        elif idx <= stations:
            values = line.split(",")
            row = list()
            for i in range(len(values)):
                row.append(float(values[i]))
            X.append(row)
            # Process a batch of stations
            if idx == stations:
                df = pd.DataFrame(data=X)
                df = df.replace(-1.0, np.NaN)
                df = df.fillna(df.mean())
                df = df.replace(np.NaN, 1.0) # in case of fillna fall
                df = df.reindex([9,8,2,16,22,23,24,10,15,5,1,17,21,13,12,3,20,19,18,11,6,4,14,0,7])
                val = df.values
                all_data.append(val.transpose())
                X = list()

    all_data = np.array(all_data)
    all_data = all_data.reshape(all_data.shape[0]*all_data.shape[1], -1).transpose()

    X = list()
    y = list()
    for i in range(0, all_data.shape[1]-T_X-T_y, 6):
        X.append(all_data[:, i:i+T_X])
        y.append(all_data[:, i+T_X:i+T_X+T_y])
    X = np.array(X)
    print(X.shape)

    y = np.array(y)
    # flip y
    for i in range(len(y)):
        tmp = np.zeros((y[i].shape[0], y[i].shape[1]))
        tmp[:] = y[i][:]
        for j in range(T_y):
            y[i][:,j] = tmp[:,T_y-1-j]
    y = np.abs(y - X)
    print(y.shape)

    return X, y

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
T_X = 24 # number of hours to be the historical hours
T_y = 24 # number of hours to be the forecasting hours
X, y = read_data(T_X, T_y)

# resize X
'''from skimage.transform import rescale, resize
X_new = [None] * X.shape[0]
for i in range(X.shape[0]):
    X_new[i] = resize(X[i], (X[i].shape[0]*4, X[i].shape[1]*2))
X = np.array(X_new)'''

# train set = year 2015 + 2016 => 731 days
# val set = year 2017 => 365 days
train_size = 731*24/6
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]
print(X_train.shape)
print(X_val.shape)
print('Reading training data done !')

### define the layers
image_height = X.shape[1]
image_width = X.shape[2]

output_T = T_y
output_size = stations*output_T

filters = [64, 128, 128, 256, 256]
kernels = [3, 3, 3, 3, 3, 3]
fc_size = output_size

model_path = "model/cnn_epoch{}.ckpt"

# define placeholders
x = tf.placeholder(tf.float32, [None, image_height, image_width, 1])
y = tf.placeholder(tf.float32, [None, output_size])

# parameters value
epochs = 300
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
W_conv = [None] * len(filters)
b_conv = [None] * len(filters)
conv = [None] * len(filters)
pool = [None] * len(filters)

W_conv[0] = weight_variable([kernels[0], kernels[0], 1, filters[0]])
b_conv[0] = bias_variable([filters[0]])
conv[0] = tf.nn.relu(conv2d(x, W_conv[0]) + b_conv[0])
pool[0] = max_pool_2x2(conv[0])

# convolution-pooling layer #
for i in range(1, len(filters)):
    W_conv[i] = weight_variable([kernels[i], kernels[i], filters[i-1], filters[i]])
    b_conv[i] = bias_variable([filters[i]])
    conv[i] = tf.nn.relu(conv2d(pool[i-1], W_conv[i]) + b_conv[i])
    pool[i] = max_pool_2x2(conv[i])

# fully connected
flatten = tf.contrib.layers.flatten(pool[len(filters)-1])
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
#loss = tf.losses.absolute_difference(labels=y, predictions=output)
loss = tf.reduce_mean(tf.abs(tf.subtract(y, output)))
beta = 0.01
regularizer = tf.nn.l2_loss(W_fc) + tf.nn.l2_loss(W_output)
loss = tf.reduce_mean(loss + beta * regularizer)
tf.summary.scalar('cnn_loss', loss)

## optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# initialize all variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Merge all the summaries and write them out
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/train', flush_secs=10)
val_writer = tf.summary.FileWriter('log/val', flush_secs=10)

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
        batch_x, batch_y = batch_creator(X_val, y_val, batch_size, X_val.shape[0])
        [validate_loss, summary] = sess.run([loss, merged], feed_dict={x: batch_x, y: batch_y})

        print("Epoch:{}. Validate error: {:.2f}".format(epoch+1, validate_loss))
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
    ax1 = fig.add_subplot(121)
    ax1.set_title('Prediction')
    X_i = pred_out[0].reshape(stations, output_T)
    #print(X_i)
    ax1.imshow(X_i, interpolation='bilinear')

    ax2 = fig.add_subplot(122)
    ax2.set_title('Actual')
    X_i = batch_y[0].reshape(stations, output_T)
    #print(X_i1)
    ax2.imshow(X_i, interpolation='bilinear')

    '''ax3 = fig.add_subplot(332)
    ax3.set_title('X')
    X_i = batch_x[1,:,:,0]
    #print(X_i1)
    ax3.imshow(X_i, interpolation='bilinear')'''

    plt.show()

