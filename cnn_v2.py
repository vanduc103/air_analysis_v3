import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    #temp_batch = unclean_batch_x / unclean_batch_x.max()
    temp_batch = unclean_batch_x / 150.0
    
    return temp_batch

def batch_creator(dataset, batch_size, dataset_length, output_size):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length-1, batch_size)
    
    batch_x = dataset[[batch_mask]]
    batch_x = batch_x[..., np.newaxis]

    batch_y = dataset[[batch_mask+1]]
    batch_y = batch_y.reshape(-1, batch_y.shape[1]*batch_y.shape[2])
    
    return batch_x, batch_y

def read_data(T=24):
    # Number of stations in Seoul
    all_stations = 33

    # Number of used stations in Seoul data
    stations = 25

    # read pm2_5 values
    all_data = list()
    X = list()
    idx = 0
    for line in open('data/aqi2_5_seoul_stations_2015_2018_03_18.csv'):
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
                idx = 0
                df = pd.DataFrame(data=X)
                df = df.replace(-1.0, np.NaN)
                df = df.fillna(df.mean())
                df = df.replace(np.NaN, 1.0) # in case of fillna fall
                df = df.reindex([9,8,2,16,22,23,24,10,15,5,1,17,21,13,12,3,20,19,18,11,6,4,14,0,7])
                val = df.values
                all_data.append(val.transpose())
                X = list()

    all_data = np.array(all_data)

    T = 24 # number of hours to be the row size of image
    X = all_data.reshape(-1, T, stations).transpose(0, 2, 1)

    return X

# random number
seed = 128
rng = np.random.RandomState(seed)

# Constants describing the training process.
MAX_EPOCHS = 10000
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1 # Initial learning rate.

# read train data
stations = 25
T = 24
X = read_data()
print(X.shape)

# split to train set and validate set
split_size = int(X.shape[0]*0.8)
X_train, X_val = X[:split_size], X[split_size:]
print('Reading training data done !')

### define the layers
image_height = stations
image_width = T

output_T = T
output_size = stations*output_T

filter_num1 = 64
filter_num2 = 64
filter_size = 3
fc_size = output_size
dropout_rate = 0.5

model_path = "model/cnn_epoch{}.ckpt"

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

# define placeholders
x = tf.placeholder(tf.float32, [None, image_height, image_width, 1])
y = tf.placeholder(tf.float32, [None, output_size])

def model(images):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[filter_size, filter_size, 1, filter_num1],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [filter_num1], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME', name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[filter_size, filter_size, filter_num1, filter_num2],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [filter_num2], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        flatten = tf.contrib.layers.flatten(pool2)
        dim = flatten.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, fc_size],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [fc_size], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(flatten, weights) + biases, name=scope.name)

    # output
    with tf.variable_scope('output') as scope:
        weights = _variable_with_weight_decay('weights', [fc_size, output_size],
                                              stddev=1/192.0, wd=None)
        biases = _variable_on_cpu('biases', [output_size],
                                  tf.constant_initializer(0.0))
        output = tf.add(tf.matmul(local3, weights), biases, name=scope.name)

    return output

# Variables that affect learning rate.
batch_size = FLAGS.batch_size
total_batch = int(X_train.shape[0]/batch_size)
decay_steps = int(total_batch * NUM_EPOCHS_PER_DECAY)

# Decay the learning rate exponentially based on the number of steps.
global_step = tf.train.get_or_create_global_step()
lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                              global_step,
                              decay_steps,
                              LEARNING_RATE_DECAY_FACTOR,
                              staircase=True)
tf.summary.scalar('learning_rate', lr)

### loss function - mean squared error
output = model(x)
loss = tf.losses.mean_squared_error(labels=y, predictions=output)
tf.summary.scalar('cnn_loss', loss)

## optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# initialize all variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Merge all the summaries and write them out
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/train', flush_secs=10)
val_writer = tf.summary.FileWriter('log/val', flush_secs=10)

# Training process
print('\nTraining start ...')
with tf.Session() as sess:
  sess.run(init)
  
  ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
  for epoch in range(MAX_EPOCHS):

    for i in range(total_batch):
      batch_x, batch_y = batch_creator(X_train, batch_size, X_train.shape[0], output_size)
      _, c, summary = sess.run([optimizer, loss, merged], feed_dict = {x: batch_x, y: batch_y})
            
    #print('Epoch:{}'.format(epoch+1) + '. Cost = {:.5f}'.format(avg_cost))
    train_writer.add_summary(summary, epoch)

    # Save model weights to disk
    if (epoch+1) % 1000 == 0:
      save_path = saver.save(sess, model_path.format(epoch+1))
      print("Model saved in file: %s" % save_path)

    # compute MSE on validate set
    batch_x, batch_y = batch_creator(X_val, X_val.shape[0]-1, X_val.shape[0], output_size)
    [validate_loss, summary] = sess.run([loss, merged], feed_dict={x: batch_x, y: batch_y})

    #print("Epoch:{}. Validate MSE: {:.2f}".format(epoch+1, validate_loss))
    val_writer.add_summary(summary, epoch)

print("\nTraining complete!")

'''# Running testing session
print("---------------------------")
print("Starting testing session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    # Restore model weights from previously saved model
    epoch = 300
    saver.restore(sess, model_path.format(epoch))
    print('Checkpoint {} restored!'.format(epoch))

    batch_x, batch_y = batch_creator(X_val, 1, X_val.shape[0], output_size)
    test_loss, pred_out = sess.run([loss, output], feed_dict={x: batch_x, y: batch_y})
    print("Test MSE: {:.2f}".format(test_loss))

    # Visualizing as image
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title('Prediction')
    X_i = pred_out.reshape(stations, output_T)
    #print(X_i)
    ax.imshow(X_i, interpolation='bilinear')

    ax = fig.add_subplot(122)
    ax.set_title('Actual')
    X_i = batch_x.reshape(batch_x.shape[1], batch_x.shape[2])
    #print(X_i)
    ax.imshow(X_i, interpolation='bilinear')

    plt.show()'''

