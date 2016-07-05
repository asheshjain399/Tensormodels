"""
Author: Ashesh Jain [asheshjain399@gmail.com]
MIT License
"""

"""
About:

This is a sample training file. It runs forward and backward pass of inception_v3 on dummy data and labels.
"""


import time
import numpy as np
import os
from tensormodels import variables
from tensormodels import losses
import tensorflow as tf

"""Start with importing these three generic classes: DataFeeder, AugmentationWrapper, and MultiGPU"""
# DataFeeder is a generic class for creating data queue.
from tensormodels.data_reader.data_feeder import DataFeeder
# AugmentationWrapper is a generic class for data augmentation.
from tensormodels.augmentation.augmentation_wrapper import AugmentationWrapper
# MultiGPU is a generic class for training on single/multi-gpu.
from tensormodels.multigpu import MultiGPU

"""For your data set implement the ReaderClass and data augmentation files"""
from tensormodels.data_reader.example_multi_class_data_reader import ReaderClass
import tensormodels.augmentation.image_size_augment as image_augment

"""Implement your model (i.e., inference and loss functions)"""
import tensormodels.models.inception as model


# Input data stats
BATCH_SIZE = 32
NN_INPUT_SIZE = [(299, 299, 3)]  # (height,width,nchannels) This should be a tensor
OUTPUT_LABEL_SIZE = [(1,)]  # This should be a list of tensors
DATA_FILE = './example_train_data.txt' # This is a dummy training data file
LABEL_FILE = './example_train_label.txt' # This is a dummy training label file

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.
decay_steps = 10000
learning_rate_decay_factor = 0.16
initial_learning_rate = 1e-2

# Restore previous model
SUMMARY_DIR = './checkpoint/'  # _inceptionv3/'
RESTORE_PREVIOUS_MODEL = False
RESTORE_NEW_LAYERS = False  # True
RESTORE_TRAIN_MODEL = ''

# Training stats
MAX_ITERATIONS = 50000
PRINT_EVERY = 5
SUMMARY_EVERY = 50
SAVE_EVERY = 1000

# Device setting
GPU_IDS = ['1']
VARIABLES_ON_DEVICE = '/cpu:0'


def normalize_image(inputs):
  """ convert image from uint8[0,255] to float32[-1,1] """
  with tf.name_scope('normalize_image'):
    norm_image = tf.image.convert_image_dtype(inputs, dtype=tf.float32)
    norm_image = (norm_image - 0.5) * 2.0
    return norm_image


def _tower_loss(scope):
  """
  Here we implement a single tower of our network
  Args:
    scope: You must pass the scope name of your tower

  Returns:
    total_loss: Final loss that needs to be optimized
    input_var_names: A list of placeholder names
    keep_log: A list of tensors that you might want from the last tower. Sometimes this is helpful for printing/debugging purpose. If you don't need any log then pass an empty list.
  """

  # Create data and label placeholders
  input_tensor_shape = [None] + list(NN_INPUT_SIZE[0])

  with tf.name_scope('images'):
    images = tf.placeholder(tf.uint8, shape=input_tensor_shape)
    images_normalized = normalize_image(images)

  with tf.name_scope('labels'):
    labels = tf.placeholder(tf.int32, shape=[None,])

  # Build the network and graph
  logits,auxiliary_logits = model.inference(images_normalized, 1000, for_training=True, scope=scope)
  model.loss(logits, labels)
  model.loss(auxiliary_logits, labels)
  _losses = tf.get_collection(losses.LOSSES_COLLECTION,scope)
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n(_losses + regularization_losses, name='total_loss')

  # Logs to keep otherwise set: keep_log=[]
  keep_log = [total_loss]

  # List of place holders name
  input_var_names = [images.name, labels.name]

  return total_loss, input_var_names, keep_log


def train():
  gg = tf.Graph()
  with gg.as_default(), tf.device(VARIABLES_ON_DEVICE):

    # Data augmentation ops
    augment_ops = ['resize']
    augment_vals = [[(299,299)]]
    data_augment = AugmentationWrapper(augment_ops=augment_ops, augment_vals=augment_vals, ops_func_holder=image_augment)

    # Starting the datapipe line
    dp = DataFeeder(DATA_FILE,
               LABEL_FILE,
               ReaderClass(augment_classes=[data_augment]),
               NN_INPUT_SIZE,
               OUTPUT_LABEL_SIZE,
               batch_size=BATCH_SIZE,
               num_preprocess_threads=1,
               data_type=np.uint8,
               label_type=np.int32)

    global_step = tf.Variable(0, trainable=False)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                    global_step,
                    decay_steps,
                    learning_rate_decay_factor,
                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                    momentum=RMSPROP_MOMENTUM,
                    epsilon=RMSPROP_EPSILON)

    # Build the network on GPU_IDS
    network_handle = MultiGPU(
      GPU_IDS,
      opt,
      _tower_loss,
      variables_on_device=VARIABLES_ON_DEVICE,
      batch_norm_in_use=True
    )

    # Get handles to loss, gradients, and sumaries
    total_loss = network_handle.get_loss()
    grads = network_handle.get_grad()
    batchnorm_updates = network_handle.get_update_ops()
    summaries = network_handle.get_summary()  # Summary from the last tower

    # Add learning rate summary
    summaries.append(tf.scalar_summary('learning_rate', lr))

    # Add summary for learning rate
    summaries.append(tf.scalar_summary('loss', total_loss))

    # Optimize
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    # Create a session and initialize variables
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.initialize_all_variables())

    # Total number of parameters in the network
    total_params = 0
    for v in tf.trainable_variables():
      count = 1
      _shape = v.get_shape()
      for j in _shape:
        count = count * int(j)
      total_params += count
    print 'Total learnable parameters: ', total_params

    # Load pre-trained model
    if RESTORE_PREVIOUS_MODEL:
      variables_to_restore = tf.get_collection(variables.VARIABLES_TO_RESTORE)
      restorer = tf.train.Saver(variables_to_restore)
      restorer.restore(sess, RESTORE_TRAIN_MODEL)

    # Create summary op
    summary_dir = SUMMARY_DIR
    if not os.path.exists(summary_dir):
      os.mkdir(summary_dir)
    summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)
    summary_op = tf.merge_summary(summaries)

    # Create checkpoint op
    saver = tf.train.Saver(tf.all_variables())

    total_dequeue_duration = 0.0

    try:
      for step in range(MAX_ITERATIONS):

        # Get one mini-batch of data from queue
        dequeue_start_time = time.time()
        data = dp.get_data()
        dequeue_duration = time.time() - dequeue_start_time
        total_dequeue_duration = total_dequeue_duration + dequeue_duration
        y = data[1].reshape((data[1].shape[0],))
        data_list = [data[0], y]

        print data[0].shape
        print y.shape
        
        # Run a training iteration
        start_time = time.time()
        network_handle.train_func(train_op, data_list, sess)
        duration = time.time() - start_time

        # Prints some stats about the training
        if step % PRINT_EVERY == 0:
          # Get the log from the last tower
          tower_log = network_handle.tower_log_func(data_list, sess)
          loss_value = tower_log[0]

          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
          examples_per_sec = BATCH_SIZE / float(duration)
          print "step={0} loss={1:.4f} {2:.3f} (sec/batch) {3:.2f} (examples/sec)".format(
            step, loss_value, duration, examples_per_sec)

        # Generates the summary
        if step % SUMMARY_EVERY == 0:
          average_dequeue_duration = total_dequeue_duration / float(SUMMARY_EVERY)
          print("Avg dequeue duration = {0:.3f} (sec/batch)".format(average_dequeue_duration))
          total_dequeue_duration = 0.0

          summary_str = network_handle.summary_func(summary_op, data_list, sess)
          summary_writer.add_summary(summary_str, step)

        # Save the network
        if step % SAVE_EVERY == 0:
          checkpoint_path = SUMMARY_DIR + 'model.ckpt'
          print saver.save(sess, checkpoint_path, global_step=step)

    finally:
      dp.clean_and_close()
      print "Queue successfully closed. Exiting program."


if __name__ == '__main__':
  train()