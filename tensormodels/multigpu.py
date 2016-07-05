"""
Author: Ashesh Jain [asheshjain399@gmail.com]
MIT License
"""

import tensorflow as tf
from tensormodels import ops
from tensormodels import scopes
from tensormodels import variables

class MultiGPU:
  """
  This class allows trianing on multiple GPUs.
  """

  def __init__(self,
               gpus,
               opt,
               tower_handle,
               variables_on_device='/cpu:0',
               batch_norm_in_use=False):
    """
    Args:
      gpus: GPU id's as a list of strings
      tower_handle: Function handle of how to build a tower
      variables_on_device: Where to keep the variables 
      batch_norm_in_use: Set this boolean to true if you are using batch norm. 
        It will return the batch norm mean/variance from the last tower
    """
    self.gpu_ids = gpus
    self.opt = opt
    self.batch_norm_in_use = batch_norm_in_use
    self.tower_handle = tower_handle
    self.variables_on_device = variables_on_device
    self._grad()  # Builds the tower on each GPU

  def get_update_ops(self):
    return self.batchnorm_updates

  def get_grad(self):
    # Returns gradient averaged across all towers
    return self.grads

  def get_loss(self):
    # Returns loss from the last tower
    return self.loss

  def get_summary(self):
    # Returns summary from the last tower
    return self.summaries

  def _grad(self):

    """
    Builds the tower on each GPU
    """

    global batchnorm_updates
    tower_grads = []  # Store the gradient from each GPU
    self.var_names = []  # Stores the input placeholder names from each tower

    # For each GPU
    for _gpu in self.gpu_ids:

      # Build the tower
      with tf.device('/gpu:{0}'.format(_gpu)):
        with tf.name_scope('tower_{0}'.format(_gpu)) as scope:
          # Here we put all the variables on 'variables_on_device'
          with scopes.arg_scope([variables.variable], device=self.variables_on_device):
            loss, input_var_names, tower_log = self.tower_handle(scope)

          self.var_names = self.var_names + input_var_names
          num_placeholders_per_tower = len(input_var_names)

          # Re-use variables for the next tower
          tf.get_variable_scope().reuse_variables()

          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          if self.batch_norm_in_use:
            batchnorm_updates = tf.get_collection(ops.UPDATE_OPS_COLLECTION, scope)

          grads = self.opt.compute_gradients(loss)
          tower_grads.append(grads)

    # Average gradient from the GPUs. This is where GPUs synchronization happens. 
    grads = _average_gradients(tower_grads)

    self.grads = grads
    self.summaries = summaries  # Summary from the last tower
    self.tower_log = tower_log  # Log from the last tower
    self.loss = loss  # Loss on the last tower
    self.num_placeholders = num_placeholders_per_tower  # Number of placeholders per tower

    if self.batch_norm_in_use:
      self.batchnorm_updates = batchnorm_updates

  def _feed_dict(self, data_list):
    """
    Splits data equally across all the GPUs and returns the feed_dict.
    
    Args:
      data_list: List of tensors

    Returns:
      feed_dict: A dictionary of placeholders and their values. 
    """

    assert (len(
      data_list) == self.num_placeholders), 'Dimension mis-match. Number of placeholders and number of input tensors should match.'

    num_examples = data_list[0].shape[0]
    batch_per_gpu = int(data_list[0].shape[0] / len(self.gpu_ids))
    assert (
    batch_per_gpu >= 1), 'Not enough data to feed all GPUs. Either reduce the number of GPU towers or increase the batch size.'

    batch_start = 0
    batch_end = batch_per_gpu

    feed_dict = {}
    var_id = 0
    for _gpu in self.gpu_ids:

      end_id = min(num_examples, batch_end)
      for data in data_list:
        var_name = self.var_names[var_id]
        feed_dict[var_name] = data[batch_start:end_id]
        var_id += 1

      batch_start += batch_per_gpu
      batch_end += batch_per_gpu

    return feed_dict

  def summary_func(self, summary_op, data_list, sess):
    """
    Generates the summary

    Args:
      summary_op: Summary op
      data_list: List of input tensors
      sess: Tensorflow session
    
    Returns:
      summary_str: Generated summary 
    """
    feed_dict = self._feed_dict(data_list)
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    return summary_str

  def tower_log_func(self, data_list, sess):
    """
    Performs a forward pass and returns the tower_log

    Args:
      data_list: List of input tensors
      sess: Tensorflow session
    
    Returns:
      tensor_log: List of tensors from the last tower  
    """
    feed_dict = self._feed_dict(data_list)
    tower_log = sess.run(self.tower_log, feed_dict=feed_dict)
    return tower_log

  def train_func(self, train_op, data_list, sess):
    """
    Performs a training iteration
    """

    feed_dict = self._feed_dict(data_list)
    sess.run(train_op, feed_dict=feed_dict)

def _average_gradients(tower_grads):
  """
  This method is taken from google/tensorflow/inception_v3

  Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads