## Training on multiple GPUs

The method `_tower_loss` defines a single tower in `example_train.py`

```shell
def _tower_loss(scope):
  """
  Here we implement a single tower of our network
  Args:
    scope: You must pass the scope name of your tower

  Returns: (_tower_loss must return)
    total_loss: Final loss that needs to be optimized
    input_var_names: A list of placeholder names
    keep_log: A list of tensors that you might want from the last tower. 
        Sometimes this is helpful for printing/debugging purpose. 
        If you don't need any log then pass an empty list.
  """
   ... 
   
  total_loss = tf.add_n(_losses + regularization_losses, name='total_loss')

  # Logs to keep, otherwise set: keep_log=[]
  keep_log = [total_loss]

  # List of place holders name
  input_var_names = [images.name, labels.name]

  return total_loss, input_var_names, keep_log
```

`_tower_loss` must return `total_loss, input_var_names, keep_log`. In order to train of multiple GPUs, initialize `MultiGPU` class in `example_train.py`.

```shell

GPU_IDS = ['1'] # List of GPUs. For example: GPU_IDS = ['0','1'] will use gpu_0 and gpu_1
VARIABLES_ON_DEVICE = '/cpu:0'

# Build the network on GPU_IDS
network_handle = MultiGPU(
  GPU_IDS,
  opt, # optimizer to use. For example: opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,momentum=RMSPROP_MOMENTUM,epsilon=RMSPROP_EPSILON)
  _tower_loss,
  variables_on_device=VARIABLES_ON_DEVICE,
  batch_norm_in_use=True
)
```

`MultiGPU` class provides symbolic handles to access loss, gradient, summary
  
```shell
# Get handles to loss, gradients, and sumaries
total_loss = network_handle.get_loss()
grads = network_handle.get_grad()
batchnorm_updates = network_handle.get_update_ops()
summaries = network_handle.get_summary()  # Summary from the last tower
```

`MultiGPU` also provides methods to run a training iteration, getting the summary from the network, and retrieving the log. 

```shell
...
network_handle.train_func(train_op, data_list, sess) # Runs one training iteration
...
tower_log = network_handle.tower_log_func(data_list, sess) # Returns keep_log (this is was one of the values returned from _tower_loss)
...
summary_str = network_handle.summary_func(summary_op, data_list, sess) # Writes the summary
...
```