"""
Author: Ashesh Jain [asheshjain399@gmail.com]
MIT License
"""

from multiprocessing import Queue, Process
import multiprocessing as mp
import numpy as np

from numpy.random import RandomState
rand = RandomState(123)


class DataFeeder():
  """
  DataFeeder creates data processing queues
  """

  def __init__(self,
               data_file, # Each line of data file is a data instnance.
               label_file, # Each line of label file is a label training instance.
               readerClass,  # This is a class which defines how to decode the data and label instances and Y. For an example see multi_class_data_reader.py for multi label image clasification.
               data_list_shapes, # This is list of shapes of the data
               label_list_shapes, # This is list of shapes for labels
               batch_size=32,
               num_preprocess_threads=int(0.3 * mp.cpu_count()),
               num_gpu_towers=1,
               num_epochs=-1,
               dataType=np.uint8, # This is dtype of each tensor in data_list_shapes. TODO: Make this a list of dtypes
               labelType=np.int32 # This is dtype of each tensor in label_list_shapes. TODO: Make this a list of dtypes
               ):

    self.data_file = data_file
    self.label_file = label_file
    self.data_list_shapes = data_list_shapes
    self.label_list_shapes = label_list_shapes
    self.readerClass = readerClass
    self.num_preprocess_threads = num_preprocess_threads
    self.num_gpu_towers = num_gpu_towers
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.dataType = dataType
    self.labelType = labelType
    self._launch_pipeline()

  def _launch_pipeline(self):
    """This method creates two queues.
    filename_queue: stores the list of filesnames in data_file and label_file
    data_queue: stores the mini-batch
    """

    self.data_processes = [] # Holds process handles

    queue_size = 2 * self.num_preprocess_threads + 2 * self.num_gpu_towers
    self.data_queue = Queue(queue_size)  # This queue stores the data
    image_files = open(self.data_file, 'r').readlines()
    labels = open(self.label_file, 'r').readlines()
    print 'Size of queue: ', queue_size

    self.filename_queue = Queue(len(image_files))  # This queue stores the filenames
    p = Process(target=self._create_filename_queue, args=(self.filename_queue, image_files, labels, self.num_epochs))
    p.start()
    self.data_processes.append(p)

    print 'Data feeder started'
    for each_worker in range(self.num_preprocess_threads):
      p = Process(target=self._each_worker_process, args=(self.data_queue))
      p.start()
      self.data_processes.append(p)

  def _each_worker_process(self, data_queue):
    """This method produces the mini-batch and puts it in the data_queue (which is a multi-process queue)"""

    # Produce mini-batches forever. (Till the end of life!!)
    while True:

      # Batch data holders
      batch_data_holder = []
      for s in self.data_list_shapes:
        batch_data_holder.append(np.repeat(
          np.expand_dims(np.zeros(s, dtype=self.dataType), axis=0),
          self.batch_size, axis=0))

      batch_label_holder = []
      for s in self.label_list_shapes:
        batch_label_holder.append(np.repeat(
          np.expand_dims(np.zeros(s, dtype=self.labelType), axis=0),
          self.batch_size, axis=0))

      # For each example in mini-batch
      for i in range(self.batch_size):

        # Get the filename to read from the filename_queue
        [image_path, label_path] = self.filename_queue.get()

        # Read the data and label at the image_path and label_path, respectively
        batch_data_list, label_data_list, _, _ = self.readerClass.reader(image_path, label_path)

        for j in range(len(batch_data_list)):
          _data = batch_data_list[j]
          batch_data_holder[j][i] = _data

        for j in range(len(label_data_list)):
          label_data = label_data_list[j]
          batch_label_holder[j][i] = label_data

      # Put the mini-batch in the queue
      batch_holder = batch_data_holder + batch_label_holder
      data_queue.put(batch_holder)

  def _create_filename_queue(self, filename_queue, image_files, labels, num_epochs=-1):
    """This method populates the filename_queue with the filenames"""

    num_examples = len(image_files)
    while not (num_epochs == 0):
      _image_files = np.array(image_files)
      _labels = np.array(labels)
      permute = rand.permutation(num_examples)
      _image_files = _image_files[permute]
      _labels = _labels[permute]
      _image_files = list(_image_files)
      _labels = list(_labels)

      for img, label in zip(_image_files, _labels):
        filename_queue.put([img.strip(), label.strip()])
      num_epochs -= 1

  def get_process_handle(self):
    return self.data_processes

  def data_qsize(self):
    try:
      return self.data_queue.qsize() # This will not work on mac
    except:
      print "Is not supported on Mac"
      return -1

  def get_data(self):
    """Returns a mini-batch"""
    return self.data_queue.get()

  def clean_and_close(self):
    """Closes all the data queues."""
    for handle in self.data_processes:
      handle.terminate()
      handle.join()
    self.data_queue.close()
    self.data_queue.join_thread()




