"""
Author: Ashesh Jain [asheshjain399@gmail.com]
MIT License
"""
import numpy as np
from numpy.random import RandomState
rand = RandomState(123)


class ReaderClass:
  """The readerClass returns a dummy (299,299,3) image and a random image label [0,1000)

  readerClass must implement the reader() method. This method is called by the data_feeder.py in order to populate the data_queue"""

  def __init__(self, augment_classes):
    """
    Args
      augment_classes: list of data augmentation classes  (i.e., utils/augmentation_wrapper.py). Each element of this list performs a sequence of data of augmentation.
    """

    self.augment_classes = augment_classes

  def reader(self, img_path, label_path):

    """
    Every readerClass must implement this method
    Args:
      img_path: This is the string where the data resides. You should implement how to read this string. For example, this can be path to an image.
      label_path: This is the string where the label resides. You should implement how to read this string.
    Returns:
      [img]: List of tensors of the data
      labels: List of tensors of the labels
      image_summary: List of intermediate images generated
      ops_summary: List of images augmentation operations applied.
    """

    img = _data_reader(img_path) # Read image
    labels = _label_reader(label_path) # Read label

    image_summary = [img]
    label_summary = [labels]
    ops_summary = ['original']

    if self.augment_classes is not None:

      # Convert image to float32 before augmentation ops
      img = img.astype(np.float32)

      # We iterate over each augment class
      for augment_class in self.augment_classes:
        [img, labels, _image_summary, _label_summary, _ops_summary] = augment_class.augment_data(img, labels)
        image_summary = image_summary + _image_summary
        label_summary = label_summary + _label_summary
        ops_summary = ops_summary + _ops_summary

      # Clip the image between [0,255] and convert it to uint8
      img = np.clip(img, 0, 255)
      img = img.astype(np.uint8)

    # First two return values must be list of tensors.
    return [img], labels, image_summary, ops_summary

def _data_reader(img_path):
  """
  Args:
    img_path: Path where the image is located.
  """

  return (150.0*rand.rand(400,400,3)).astype(np.uint8)

def _label_reader(label_path):
  """
  Args:
    label_path: this is simply the image class label.
  """
  return [rand.randint(0,1000)]