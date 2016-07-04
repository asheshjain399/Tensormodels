import tensormodels.utils.image_augmentation as augment
import numpy as np
from numpy.random import RandomState
rand = RandomState(123)


class AugmentationWrapper:
  """This is the wrapper function to perform data augmentation. Whatever augmentation you'd like to perform send in 'augment-ops' that can be found in 'ops_func_holder'  """
  def __init__(self,
               augment_ops=None,
               augment_vals=None,
               ops_func_holder=augment,
               shuffle_ops=False):
    """
    augment_ops: List of augment ops
    augment_vals: List of input args into augment_ops function
    ops_func_holder: function for accessing augment_ops
    shuffle_ops: boolean, used to shuffle augment_ops and augment_vals
    """
    self.ops = augment_ops
    self.ops_val = augment_vals
    self.shuffle_ops = shuffle_ops
    self.ops_func_holder = ops_func_holder

  def augment_data(self, data, label):
    """
    This function is called by the ReaderClass to augment the data.
    Args:
      data: to be augmented (For e.g., image) 
      label:  to be augmented (For e.g., bounding box)

    Returns:
      data: after augmentation (For e.g., cropped image)
      label: after augmentation (For e.g., updated bounding box)
      data_summary: is a list of intermediate states of data
      label_summary: is a list of intermediate states of data
      ops_summary: is a list of names (string) of ops applied
    """

    data_summary = []
    label_summary = []
    ops_summary = []

    augment_ops = self.ops
    augment_vals = self.ops_val
    if self.shuffle_ops:
      augment_ops = np.array(self.ops)  # It is easier to permute arrays
      augment_vals = np.array(self.ops_val)
      permute = rand.permutation(len(augment_ops))
      augment_ops = list(augment_ops[permute])  # Convert back to list
      augment_vals = list(augment_vals[permute])

    for _ops, _vals in zip(augment_ops, augment_vals):
      if rand.rand() < 0.1 and self.shuffle_ops:  # with a small probability drop an op (only when shuffle op is true)
        continue
      ops_func = getattr(self.ops_func_holder, _ops)
      params = [data, label] + _vals
      data, label = ops_func(*params)
      data_summary.append(data)
      label_summary.append(label)
      ops_summary.append(_ops)

    return data, label, data_summary, label_summary, ops_summary