# Tensormodels
A python package for Tensorflow. It has the right level of abstraction needed to built state-of-the-art deep learning models. Tensormodels also provides queues for data prefetching, so you don't have to use Tensorflow data queues and you can write custom augmentation functions. It also allows training on multi-gpus without any change in code.   

## Install

To install with sudo permission ```sudo python setup.py develop```

### Test the installation

`example/example_train.py` loads some dummy data (images) and labels, and run few iterations of inception_v3.

```shell
cd example/
python example_train.py

step=0 loss=17.0721 10.698 (sec/batch) 2.99 (examples/sec)
step=5 loss=17.1091 1.533 (sec/batch) 20.87 (examples/sec)
step=10 loss=17.1296 1.532 (sec/batch) 20.89 (examples/sec)
step=15 loss=17.0957 1.534 (sec/batch) 20.86 (examples/sec)
step=20 loss=17.0909 1.540 (sec/batch) 20.77 (examples/sec)
```

You can set `GPU_IDS` in `example_train.py` to the GPUs you would like to use.
 
