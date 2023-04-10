# deep-learning-numpy

This is a library for the creation of neural networks, including fully connected and convolutional neural networks.

## Installation
First, clone this repository:
```console
$ git clone https://github.com/oliver-hamilton/deep-learning-numpy
```

Then change directory to the cloned repo and install the library using pip:
```console
$ pip install .
```

## Getting started
Note that the ```digit_recognition_test.py``` file requires the MNIST database of handwritten digits to run correctly.

Download the four files available at this link: http://web.archive.org/web/20230326164614/http://yann.lecun.com/exdb/mnist/.

Then decompress them, e.g. with:
```console
$ gunzip train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
```

These decompressed files should then be placed in a subdirectory ```deeplearningnumpy/resources``` from the root directory of this repo.
