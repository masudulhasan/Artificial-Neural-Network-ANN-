import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros,arange

from sklearn import cross_validation


def vectorized_result(j):
    e = zeros((10, 1))
    j=int(j)
    e[j] = 1.0
    return e

def load_mnist(dataset="training", digits=arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    i=1
    data=[]
    index=0
    for I in images:
        Ara = []
        for J in I:
            for K in J:
                Ara.append(K)
        npAra=array(Ara)
        data.append(npAra)
        # print type(data_train[0])
    # Label=labels
    if dataset == "training":
        Label = [vectorized_result(y) for y in labels]
    # print Label
    else:
        Label = []
        for I in labels:
            for J in I:
                Label.append(J)

        Label=array(Label)


    return data, Label

def load_data():
    from pylab import *
    from numpy import *

    # images, labels = load_mnist('training', digits=[2])
    # for x in xrange(0):
    images, labels = load_mnist('training')

    for x in xrange(len(images)):
        images[x] = images[x] / 255.0

    # print images[0].shape
    # print images[0]
    # print labels
    data_train, data_test, target_train, target_test = cross_validation.train_test_split(images,
                                                                                         labels,
                                                                                         test_size=0.99,
                                                                                         random_state=43)
    # training_data = zip(images,labels)
    training_data = zip(data_train,target_train)

    images, labels = load_mnist('testing')

    for x in xrange(len(images)):
        images[x] = images[x] / 255.0

    test_data = zip(images,labels)
    validation_data=[]
    return (training_data, validation_data, test_data)
load_data()