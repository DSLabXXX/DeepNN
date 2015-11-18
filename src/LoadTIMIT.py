__author__ = 'iankuoli'

import _pickle
import gzip
import os

import numpy

import theano
import theano.tensor as T

def load_data(dataset_dir, partiotnguide_dir):

    #############
    # LOAD DATA #
    #############

    print('Start loading data ...')

    label_48to39 = dict()
    train_index = dict()
    test_index = dict()

    label2index = dict()
    index2label = dict()

    #
    # Path settings
    #
    map_path = dataset_dir + '/phones/48_39.map'
    train_guide_path = partiotnguide_dir + '/training.csv.gz'
    test_guide_path = partiotnguide_dir + '/testing.csv.gz'
    train_set_path = dataset_dir + '/fbank/train.ark'

    #
    # Mapping 48 phones to 39 phones
    #
    index = 0
    f_map48to39 = open(map_path, 'r')
    for l in f_map48to39:
        line = l.strip('\n').split('\t')
        label_48to39[line[0]] = line[1]

        if line[1] not in label2index:
            label2index[line[1]] = index
            index2label[index] = line[1]
            index += 1

    f_map48to39.close()

    #
    # Get the training index
    #
    f_train_guide = gzip.open(train_guide_path, 'rb')
    next(f_train_guide)
    for l in f_train_guide:
        line = l.decode("utf-8").strip('\n').split(',')
        train_index[line[0]] = line[1]
    f_train_guide.close()

    #
    # Get the testing index
    #
    f_test_guide = gzip.open(test_guide_path, 'rb')
    next(f_test_guide)
    for l in f_test_guide:
        line = l.decode("utf-8").strip('\n').split(',')
        test_index[line[0]] = line[1]
    f_test_guide.close()

    train_size = len(train_index)
    test_size = len(test_index)

    test_set_x = numpy.ndarray(shape=(test_size, 69), dtype=float)
    test_set_y = numpy.ndarray(shape=(test_size), dtype=int)
    train_set_x = numpy.ndarray(shape=(train_size, 69), dtype=float)
    train_set_y = numpy.ndarray(shape=(train_size), dtype=int)

    #
    # Read training data and partition it to training/testing set according to the index
    #

    train_num = 0
    test_num = 0
    f_train = open(train_set_path, 'r')
    for l in f_train:
        line = l.strip('\n').split(' ')
        id = line[0]

        if id in train_index.keys():
            x = numpy.asarray(line[1:], dtype=float)
            label = train_index[id]
            y = label2index[label]
            train_set_x[train_num, :] = x
            train_set_y[train_num] = y
            train_num += 1
        elif id in test_index.keys():
            x = numpy.asarray(line[1:], dtype=float)
            label = test_index[id]
            y = label2index[label]
            test_set_x[test_num, :] = x
            test_set_y[test_num] = y
            test_num += 1

    f_train.close()

    def shared_dataset(data_x, data_y, borrow=True):

        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_shared_x, test_shared_y = shared_dataset(test_set_x, test_set_y)
    train_shared_x, train_shared_y = shared_dataset(train_set_x, train_set_y)

    return (train_shared_x, train_shared_y), (test_shared_x, test_shared_y), label_48to39, label2index, index2label