from src import LoadTIMIT, DNN

__author__ = 'iankuoli'

import math
import _pickle

import numpy

import theano
import theano.tensor as T


dataset='/Volumes/My Book/Downloads/MLDS_HW1_RELEASE_v1'
partiotnguide = 'sample1'
batch_size = 20

(train_set_x, train_set_y), (test_set_x, test_set_y), label_48to39, label2index, index2label = LoadTIMIT.load_data(dataset, partiotnguide)
n_test_batches = math.floor(test_set_x.get_value(borrow=True).shape[0] / batch_size)

f = open('model_8layer_lr0.0005_0.376.pkl', 'rb')
load_dnn = _pickle.load(f, encoding='latin1')
f.close()

index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # the feature vectors of training data
y = T.ivector('y')  # the labels

rng = numpy.random.RandomState(1234)
dnn = DNN.DNN(rng=rng, inputdata=x, num_in=train_set_x.container.data.shape[1], num_hidden=500,
              num_out=len(label2index), num_layer=len(load_dnn.layers))

print(len(load_dnn.layers))
print(train_set_x.container.data.shape[1])

for i in range(len(dnn.layers)):
    dnn.layers[i].mat_W.set_value(load_dnn.layers[i].mat_W.get_value(borrow=True))
    dnn.layers[i].vec_b.set_value(load_dnn.layers[i].vec_b.get_value(borrow=True))
    dnn.layers[i].params = [dnn.layers[i].mat_W, dnn.layers[i].vec_b]
    '''
    dnn.layers[i].vec_z = T.dot(input, dnn.layers[i].mat_W) + dnn.layers[i].vec_b

    if i == 0:
        dnn.layers[i].input = x
        dnn.layers[i].vec_a = T.tanh(dnn.layers[i].vec_z)
    elif i == (dnn.num_layer - 1):
        dnn.layers[i].input = dnn.layers[i - 1].vec_a
        dnn.layers[i].vec_a = T.nnet.softmax(dnn.layers[i].vec_z)
    else:
        dnn.layers[i].input = dnn.layers[i - 1].vec_a
        dnn.layers[i].vec_a = T.tanh(dnn.layers[i].vec_z)
    '''


###############
# TEST MODEL #
###############

test_num = 0
f_test = open(dataset + '/fbank/test.ark', 'r')
list_test = list()
testIDs = list()
for l in f_test:
    line = l.strip('\n').split(' ')
    testIDs.append(line[0])

    list_test.append(numpy.asarray(line[1:], dtype=float))
    test_num += 1

f_test.close()

test2_set_x = numpy.asarray(list_test)

'''
col_sums = test2_set_x.sum(axis=0)
tmp = test2_set_x / col_sums[numpy.newaxis, :]
test2_set_x = tmp
'''
shared_x = theano.shared(numpy.asarray(test2_set_x, dtype=theano.config.floatX), borrow=True)

'''
test_model = theano.function(
    inputs=[index],
    outputs=dnn.errors(y),
    givens={
        x: test_set_x[index * batch_size:(index + 1) * batch_size],
        y: test_set_y[index * batch_size:(index + 1) * batch_size]
    }
)
'''
test_model = theano.function(
    inputs=[index],
    on_unused_input='ignore',
    outputs=dnn.predict_y,
    givens={
        x: shared_x,
    }
)

test_losses = [test_model(i) for i in range(n_test_batches)]
this_test_loss = numpy.mean(test_losses)

print('validation error %f %%' %
    (
        this_test_loss * 100.
    )
)




###############
# TEST MODEL #
###############

theano.config.exception_verbosity = 'high'

test_model2 = theano.function(
    inputs=[index],
    on_unused_input='ignore',
    outputs=dnn.predict_y,
    givens={
        x: shared_x,
    }
)

list_pred_y = list()

print(len(test2_set_x))

list_pred_y = test_model2(0)

list_pred_labels = [index2label[i] for i in list_pred_y]

f_lables = open('label_pred.csv', 'w')

for i in range(len(list_pred_labels)):
    str = testIDs[i] + ',' + list_pred_labels[i] + '\n'
    f_lables.write(str)

f_lables.close()