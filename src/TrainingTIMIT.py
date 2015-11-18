from src import LoadTIMIT, DNN

__author__ = 'iankuoli'

__docformat__ = 'restructedtext en'

import time
import math
import _pickle

import numpy

import theano
import theano.tensor as T


def training(lr=0.0005, L1_reg=0.00, L2_reg=0.0000, n_epochs=200,
             dataset='/Volumes/My Book/Downloads/MLDS_HW1_RELEASE_v1', partiotnguide='sample1', batch_size=20, n_hidden=500, n_layer=8):

    (train_set_x, train_set_y), (test_set_x, test_set_y), label_48to39, label2index, index2label = LoadTIMIT.load_data(dataset, partiotnguide)

    # compute number of minibatches for training, validation and testing
    n_train_batches = math.floor(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_test_batches = math.floor(test_set_x.get_value(borrow=True).shape[0] / batch_size)

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the feature vectors of training data
    y = T.ivector('y')  # the labels

    rng = numpy.random.RandomState(1234)

    dnn = DNN.DNN(rng=rng, inputdata=x, num_in=train_set_x.container.data.shape[1], num_hidden=n_hidden,
                  num_out=len(label2index), num_layer=n_layer)

    cost = (dnn.negative_log_likelihood(y) + L1_reg * dnn.L1 + L2_reg * dnn.L2_sqr)
    #cost = (dnn.l2_norm(y) + L1_reg * dnn.L1 + L2_reg * dnn.L2_sqr)

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=dnn.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in dnn.params]

    updates = [(param, param - lr * gparam) for param, gparam in zip(dnn.params, gparams)]
    ll = 0.9
    train_model = theano.function(
        inputs=[index, ll],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('Start training ...')

    #
    # --- early-stopping parameters ---
    #

    # look as this many examples regardless
    patience = 500000
    # wait this much longer when a new best is found
    patience_increase = 2
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995
    # go through this many minibatche before checking the network on the validation set;
    # in this case we check every epoch
    test_frequency = min(n_train_batches, patience / 100)

    best_test_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % test_frequency == 0:

                # compute zero-one loss on validation set
                test_losses = [test_model(i) for i in range(n_test_batches)]
                this_test_loss = numpy.mean(test_losses)

                print('Epoch: %i; Batch: %i/%i, ValidationError: %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_test_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_test_loss < best_test_loss:
                    #improve patience if loss improvement is good enough
                    if this_test_loss < best_test_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        fh = open('model.pkl', 'wb')
                        _pickle.dump(dnn, fh)

                    best_test_loss = this_test_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = False
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'Total iteration: %i. Best performance: %f %%') %
          ((1 - best_test_loss) * 100., best_iter + 1, (1 - test_score) * 100.))


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

    ###############
    # TEST MODEL #
    ###############

    test_model2 = theano.function(
        inputs=[index],
        on_unused_input='ignore',
        outputs=dnn.predict_labels(),
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
        strii = testIDs[i] + ',' + list_pred_labels[i] + '\n'
        f_lables.write(strii)

    f_lables.close()

if __name__ == '__main__':
    training()
