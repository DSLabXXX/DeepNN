from src import Layer

__author__ = 'iankuoli'

__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T


class DNN(object):
    #
    # Initialization
    #
    def __init__(self, rng, inputdata, num_layer, num_in, num_out, num_hidden):

        self.layers = list()
        self.vec_output = numpy.zeros((num_out,), dtype=theano.config.floatX)

        self.L1 = 0
        self.L2_sqr = 0
        self.params = []

        for i in range(num_layer):
            # the (i+1)-th Layer

            if i == 0:
                layer = Layer.Layer(rng=rng, input=inputdata, num_in=num_in, num_out=num_hidden)
            elif i == (num_layer - 1):
                layer = Layer.Layer(rng=rng, input=self.layers[i - 1].vec_a, num_in=num_hidden, num_out=num_out, activation=T.nnet.softmax)
            else:
                layer = Layer.Layer(rng=rng, input=self.layers[i - 1].vec_a, num_in=num_hidden, num_out=num_hidden)

            self.layers.append(layer)
            self.L1 += abs(layer.mat_W).sum()
            self.L2_sqr += (layer.mat_W ** 2).sum()
            self.params += layer.params

        self.predict_y = self.layers[num_layer - 1].vec_a
        self.y_pred = T.argmax(self.predict_y, axis=1)
        print(self.y_pred)

    #
    # Negative log likelihood
    #
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.predict_y)[T.arange(y.shape[0]), y])

    #
    # L2 Norm
    #
    def l2_norm(self, y):
        return T.mean(T.sub(self.predict_y[T.arange(y.shape[0]), y], y) ** 2)

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def predict_labels(self):
        return self.y_pred
