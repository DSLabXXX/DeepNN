__author__ = 'iankuoli'

__docformat__ = 'restructedtext en'

import numpy
import theano
import theano.tensor as T

#rng = numpy.random.RandomState(1234)
#srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

"""
.. module:: Layer
   :synopsis: To drop out a part of the units we use a Bernoulli distribution which is the same as a Binomial
   distribution with just one trial, i.e. n=1

.. moduleauthor:: Ian Kuo

"""

def drop(input, srng, p=0.5):

    """
    To drop out a part of the units we use a Bernoulli distribution which is the same as a Binomial distribution
    with just one trial, i.e. n=1

    Args:
        :type input: numpy.array
        :param input: layer or weight matrix on which dropout resp. dropconnect is applied

        :type p: float or double between 0. and 1.
        :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.

    Returns:
        :type input: numpy.array
        :param input: layer or weight matrix on which dropout resp. dropconnect is applied

    Raises:
        AttributeError, KeyError
    """

    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask

class Layer(object):

    def __init__(self, rng, srng, is_train, input, num_in, num_out, mat_W=None, vec_b=None, activation=T.tanh, p=0.5):

        """
        Hidden unit activation is given by: activation(dot(input,W) + b)

        Args:
            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights

            :type srng: theano.tensor.shared_randomstreams.RandomStreamse
            :param srng: a random stream

            :type is_train: theano.iscalar
            :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

            :type input: theano.tensor.dmatrix
            :param input: a symbolic tensor of shape (n_examples, n_in)

            :type num_in: int
            :param num_in: dimensionality of input vector

            :type num_out: int
            :param num_out: dimensionality of output vector

            :type activation: theano.Op or function
            :param activation: Non linearity to be applied in the hidden layer

            :type p: float or double
            :param p: probability of NOT dropping out a unit

        Returns:

        Raises:

        """

        self.input = input
        self.vec_Delta = T.matrix('mat_Delta')
        self.vec_Sigma = T.vector('vec_Sigma')

        if mat_W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (num_in + num_out)),
                    high=numpy.sqrt(6. / (num_in + num_out)),
                    size=(num_in, num_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            mat_W = theano.shared(value=W_values, name='mat_W', borrow=True)

        if vec_b is None:
            b_values = numpy.zeros((num_out,), dtype=theano.config.floatX)
            vec_b = theano.shared(value=b_values, name='vec_b', borrow=True)

        """ parameters of the layer """
        self.mat_W = mat_W
        self.vec_b = vec_b

        """ linear combination """
        self.vec_z = T.dot(input, self.mat_W) + self.vec_b

        output = (
            self.vec_z if activation is None
            else activation(self.vec_z)
        )

        """ multiply output and drop -> in an approximation the scaling effects cancel out """
        train_output = drop(numpy.cast[theano.config.floatX](1./p) * output, srng)

        """
        Output vector by an activation function (tanh or sigmoid)
        is_train is a pseudo boolean theano variable for switching between training and prediction
        In the implementation there is a pseudo-boolean for switching between train and prediction.
        Theano has no boolean variables so an integer is used.
        """
        self.vec_a = T.switch(T.neq(is_train, 0), train_output, output)

        # parameters of the model
        self.params = [self.mat_W, self.vec_b]