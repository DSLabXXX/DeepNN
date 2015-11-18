__author__ = 'iankuoli'

import numpy
import theano.tensor as T

a = numpy.asarray([1, 2, 3])
b = T.arange(a.shape[0])
print(b)
print([T.arange(a.shape[0]), a])