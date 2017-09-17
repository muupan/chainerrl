from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import numpy

import chainer
from chainer.utils import type_check

import scipy


def _tuple_to_gpu(xs):
    return tuple(chainer.cuda.to_gpu(x) for x in xs)


def _tuple_to_cpu(xs):
    return tuple(chainer.cuda.to_cpu(x) for x in xs)


class Polygamma(chainer.Function):
    """Polygamma function"""

    def __init__(self, n):
        self.n = n

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype == numpy.float32
        )

    def forward_cpu(self, inputs):
        return scipy.special.polygamma(self.n, inputs[0]),

    def forward_gpu(self, inputs):
        # TODO(muupan): use gpu
        return _tuple_to_gpu(
            self.forward_cpu(_tuple_to_cpu(inputs)))

    def backward_cpu(self, inputs, grad_outputs):
        return (scipy.special.polygamma(self.n + 1, inputs[0])
                * grad_outputs[0]),

    def backward_gpu(self, inputs, grad_outputs):
        # TODO(muupan): use gpu
        return _tuple_to_gpu(
            self.backward_cpu(_tuple_to_cpu(inputs),
                              _tuple_to_cpu(grad_outputs)))


def polygamma(n, x):
    """Polygamma function"""
    return Polygamma(n)(x)
