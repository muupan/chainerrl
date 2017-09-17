from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

import numpy

from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

import chainerrl

import scipy


@testing.parameterize(
    *testing.product({
        'n': [0, 1, 2],
        'shape': [(1,), (1, 1), (2,), (2, 3)],
    })
)
class TestPolygamma(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(
            0.5, 2, self.shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, self.shape).astype(numpy.float32)

    def check_forward(self, x):
        y = chainerrl.functions.polygamma(self.n, x)
        correct_y = scipy.special.polygamma(self.n, cuda.to_cpu(x))
        gradient_check.assert_allclose(correct_y, cuda.to_cpu(y.data))

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            chainerrl.functions.Polygamma(self.n),
            x_data, y_grad, eps=1e-2, rtol=1e-2)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
