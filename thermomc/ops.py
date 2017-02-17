# -*- coding: utf-8 -*-
""" Additional Theano ops. """

from __future__ import absolute_import, print_function, division

import numpy

import theano
from theano.scalar.basic import UnaryScalarOp, upgrade_to_float, float_types
from theano.tensor import elemwise

imported_scipy_special = False
try:
    import scipy.special
    import scipy.stats
    imported_scipy_special = True
except (ImportError, ValueError):
    pass


def construct_elemwise(name, doc_str, scalar_op):
    n = "Elemwise{%s, no_inplace}" % (name,)
    rval = elemwise.Elemwise(scalar_op, name=n,
                             nfunc_spec=(None and (None, None, None)))
    rval.__doc__ = doc_str + '\n' + rval.__doc__
    rval.__module__ = 'tensor'
    return rval


class I1(UnaryScalarOp):
    """Modified Bessel function of order 1."""

    @staticmethod
    def st_impl(x):
        return scipy.special.i1(x)

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(I1, self).impl(x)

    def grad(self, inp, grads):
        raise NotImplementedError()

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                i1(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')

i1_scalar = I1(upgrade_to_float, name='i1')
i1 = construct_elemwise(
    'i1', 'Modified Bessel function of order 1', i1_scalar)


class I0(UnaryScalarOp):
    """Modified Bessel function of order 0."""

    @staticmethod
    def st_impl(x):
        return scipy.special.i0(x)

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(I0, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [gz * i1(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                i0(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')

i0_scalar = I0(upgrade_to_float, name='i0')
i0 = construct_elemwise(
    'i0', 'Modified Bessel function of order 0', i0_scalar)
