# -*- coding: utf-8 -*-
""" Continuous inverse temperature control function classes."""

import numpy as np
import theano.tensor as tt
from thermomc.ops import i0, i1


class CircularControlFunction(object):

    def __init__(self, scale):
        assert scale > 0, 'scale should be a positive scalar.'
        self.scale = scale

    def inv_temp_func(self, tmp_ctrl_var):
        u = tmp_ctrl_var / self.scale
        return (1. + tt.cos(np.pi * u)) / 2.

    def inv_temp_cond_prob_0_1(self, delta):
        prob_0 = tt.exp(delta / 2) / (2 * i0(delta / 2))
        prob_1 = tt.exp(-delta / 2) / (2 * i0(delta / 2))
        return prob_0, prob_1

    def inv_temp_cond_prob_func(self, inv_temp, delta):
        return tt.exp(-inv_temp * delta) / (
            2 * tt.exp(-delta / 2) * i0(delta / 2)
        )

    def log_jacobian_term(self, tmp_ctrl_var):
        return 0.


class ThresholdedCircularControlFunction(object):

    def __init__(self, scale, theta_1, theta_2):
        assert scale > 0, 'scale should be a positive scalar.'
        assert theta_1 >= 0 and theta_1 < theta_2, (
            'Lower threshold should be in [0, 1] and strictly less than upper.'
        )
        assert theta_2 <= 1., (
            'Upper threshold should in [0, 1].'
        )
        self.scale = scale
        self.theta_1 = theta_1
        self.theta_2 = theta_2

    def inv_temp_func(self, tmp_ctrl_var):
        u = tt.abs_(tt.mod((tmp_ctrl_var / self.scale) + 1., 2.) - 1.)
        z = (u - self.theta_1) / (self.theta_2 - self.theta_1)
        return tt.switch(
            tt.le(z, 0), 1.,
            tt.switch(tt.ge(z, 1), 0., (1. + tt.cos(np.pi * z)) / 2.)
        )

    def inv_temp_cond_prob_0_1(self, delta):
        prob_0 = 1. / (2 * (
            self.theta_1 * tt.exp(-delta) + 1. - self.theta_2 +
            (self.theta_2 - self.theta_1) * tt.exp(-delta / 2) * i0(delta / 2)
            )
        )
        prob_1 = tt.exp(-delta) * prob_0
        return prob_0, prob_1

    def inv_temp_cond_prob_func(self, inv_temp, delta):
        return tt.exp(-inv_temp * delta) / (2 * (
            self.theta_1 * tt.exp(-delta) + 1. - self.theta_2 +
            (self.theta_2 - self.theta_1) * tt.exp(-delta / 2) * i0(delta / 2)
            )
        )

    def log_jacobian_term(self, tmp_ctrl_var):
        return 0.


class SigmoidalControlFunction(object):

    def __init__(self, scale):
        assert scale > 0, 'scale should be a positive scalar.'
        self.scale = scale

    def inv_temp_func(self, tmp_ctrl_var):
        u = tmp_ctrl_var / self.scale
        return tt.nnet.sigmoid(u)

    def inv_temp_cond_prob_0_1(self, delta):
        prob_0 = -delta / tt.expm1(-delta)
        prob_1 = delta / tt.expm1(delta)
        return prob_0, prob_1

    def inv_temp_cond_prob_func(self, inv_temp, delta):
        return -tt.exp(-inv_temp * delta) * delta / tt.expm1(-delta)

    def log_jacobian_term(self, tmp_ctrl_var):
        u = tmp_ctrl_var / self.scale
        return (
            tt.log(tt.nnet.sigmoid(u)) + tt.log(1. - tt.nnet.sigmoid(u))
        )
