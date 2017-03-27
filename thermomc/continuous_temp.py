# -*- coding: utf-8 -*-
""" Continuous temperature thermodynamic Monte Carlo algorithms. """

import numpy as np
import theano as th
import theano.tensor as tt
from thermomc.hmc import HamiltonianSampler


class GibbsContinuousTemperingSampler(HamiltonianSampler):
    """Gibbs updates continuous tempering sampler.

    Performs alternating updates of position state given inverse temperature
    and inverse temperature given position state akin to standard simulating
    tempering set up of Marinari and Parisi (1992) but with continuous inverse
    temperature formulation.

    Position state update uses Hamiltonian Monte Carlo transition which leaves
    conditional invariant.

    Inverse temperature update independently samples from truncated exponential
    conditional on inverse temperatures.

    Normalised conditional probabilities of zero and unity inverse temperatures
    given position state samples are calculated to allow 'Rao-Blackwellized' /
    importance sampling estimates of normalising constant (and expectations
    with respect to the target distribution) as proposed in Carlson et al.
    (2016).

    References:

    1. Marinari, E. and Parisi, G. Simulated Tempering: A New Monte Carlo
       Scheme, 1992.
    2. Carlson, D.E., Stinson, P., Pakman A. and Paninski L. Partition
       Functions from Rao-Blackwellized Tempered Sampling, 2016.
    """

    def chain(self, pos_init, mom_init, inv_temp_init, phi_func, psi_func,
              n_sample, hmc_params):
        # Define energy function
        def energy_func(pos, inv_temp, return_components):
            phi = phi_func(pos)
            psi = psi_func(pos)
            energy = inv_temp * phi + (1 - inv_temp) * psi
            if return_components:
                return energy, [phi, psi]
            else:
                return energy
        if mom_init is None:
            mom_init = self.srng.normal(size=pos_init.shape)
        energy_init, [phi_init, psi_init] = energy_func(
            pos_init, inv_temp_init, True
        )
        (pos_samples, mom_samples, inv_temp_samples, _, _, _,
         probs_0, probs_1, accepts), updates = th.scan(
            fn=lambda pos, mom, inv_temp, energy, phi, psi: self.transition(
                pos, mom, inv_temp, energy, phi, psi, energy_func, hmc_params),
            outputs_info=[
                pos_init, mom_init, inv_temp_init,
                energy_init, phi_init, psi_init,
                None, None, None],
            n_steps=n_sample
        )
        return (pos_samples, inv_temp_samples, probs_0, probs_1,
                accepts.mean(0), updates)

    def sample_inv_temp(self, delta):
        """Sample inverse temperature from trunc. exponential conditional."""
        uni_rnd = self.srng.uniform(delta.shape)
        abs_delta = abs(delta)
        # if delta ~ 0 to floating point precision, return uniform otherwise
        # sample from truncated exponential with rate parameter delta
        return tt.switch(
            abs_delta < np.finfo(th.config.floatX).tiny, uni_rnd,
            tt.cast((delta < 0) - tt.sgn(delta) *
                    tt.log1p(tt.expm1(-abs_delta) * uni_rnd) / abs_delta,
                    th.config.floatX)
        )

    def transition(self, pos, mom, inv_temp, energy, phi, psi, energy_func,
                   hmc_params):
        """Update positions given inv. temp and inv. temp. given positions."""
        pos_new, mom_new, _, _, accept, _, _, phi_new, psi_new = (
            self.hmc_transition(
                pos, mom, lambda p, r=False: energy_func(p, inv_temp, r),
                energy_init=energy, comp_init=[phi, psi],
                return_energy_components=True,
                **hmc_params
            )
        )
        delta_new = phi_new - psi_new
        inv_temp_new = self.sample_inv_temp(delta_new)
        energies_new = inv_temp_new * phi_new + (1 - inv_temp_new) * psi_new
        probs_0 = tt.switch(tt.eq(delta_new,  0.),
                            tt.ones_like(delta_new),
                            -delta_new / tt.expm1(-delta_new))
        probs_1 = tt.switch(tt.eq(delta_new, 0.),
                            tt.ones_like(delta_new),
                            delta_new / tt.expm1(delta_new))
        return [pos_new, mom_new, inv_temp_new,
                energies_new, phi_new, psi_new,
                probs_0, probs_1, accept]


class JointContinuousTemperingSampler(HamiltonianSampler):
    """Joint updates continuous tempering sampler.

    Performs joint updates of unconstrained inverse temperature control
    variable (which is mapped via a smooth function to an inverse temperature
    in [0, 1]) and position state using Hamiltonian Monte Carlo updates in
    extended Hamiltonian system.

    References:

    1. Gobbo, G. and Leimkuhler, B. Extended Hamiltonian approach to
       continuous tempering, 2015.
    2. Graham, M.M. and Storkey, A.J. Continuously tempered Hamiltonian Monte
       Carlo, 2016.
    """

    def chain(self, pos_init, tmp_ctrl_init, mom_init, phi_func, psi_func,
              control_func, n_sample, hmc_params):
        # Define energy function
        def energy_func(aug_pos, return_components=False):
            tmp_ctrl, pos = aug_pos[:, 0], aug_pos[:, 1:]
            inv_temp = control_func.inv_temp_func(tmp_ctrl)
            phi = phi_func(pos)
            psi = psi_func(pos)
            energy = (
                inv_temp * phi + (1 - inv_temp) * psi -
                control_func.log_jacobian_term(tmp_ctrl)
            )
            if return_components:
                return energy, [phi, psi]
            else:
                return energy
        aug_pos_init = tt.concatenate([tmp_ctrl_init[:, None], pos_init], 1)
        if mom_init is None:
            mom_init = self.srng.normal(size=aug_pos_init.shape)
        energy_init, [phi_init, psi_init] = energy_func(
            aug_pos_init, True)
        (aug_pos_samples, _, _, _, _,
         probs_0, probs_1, accepts), updates = th.scan(
            fn=lambda aug_pos, mom, energy, phi, psi: self.transition(
                aug_pos, mom, energy, phi, psi, energy_func, control_func,
                hmc_params),
            outputs_info=[
                aug_pos_init, mom_init, energy_init, phi_init, psi_init,
                None, None, None],
            n_steps=n_sample
        )
        tmp_ctrl_samples = aug_pos_samples[:, :, 0]
        inv_temp_samples = control_func.inv_temp_func(tmp_ctrl_samples)
        pos_samples = aug_pos_samples[:, :, 1:]
        return (pos_samples, tmp_ctrl_samples, inv_temp_samples,
                probs_0, probs_1, accepts.mean(0), updates)

    def transition(self, pos, mom, energy, phi, psi, energy_func,
                   control_func, hmc_params):
        pos_new, mom_new, _, energies_new, accept, _, _, phi_new, psi_new = (
            self.hmc_transition(
                pos, mom, energy_func,
                energy_init=energy, comp_init=[phi, psi],
                return_energy_components=True,
                **hmc_params
            )
        )
        delta_new = phi_new - psi_new
        probs_0, probs_1 = control_func.inv_temp_cond_prob_0_1(delta_new)
        return [pos_new, mom_new, energies_new, phi_new, psi_new,
                probs_0, probs_1, accept]
