# -*- coding: utf-8 -*-
""" Discrete temperature thermodynamic Monte Carlo algorithms. """

import theano as th
import theano.tensor as tt
import theano.sandbox.rng_mrg as mrg_rand
from thermomc.hmc import HamiltonianSampler


class AnnealedImportanceSampler(HamiltonianSampler):
    """Annealed Importance Sampler (Neal, 1998).

    Forms importance weighted ensemble of samples for target distribution
    given independent samples from base distribution by deterministically
    annealing inverse temperature across a discrete schedule and applying
    Markov transitions at each inverse temperature which leave the
    corresponding bridged distribution invariant.

    Hamiltonian Monte Carlo is used for the inverse temperature dependent
    transitions following the *Hamiltonian Annealed Importance Sampling*
    method of Sohl-Dickstein and Culpepper (2012).

    References:

    1. Neal, R.M. Annealed Importance Sampling, 1998.
    2. Sohl-Dickstein, J. and Culpepper B.J. Hamiltonian Annealed Importance
       Sampling for partition function estimation, 2012.
    """

    def run(self, pos_init, mom_init, inv_temps, phi_func, psi_func,
            hmc_params):
        """Construct symbolic graph corresponding to Hamiltonian AIS run.

        Args:
            pos_init: Initial position state tensor. This should be a 2D
                tensor with first axis corresponding to chain dimension and
                the second axis the state dimension. The tensor should be
                a batch of (usually independent) samples from the base
                distribution.
            mom_init: Initial momentum state tensor. This should be a 2D
                tensor with first axis corresponding to replica / batch
                dimension and the second axis the state dimension. If `None`
                will be independently sampled from Gaussian marginal.
            inv_temps: Inverse temperature schedule tensor. This should be a
                1D tensor of scalars in [0, 1] and will usually be either
                monotonically increasing or decreasing.
            phi_func: Target distribution energy symbolic function. This should
                be a single argument function which takes a 2D tensor input
                and returns a 1D vector corresponding to the negative logarithm
                of the unnormalised density of the target distribution (i.e.
                that corresponding to an inverse temperature of 0).
            psi_func: Base distribution energy symbolic function. This should
                be a single argument function which takes a 2D tensor input
                and returns a 1D vector corresponding to the negative logarithm
                of the *normalised* density of the base distribution (i.e. that
                corresponding to an inverse temperature of 1).
            hmc_params: Dictionary of HMC transition parameters. This should
                specify the integrator step-size `dt`, a positive scalar, the
                number of integrator steps per proposal 'n_step' a positive
                integer scalar and the momentum resampling coefficient
                `mom_resample_coeff`, a scalar in [0, 1] which bridges
                between independent resampling of the momenta from their
                Gaussian marginal at the end of each transition (0) and
                retaining the previous values with no resampling (1).
                Intermediate values produce updated values which are
                correlated to the previous values and leave the Gaussian
                marginal invariant.
        """
        def energy_func(pos, inv_temp):
            return inv_temp * phi_func(pos) + (1 - inv_temp) * psi_func(pos)
        if mom_init is None:
            mom_init = self.srng.normal(size=pos_init.shape)
        energy_init = energy_func(pos_init, inv_temps[0])
        (pos_samples, _, e_beg, e_end, accepts), updates = th.scan(
            fn=lambda inv_temp, pos, mom: self.hmc_transition(
                    pos, mom, lambda pos_: energy_func(pos_, inv_temp),
                    energy_init=None, comp_init=None,
                    return_energy_components=False, **hmc_params),
            sequences=inv_temps[1:-1],
            outputs_info=[pos_init, mom_init, None, None, None]
        )
        energy_final = energy_func(pos_samples[-1], inv_temps[-1])
        log_weights = energy_init - energy_final + (e_end - e_beg).sum(0)
        return pos_samples[-1], log_weights, accepts.mean(0), updates


class SimulatedTemperingSampler(HamiltonianSampler):
    """Simulated tempering sampler (Marinari and Parisi, 1992).

    Performs alternating updates of position state given discrete inverse
    temperature index and inverse temperature index given position state.

    Position state update uses Hamiltonian Monte Carlo transition which leaves
    conditional invariant.

    Inverse temperature index update independently samples indices from
    disrete conditional distribution.

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

    def chain(self, pos_init, mom_init, idx_init, inv_temps, inv_temp_weights,
              phi_func, psi_func, n_sample, hmc_params):
        # Define energy function.
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
            pos_init, inv_temps[idx_init], True)
        (pos_samples, mom_samples, idx_samples, _, _, _,
         probs_0, probs_1, accepts), updates = th.scan(
            fn=lambda pos, mom, idx, energy, phi, psi: self.transition(
                pos, mom, idx, energy, phi, psi, inv_temps, inv_temp_weights,
                energy_func, hmc_params),
            outputs_info=[
                pos_init, mom_init, idx_init,
                energy_init, phi_init, psi_init,
                None, None, None],
            n_steps=n_sample
        )
        return (pos_samples, idx_samples, probs_0, probs_1,
                accepts.mean(0), updates)

    def sample_indices(self, phi, psi, inv_temps, inv_temp_weights):
        energies_all = (
            inv_temps[None, :] * phi[:, None] +
            (1 - inv_temps[None, :]) * psi[:, None]
        )
        probs = tt.nnet.softmax(-energies_all + inv_temp_weights)
        if self.srng isinstance(mrg_rand.MRG_RandomStreams):
            idx, updates = self.srng.choice(
                a=probs.shape[-1], size=1, replace=False, p=probs), {}
        else:
            idx, updates = th.scan(
                fn=lambda p: self.srng.choice(
                    a=p.shape[0], size=[], replace=False, p=p),
                sequences=probs
            )
        energy = energies_all[tt.arange(probs.shape[0]), idx]
        return [idx, energy, probs[:, 0], probs[:, -1]], updates

    def transition(self, pos, mom, idx, energy, phi, psi, inv_temps,
                   inv_temp_weights, energy_func, hmc_params):
        pos_new, mom_new, _, _, accept, _, _, phi_new, psi_new = (
            self.hmc_transition(
                pos, mom, lambda p, r=False: energy_func(p, inv_temps[idx], r),
                energy_init=energy, comp_init=[phi, psi],
                return_energy_components=True,
                **hmc_params
            )
        )
        (idx_new, energies_new, prob_0, prob_1), updates = self.sample_indices(
            phi_new, psi_new, inv_temps, inv_temp_weights
        )
        return [pos_new, mom_new, idx_new,
                energies_new, phi_new, psi_new,
                prob_0, prob_1, accept], updates
