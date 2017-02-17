# -*- coding: utf-8 -*-
""" Base Hamiltonian Monte Carlo algorithm. """

import theano as th
import theano.tensor as tt


class HamiltonianSampler(object):
    """Hamiltonian / Hybrid Monte Carlo sampler (Duane et al. 1987).

    Markov chain Monte Carlo sampler with transition operator constructed from
    simulated Hamiltonian dynamics. Optionally uses partial momentum
    resampling as proposed in Horowitz (1991).

    References:

    1. Duane, S, Kennedy AD, Pendleton, BJ and Roweth, D. Hybrid Monte Carlo,
       1987.
    2. Neal, RM. MCMC using Hamiltonian dynamics, 2012.
    3. Horowitz, AM. A generalized guided Monte Carlo algorithm, 1991.
    """

    def __init__(self, srng, unroll_inner_scan=False):
        """Create a new HamiltonianSampler instance.

        Args:
            srng: Seeded RandomStreams object for random number generation.
            unroll_inner_scan: Whether to 'unroll' inner leapfrog step
                iteration graph operations by explicitly looping and adding
                all iterations to graph or to use scan. Unrolling generally
                gives runtime performance boosts but comes at the cost of
                larger compile times, potentially less efficient memory usage
                and the need to fix the number of leapfrog steps at compile
                time.
        """
        self.srng = srng
        self.unroll_inner_scan = unroll_inner_scan

    def leapfrog_step(self, pos, mom, energy_func, dt):
        """Perform a single full leapfrog update of state.

        Args:
            pos: Position state tensor. This should be a 2D tensor with first
                axis corresponding to replica / batch dimension and the second
                axis the state dimension.
            mom: Momentum state tensor. This should be a 2D tensor with first
                axis corresponding to replica / batch dimension and the second
                axis the state dimension.
            energy_func: Function taking a position tensor as input which
                applies the Theano graph operations corresponding to
                calculating the potential energy value (negative logarithm of
                the unnormalised target density) for each position state
                vector in input, returning a 1D tensor corresponding to
                calculated energy values.
            dt: Scalar tensor / constant representing size of discrete
                timestep.

        Returns:
            Tuple with first entry the Theano tensor / graph node
            corresponding to the updated position tensor and the second the
            updated momentum tensor.
        """
        mom = mom - dt * tt.grad(energy_func(pos).sum(), pos)
        pos = pos + dt * mom
        return pos, mom

    def simulate_dynamic(self, pos, mom, energy_func, dt, n_step):
        """Use leapfrog integrator to simulate Hamiltonian dynamic.

        Args:
            pos: Position state tensor. This should be a 2D tensor with first
                axis corresponding to replica / batch dimension and the second
                axis the state dimension.
            mom: Momentum state tensor. This should be a 2D tensor with first
                axis corresponding to replica / batch dimension and the second
                axis the state dimension.
            energy_func: Function taking a position tensor as input which
                applies the Theano graph operations corresponding to
                calculating the potential energy value (negative logarithm of
                the unnormalised target density) for each position state
                vector in input, returning a 1D tensor corresponding to
                calculated energy values.
            dt: Scalar tensor / constant representing size of discrete
                timestep.
            n_step: Integer scalar indicating number of leapfrog integrator
                steps to simulate forwards.

        Returns:
            Tuple with first entry the Theano tensor / graph node
            corresponding to the updated position tensor and the second the
            updated momentum tensor.
        """
        # Initial half timestep for positions
        pos = pos + 0.5 * dt * mom
        # Perform (n_step - 1) leapfrogging full timesteps
        if self.unroll_inner_scan:
            for s in range(n_step - 1):
                pos, mom = self.leapfrog_step(pos, mom, energy_func, dt)
        else:
            (pos_seq, mom_seq), _ = th.scan(
                fn=lambda pos, mom: self.leapfrog_step(
                    pos, mom, energy_func, dt),
                outputs_info=[pos, mom],
                n_steps=n_step - 1
            )
            pos, mom = pos_seq[-1], mom_seq[-1]
        # Final full timestep for momenta
        mom = mom - dt * tt.grad(energy_func(pos).sum(), pos)
        # Final half timestep for positions
        pos = pos + 0.5 * dt * mom
        # Return negated momentum so proposal is reversible
        return pos, -mom

    def hmc_transition(self, pos_init, mom_init, energy_func, dt, n_step,
                       mom_resample_coeff, energy_init=None, comp_init=None,
                       return_energy_components=False):
        """Perform a full Hamiltonian Monte Carlo transition operation.

        Transition is a sequential concatenation of three reversible operators,
        each of which leaves the joint target density on position and momentum
        states invariant:

          1. Metropolis-Hastings step using simulated dynamic proposal.
          2. Momentum negation.
          3. Partial / full momentum resampling from Gaussian marginal.

        Args:
            pos_init: Initial position state tensor. This should be a 2D
                tensor with first axis corresponding to replica / batch
                dimension and the second axis the state dimension.
            mom_init: Initial momentum state tensor. This should be a 2D
                tensor with first axis corresponding to replica / batch
                dimension and the second axis the state dimension.
            energy_func: Function taking a position tensor as input which
                applies the Theano graph operations corresponding to
                calculating the potential energy value (negative logarithm of
                the unnormalised target density) for each position state
                vector in input, returning a 1D tensor corresponding to
                calculated energy values.
            dt: Scalar tensor / constant representing size of discrete
                timestep.
            n_step: Integer scalar indicating number of leapfrog integrator
                steps to simulate forwards for each proposal.
            mom_resample_coeff: Scalar in [0, 1] controlling degree of
                momentum resampling. A value of 1 indicates independent
                momentum values should be sampled from their Gaussian marginal
                after each dynamics based Metropolis-Hastings transition. A
                value of 0 indicates no resampling of the momenta. An
                intermediate value between 0 and 1 produces partial momentum
                resampling, with the new momentum formed as an autoregressive
                update of the current momentum.

        Returns:
            Tuple with first entry the Theano tensor / graph node
            corresponding to the new position tensor, the second the
            new momentum tensor, the third entry the potential energy
            difference between initial and final position states and the final
            entry a binary tensor specifying whether the Metropolis-Hastings
            update for each chain where accepted.
        """
        if energy_init is None:
            if return_energy_components:
                energy_init, comp_init = energy_func(pos_init, True)
            else:
                energy_init = energy_func(pos_init)
        hamiltonian_init = energy_init + 0.5 * (mom_init**2).sum(-1)
        # 1. Metropolis-Hastings step using simulated  dynamic proposal
        pos_prop, mom_prop = self.simulate_dynamic(
            pos_init, mom_init, energy_func, dt, n_step)
        if return_energy_components:
            energy_prop, comp_prop = energy_func(pos_prop, True)
        else:
            energy_prop = energy_func(pos_prop)
        hamiltonian_prop = energy_prop + 0.5 * (mom_prop**2).sum(-1)
        uni_rnd = self.srng.uniform(energy_prop.shape)
        accept = tt.le(uni_rnd, tt.exp(hamiltonian_init - hamiltonian_prop))
        pos_next = tt.switch(accept[:, None], pos_prop, pos_init)
        mom_next = tt.switch(accept[:, None], mom_prop, mom_init)
        energy_next = tt.switch(accept, energy_prop, energy_init)
        if return_energy_components:
            comp_next = [
                tt.switch(accept, cp, ci)
                for cp, ci in zip(comp_prop, comp_init)
            ]
        # 2. Momentum negation
        mom_next = -mom_next
        # 3. Partial / full momentum resampling from Gaussian marginal
        mom_indp = self.srng.normal(mom_next.shape)
        mom_next = (
            mom_resample_coeff * mom_indp +
            (1. - mom_resample_coeff**2)**0.5 * mom_next
        )
        if not return_energy_components:
            return [
                pos_next, mom_next, energy_init, energy_next,
                tt.cast(accept, th.config.floatX)
            ]
        else:
            return [
                pos_next, mom_next, energy_init, energy_next,
                tt.cast(accept, th.config.floatX)
            ] + comp_init + comp_next

    def hmc_chain(self, pos_init, mom_init, energy_func, n_sample,
                  dt, n_step, mom_resample_coeff):
        """Constructs Markov chains by iteratively applying HMC transitions.

        Args:
            pos_init: Initial position state tensor. This should be a 2D
                tensor with first axis corresponding to chain dimension and
                the second axis the state dimension.
            mom_init: Initial momentum state tensor. This should be a 2D
                tensor with first axis corresponding to replica / batch
                dimension and the second axis the state dimension.
            energy_func: Function taking a position tensor as input which
                applies the Theano graph operations corresponding to
                calculating the potential energy value (negative logarithm of
                the unnormalised target density) for each position state
                vector in input, returning a 1D tensor corresponding to
                calculated energy values.
            dt: Scalar tensor / constant representing size of discrete
                timestep.
            n_step: Integer scalar indicating number of leapfrog integrator
                steps to simulate forwards for each proposal.
            mom_resample_coeff: Scalar in [0, 1] controlling degree of
                momentum resampling. A value of 1 indicates independent
                momentum values should be sampled from their Gaussian marginal
                after each dynamics based Metropolis-Hastings transition. A
                value of 0 indicates no resampling of the momenta. An
                intermediate value between 0 and 1 produces partial momentum
                resampling, with the new momentum formed as an autoregressive
                update of the current momentum.

        Returns:
            Tuple with first entry the Theano tensor / graph node
            corresponding to the new position tensor, the second the
            new momentum tensor, the third entry the potential energy
            difference between initial and final position states and the final
            entry a binary tensor specifying whether the Metropolis-Hastings
            update for each chain where accepted.
        """
        n_batch = pos_init.shape[0]
        energy_init = energy_func(pos_init)
        (pos_samples, mom_samples, _, _, accept_seq), updates = th.scan(
            fn=lambda pos, mom, e_init: self.hmc_transition(
                pos, mom, energy_func, dt, n_step, mom_resample_coeff,
                e_init, None, False),
            outputs_info=[pos_init, mom_init, None, energy_init, None],
            n_steps=n_sample
        )
        return pos_samples, mom_samples, accept_seq.mean(0), updates
