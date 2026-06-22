import numpy as np
import emcee


class MCMCSampler:
    """
    Bayesian parameter estimation for hydrological models using the
    affine invariant ensemble sampler (emcee). Produces posterior
    distributions over model parameters, enabling uncertainty
    quantification and model performance assessment.

    Intended to be used after a preliminary calibration with the
    Calibrator class, using the calibrated parameter set to
    initialize the walkers in a region of high posterior probability.

    The sampler is agnostic to the model structure — it interacts
    with the model exclusively through a user-defined run function
    and a ParameterRegistry that handles all parameter get/set
    operations.
    """

    def __init__(self, run_function, parameter_registry,
                 log_likelihood, log_prior,
                 n_walkers=32, n_steps=5000, warmup=0):
        """
        Parameters
        ----------
        run_function : callable
            A zero-argument function that runs the model with its
            current parameter values and returns the model output
            as a tuple of timeseries (Q_sim, C_sim, TTD_sim). The
            sampler will call registry.set_values() before each
            call to run_function.
        parameter_registry : ParameterRegistry
            Registry of parameters to sample, their bounds, and
            their getter/setter pairs linking them to model elements.
        log_likelihood : callable
            A function of the form f(observed, simulated) -> float
            returning the log-likelihood of the observed data given
            the simulated output. Should return -inf for degenerate
            or unstable model runs.
        log_prior : callable
            A function of the form f(parameter_values, bounds) -> float
            returning the log-prior probability of the parameter
            vector. Should return -inf for parameter values outside
            their bounds.
        n_walkers : int
            Number of ensemble walkers. Must be even and at least
            2 * n_parameters. More walkers improve mixing but
            increase cost per step. Default is 32.
        n_steps : int
            Number of MCMC steps per walker. Total number of model
            evaluations is n_walkers * n_steps. Default is 5000.
        warmup : int
            Number of timesteps to exclude from the likelihood
            evaluation at the start of the simulation, to allow
            the model to reach a realistic state from initial
            conditions.
        """
        self.run_function = run_function
        self.registry = parameter_registry
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.warmup = warmup

        # Populated after sample() is called
        self.sampler = None
        self._observations = None

    def _log_posterior(self, parameter_values, observations):
        """
        Computes the log-posterior probability for a given parameter
        vector. Returns -inf immediately if the prior is -inf, avoiding
        unnecessary model runs for out-of-bounds parameters.

        Parameters
        ----------
        parameter_values : numpy.ndarray
            Candidate parameter vector proposed by a walker.
        observations : numpy.ndarray
            Observed timeseries as a plain array.

        Returns
        -------
        float
            Log-posterior probability. Returns -inf for invalid
            parameter sets or numerically unstable model runs.
        """
        # Evaluate prior first — skip model run if outside bounds
        bounds = self.registry.get_bounds()
        lp = self.log_prior(parameter_values, bounds)
        if not np.isfinite(lp):
            return -np.inf

        # Update model parameters and run
        self.registry.set_values(parameter_values)

        try:
            simulated = self.run_function()
        except Exception:
            return -np.inf

        # Check for numerical instability on the plain array
        # for discharge, could add: or np.any(simulated < 0)
        if not np.all(np.isfinite(simulated)):  
            return -np.inf

        # Trim warmup period from both observed and simulated
        obs_trimmed = observations[self.warmup:]
        sim_trimmed = simulated[self.warmup:]

        # Evaluate log-likelihood
        ll = self.log_likelihood(obs_trimmed, sim_trimmed)
        if not np.isfinite(ll):
            return -np.inf

        return lp + ll

    def sample(self, observations, start_from=None, progress=True):
        """
        Runs the MCMC sampling and stores the emcee sampler object.

        Parameters
        ----------
        observations : tuple(numpy.ndarray)
            Tuple of observed timeseries against which the model is
            evaluated, in the same order as the model output returned
            by run_function (e.g. (Q_obs, C_obs)).
        start_from : dict(str, float) or numpy.ndarray or None
            Initial parameter values for the walker ensemble centre.
            If a dict, keys must match the parameter names in the
            registry (e.g. as returned by Calibrator.calibrate()).
            If a numpy array, must have length n_parameters.
            If None, walkers are initialized randomly within bounds.
        progress : bool
            Whether to show a tqdm progress bar. Default is True.

        Returns
        -------
        emcee.EnsembleSampler
            The emcee sampler object after completing all steps.
        """
        self._observations = observations
        n_params = len(self.registry)
        bounds = self.registry.get_bounds()
        names = self.registry.get_names()

        # Validate n_walkers
        if self.n_walkers < 2 * n_params:
            raise ValueError(
                f'n_walkers ({self.n_walkers}) must be at least '
                f'2 * n_parameters ({2 * n_params}).'
            )
        if self.n_walkers % 2 != 0:
            raise ValueError('n_walkers must be even.')

        # Initialize walker positions
        if start_from is None:
            # Random initialization within bounds
            p0 = np.array([
                [np.random.uniform(lo, hi) for lo, hi in bounds]
                for _ in range(self.n_walkers)
            ])
        else:
            # Initialize in a tight Gaussian ball around the starting point
            if isinstance(start_from, dict):
                centre = np.array([start_from[name] for name in names])
            else:
                centre = np.asarray(start_from)

            # Ball width is 0.1% of each parameter's range
            widths = np.array([hi - lo for lo, hi in bounds]) * 1e-3
            p0 = centre + widths * np.random.randn(self.n_walkers, n_params)

            # Clip to bounds to ensure all walkers start within prior
            for j, (lo, hi) in enumerate(bounds):
                p0[:, j] = np.clip(p0[:, j], lo, hi)

        # Build and run the sampler
        self.sampler = emcee.EnsembleSampler(
            nwalkers=self.n_walkers,
            ndim=n_params,
            log_prob_fn=self._log_posterior,
            args=(observations,)
        )

        self.sampler.run_mcmc(p0, self.n_steps, progress=progress)

        return self.sampler

    def get_flat_samples(self, discard=500, thin=10):
        """
        Returns the flattened chain after removing burn-in and
        thinning to reduce autocorrelation.

        Parameters
        ----------
        discard : int
            Number of initial steps to discard as burn-in.
        thin : int
            Thinning factor — keep one sample every `thin` steps.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_samples, n_parameters).
        """
        if self.sampler is None:
            raise RuntimeError('No samples available. Run sample() first.')
        return self.sampler.get_chain(discard=discard, thin=thin, flat=True)

    def get_parameter_distributions(self, discard=500, thin=10):
        """
        Returns the posterior samples for each parameter as a
        dictionary, suitable for plotting or further analysis.

        Parameters
        ----------
        discard : int
            Number of initial steps to discard as burn-in.
        thin : int
            Thinning factor.

        Returns
        -------
        dict(str, numpy.ndarray)
            Dictionary mapping parameter names to their posterior
            sample arrays.
        """
        flat_samples = self.get_flat_samples(discard=discard, thin=thin)
        return {
            name: flat_samples[:, i]
            for i, name in enumerate(self.registry.get_names())
        }

    def get_autocorrelation_time(self):
        """
        Estimates the integrated autocorrelation time for each
        parameter, which indicates the number of steps needed for
        independent samples. A rule of thumb is that the chain should
        be at least 50 times the autocorrelation time.

        Returns
        -------
        dict(str, float)
            Dictionary mapping parameter names to their estimated
            autocorrelation times.
        """
        if self.sampler is None:
            raise RuntimeError('No samples available. Run sample() first.')
        try:
            tau = self.sampler.get_autocorr_time()
            return dict(zip(self.registry.get_names(), tau))
        except emcee.autocorr.AutocorrError as e:
            print(f'Autocorrelation time could not be reliably estimated: {e}')
            return None

    def summary(self, discard=500, thin=10, credible_interval=0.95):
        """
        Prints a summary of the posterior distributions, including
        the median and credible intervals for each parameter.

        Parameters
        ----------
        discard : int
            Number of initial steps to discard as burn-in.
        thin : int
            Thinning factor.
        credible_interval : float
            Width of the credible interval to report, between 0 and 1.
            Default is 0.95 (95% credible interval).
        """
        if self.sampler is None:
            print('No samples available. Run sample() first.')
            return

        flat_samples = self.get_flat_samples(discard=discard, thin=thin)
        names = self.registry.get_names()
        alpha = (1 - credible_interval) / 2

        print(f'MCMC Summary ({len(flat_samples)} samples after burn-in and thinning)')
        print(f'{"Parameter":<20} {"Median":>10} {"Lower":>10} {"Upper":>10}')
        print('-' * 52)

        for i, name in enumerate(names):
            samples = flat_samples[:, i]
            median = np.median(samples)
            lower = np.quantile(samples, alpha)
            upper = np.quantile(samples, 1 - alpha)
            print(f'{name:<20} {median:>10.4f} {lower:>10.4f} {upper:>10.4f}')

        # Autocorrelation time
        tau = self.get_autocorrelation_time()
        if tau is not None:
            print()
            print('Autocorrelation times:')
            for name, t in tau.items():
                print(f'  {name:<20} {t:.1f} steps')
            print(f'  (chain length: {self.n_steps} steps, '
                  f'recommend > {50 * max(tau.values()):.0f})')

    def plot_chains(self):
        """
        Plots the walker chains for each parameter as a function of
        step number. Useful for visually assessing convergence and
        identifying the appropriate burn-in length.
        """
        if self.sampler is None:
            raise RuntimeError('No samples available. Run sample() first.')

        import matplotlib.pyplot as plt

        names = self.registry.get_names()
        n_params = len(names)
        chain = self.sampler.get_chain()  # shape (n_steps, n_walkers, n_params)

        fig, axs = plt.subplots(n_params, 1, figsize=(10, 2.5 * n_params), sharex=True)
        if n_params == 1:
            axs = [axs]

        for i, (ax, name) in enumerate(zip(axs, names)):
            ax.plot(chain[:, :, i], c='royalblue', alpha=0.3, lw=0.5)
            ax.set_ylabel(name)

        axs[-1].set_xlabel('Step')
        fig.suptitle('Walker chains', y=1.01)
        fig.tight_layout()
        return fig

    def plot_corner(self, discard=500, thin=10, truths=None):
        """
        Produces a corner plot showing the 1D and 2D marginal
        posterior distributions for all parameters.

        Parameters
        ----------
        discard : int
            Number of initial steps to discard as burn-in.
        thin : int
            Thinning factor.
        truths : dict(str, float) or None
            Optional reference values to overplot on the corner plot
            (e.g. the calibrated parameter set). Keys must match
            parameter names.
        """
        try:
            import corner
        except ImportError:
            raise ImportError(
                'The corner package is required for corner plots. '
                'Install it with: pip install corner'
            )

        import matplotlib.pyplot as plt

        flat_samples = self.get_flat_samples(discard=discard, thin=thin)
        names = self.registry.get_names()

        truth_values = None
        if truths is not None:
            truth_values = [truths[name] for name in names]

        fig = corner.corner(
            flat_samples,
            labels=names,
            truths=truth_values,
            quantiles=[0.025, 0.5, 0.975],
            show_titles=True,
            title_fmt='.4f',
            title_kwargs={'fontsize': 10}
        )

        return fig
