import numpy as np
from scipy.optimize import differential_evolution


class ParameterRegistry:
    """
    Defines the set of parameters to be calibrated, their bounds,
    and how to get/set them on the model elements. Each registered
    parameter is linked directly to the element it belongs to via
    a getter/setter pair, so the calibrator can update the model
    without knowing anything about its internal structure.
    """
    def __init__(self):
        self._parameters = []

    def register(self, name, element, parameter_key, bounds):
        """
        Registers a reservoir parameter for calibration.

        Parameters
        ----------
        name : str
            Human-readable name used in results (e.g. 'R1_S_ref').
        element : BaseReservoir
            The reservoir element that owns this parameter.
        parameter_key : str
            The parameter key as used in the reservoir's parameter
            dict (e.g. 'S_ref', 'b', 'IN_bar').
        bounds : tuple(float, float)
            Lower and upper bounds (min, max) for the parameter.
        """
        self._parameters.append({
            'name':          name,
            'element':       element,
            'parameter_key': parameter_key,
            'bounds':        bounds
        })

    def get_bounds(self):
        """Returns the list of (min, max) bounds for all parameters."""
        return [p['bounds'] for p in self._parameters]

    def get_names(self):
        """Returns the list of parameter names."""
        return [p['name'] for p in self._parameters]

    def get_values(self):
        """Returns the current parameter values from all elements."""
        return [p['element'].get_parameter(p['parameter_key'])
                for p in self._parameters]

    def set_values(self, values):
        """
        Sets all parameter values on their respective elements.

        Parameters
        ----------
        values : list or numpy.ndarray
            Parameter values in the same order as registration.
        """
        for p, v in zip(self._parameters, values):
            p['element'].set_parameter(p['parameter_key'], v)

    def __len__(self):
        return len(self._parameters)


class Calibrator:
    """
    Calibrates a hydrological model by finding the parameter set that
    optimizes a given objective function. Uses differential evolution
    as the default global optimization algorithm, which is robust to
    multimodal objective function landscapes typical of hydrological
    models.

    The calibrator is agnostic to the model structure — it interacts
    with the model exclusively through a user-defined run function and
    a ParameterRegistry that handles all parameter get/set operations.
    """

    def __init__(self, run_function, parameter_registry,
                 objective_function, warmup=0,
                 algorithm='differential_evolution',
                 algorithm_options=None):
        """
        Parameters
        ----------
        run_function : callable
            A zero-argument function that runs the model with its
            current parameter values and returns the model output
            as a tuple of timeseries (e.g. (Q_sim, C_sim)). The
            calibrator will call registry.set_values() before each
            call to run_function, so the function should simply run
            the model and return its output without worrying about
            parameter setting.
        parameter_registry : ParameterRegistry
            Registry of parameters to calibrate, their bounds, and
            their getter/setter pairs linking them to model elements.
        objective_function : callable
            A function of the form f(observed, simulated) -> float
            to be minimized. observed and simulated are both tuples
            of timeseries in the same order as returned by
            run_function. A composite objective function combining
            multiple metrics should be constructed externally and
            passed here as a single callable.
        warmup : int
            Number of timesteps to exclude from the objective function
            evaluation at the start of the simulation, to allow the
            model to reach a realistic state from initial conditions.
        algorithm : str
            Optimization algorithm to use. Currently supported:
            'differential_evolution' (default).
        algorithm_options : dict or None
            Additional keyword arguments passed directly to the
            optimization algorithm. For differential_evolution, see
            scipy.optimize.differential_evolution for available
            options (e.g. maxiter, popsize, tol, seed, workers).
            If None, scipy defaults are used.
        """
        self.run_function = run_function
        self.registry = parameter_registry
        self.objective_function = objective_function
        self.warmup = warmup
        self.algorithm = algorithm
        self.algorithm_options = algorithm_options or {}

        # Calibration results, populated after calibrate() is called
        self.best_parameters = None
        self.best_score = None
        self.convergence_history = []
        self.result = None

    def _objective_wrapper(self, parameter_values, observations):
        """
        Internal wrapper called by the optimizer at each iteration.
        Sets the candidate parameter values on the model, runs it,
        and returns the objective function score.

        Parameters
        ----------
        parameter_values : numpy.ndarray
            Candidate parameter vector proposed by the optimizer.
        observations : tuple(numpy.ndarray)
            Tuple of observed timeseries in the same order as the
            model output returned by run_function.

        Returns
        -------
        float
            Objective function score to be minimized.
        """
        # Update model parameters
        self.registry.set_values(parameter_values)

        # Run model
        simulated = self.run_function()

        # Trim warmup period from both observed and simulated
        obs_trimmed = tuple(o[self.warmup:] for o in observations)
        sim_trimmed = tuple(s[self.warmup:] for s in simulated)

        # Evaluate and record objective function
        score = self.objective_function(obs_trimmed, sim_trimmed)
        self.convergence_history.append(score)

        return score

    def calibrate(self, observations):
        """
        Runs the calibration procedure and stores the results.

        Parameters
        ----------
        observations : tuple(numpy.ndarray)
            Tuple of observed timeseries against which the model is
            calibrated, in the same order as the model output returned
            by run_function (e.g. (Q_obs, C_obs)).

        Returns
        -------
        best_parameters : dict(str, float)
            Dictionary mapping parameter names to their optimal values.
        best_score : float
            Objective function value at the optimal parameter set.
        """
        bounds = self.registry.get_bounds()
        names = self.registry.get_names()

        if self.algorithm == 'differential_evolution':
            self.result = differential_evolution(
                func=self._objective_wrapper,
                bounds=bounds,
                args=(observations,),
                **self.algorithm_options
            )
        else:
            raise NotImplementedError(
                f'Algorithm "{self.algorithm}" is not yet implemented. '
                f'Currently supported: "differential_evolution".'
            )

        # Store and return results
        self.best_parameters = dict(zip(names, self.result.x))
        self.best_score = self.result.fun

        # Apply the best parameter set to the model
        self.registry.set_values(self.result.x)

        return self.best_parameters, self.best_score

    def summary(self):
        """
        Prints a summary of the calibration results.
        """
        if self.best_parameters is None:
            print('Calibration has not been run yet.')
            return

        print(f'Calibration completed.')
        print(f'Best score : {self.best_score:.6f}')
        print(f'Converged  : {self.result.success}')
        print(f'Iterations : {self.result.nit}')
        print(f'Parameters :')
        for name, value in self.best_parameters.items():
            print(f'  {name:<20} {value:.6f}')
