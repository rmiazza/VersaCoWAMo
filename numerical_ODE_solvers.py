"""
Different numerical solvers for solving the first-order ordinary differential equation:
    dS/dt = P - ET - Q
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp, quad


class ForwardEulerSolver:
    def __init__(self):
        """
        Initializes the Forward Euler solver.
        """
        pass

    def solve(self, f, input, y0, dt, ET_func=None, ETp=None, **kwargs):
        """
        Solves a first-order ODE of the form dS/dt = f(S, IN, **kwargs)
        using the Forward Euler method. Output flux is derived from the
        mass balance at each timestep. If an ET function is provided,
        ET is evaluated a posteriori at each timestep from the storage
        state and returned as a separate timeseries.

        Parameters
        ----------
        f : callable
            ODE function returning dS/dt as a float. Must accept (S, IN)
            as positional arguments plus any additional kwargs.
        input : numpy.ndarray
            Net input flux timeseries of length t.
        y0 : float
            Initial storage.
        dt : float
            Timestep size, in the same time units as the input fluxes.
        ET_func : callable or None
            ET function returning ET as a float given (S, ETp). If None,
            no ET timeseries is computed or returned.
        ETp : numpy.ndarray or None
            Potential ET timeseries of length t. Required if ET_func
            is provided.
        **kwargs : optional
            Additional timeseries or scalar arguments passed to f at each
            timestep. Timeseries (arrays of length t) are indexed per timestep;
            scalars are passed as-is. These are only needed for ODE functions
            that require arguments beyond S and IN that cannot be captured via
            closure — for example, time-varying parameters that must be passed
            as external timeseries rather than defined inside _define_ode.
            In most cases, reservoir parameters are captured via closure in
            _define_ode and kwargs will be empty.

        Returns
        -------
        S_values : numpy.ndarray
            Storage timeseries of length t+1 (includes initial condition).
        discharge_values : numpy.ndarray
            Output flux timeseries of length t, derived from mass balance.
        ET_values : numpy.ndarray or None
            ET timeseries of length t if ET_func was provided, else None.
        """
        n_steps = len(input)
        S_values = np.zeros(n_steps + 1)
        S_values[0] = y0

        # Separate timeseries kwargs from scalar kwargs
        ts_kwargs = {k: v for k, v in kwargs.items() if hasattr(v, '__len__')}
        scalar_kwargs = {k: v for k, v in kwargs.items() if not hasattr(v, '__len__')}

        compute_ET_internal = ET_func is not None
        if compute_ET_internal:
            ET_values = np.zeros(n_steps)

        for i in range(n_steps):
            step_kwargs = {k: v[i] for k, v in ts_kwargs.items()}
            step_kwargs.update(scalar_kwargs)

            # Add ETp to ODE kwargs if computing ET internally
            if compute_ET_internal:
                step_kwargs['ETp'] = ETp[i]

            # Advance storage
            S_values[i+1] = S_values[i] + dt * f(S_values[i], input[i], **step_kwargs)

            # Evaluate ET a posteriori from current storage state
            if compute_ET_internal:
                ET_values[i] = ET_func(S_values[i], ETp[i])

        # Derive output flux from mass balance
        if compute_ET_internal:
            discharge_values = input - ET_values - (S_values[1:] - S_values[:-1]) / dt
            return S_values, discharge_values, ET_values
        else:
            discharge_values = input - (S_values[1:] - S_values[:-1]) / dt
            return S_values, discharge_values, None


class BackwardEulerSolver:
    def __init__(self):
        """
        Initializes the Backward Euler solver.
        """
        pass

    def solve(self, f, input, y0, dt, ET_func=None, ETp=None, **kwargs):
        """
        Solves a first-order ODE of the form dS/dt = f(S, IN, **kwargs)
        using the Backward Euler method. Unlike the Forward Euler method,
        the ODE is evaluated at the unknown future state S[i+1], making
        the scheme implicit and requiring a nonlinear solve at each
        timestep. This makes it unconditionally stable at the cost of
        one nonlinear solve per timestep. Output flux is derived from
        the mass balance at each timestep. If an ET function is provided,
        ET is evaluated a posteriori at each timestep from the storage
        state and returned as a separate timeseries.

        Parameters
        ----------
        f : callable
            ODE function returning dS/dt as a float. Must accept (S, IN)
            as positional arguments plus any additional kwargs.
        input : numpy.ndarray
            Net input flux timeseries of length t.
        y0 : float
            Initial storage.
        dt : float
            Timestep size, in the same time units as the input fluxes.
        ET_func : callable or None
            ET function returning ET as a float given (S, ETp). If None,
            no ET timeseries is computed or returned.
        ETp : numpy.ndarray or None
            Potential ET timeseries of length t. Required if ET_func
            is provided.
        **kwargs : optional
            Additional timeseries or scalar arguments passed to f at each
            timestep. Timeseries (arrays of length t) are indexed per
            timestep; scalars are passed as-is. These are only needed for
            ODE functions that require arguments beyond S and IN that
            cannot be captured via closure — for example, time-varying
            parameters that must be passed as external timeseries rather
            than defined inside _define_ode. In most cases, reservoir
            parameters are captured via closure in _define_ode and kwargs
            will be empty.

        Returns
        -------
        S_values : numpy.ndarray
            Storage timeseries of length t+1 (includes initial condition).
        discharge_values : numpy.ndarray
            Output flux timeseries of length t, derived from mass balance.
        ET_values : numpy.ndarray or None
            ET timeseries of length t if ET_func was provided, else None.
        """
        n_steps = len(input)
        S_values = np.zeros(n_steps + 1)
        S_values[0] = y0

        # Separate timeseries kwargs from scalar kwargs
        ts_kwargs = {k: v for k, v in kwargs.items() if hasattr(v, '__len__')}
        scalar_kwargs = {k: v for k, v in kwargs.items() if not hasattr(v, '__len__')}

        compute_ET_internal = ET_func is not None
        if compute_ET_internal:
            ET_values = np.zeros(n_steps)

        for i in range(n_steps):
            step_kwargs = {k: v[i] for k, v in ts_kwargs.items()}
            step_kwargs.update(scalar_kwargs)

            if compute_ET_internal:
                step_kwargs['ETp'] = ETp[i]

            # Define the implicit equation to solve:
            # S[i+1] - S[i] - dt * f(S[i+1], IN[i], **step_kwargs) = 0
            def implicit_equation(S_next):
                return S_next - S_values[i] - dt * f(S_next, input[i], **step_kwargs)

            # Solve for S[i+1] using S[i] as initial guess
            S_values[i+1] = fsolve(implicit_equation, S_values[i])[0]

            # Evaluate ET a posteriori from the updated storage state
            if compute_ET_internal:
                ET_values[i] = ET_func(S_values[i+1], ETp[i])

        # Derive output flux from mass balance
        if compute_ET_internal:
            discharge_values = input - ET_values - (S_values[1:] - S_values[:-1]) / dt
            return S_values, discharge_values, ET_values
        else:
            discharge_values = input - (S_values[1:] - S_values[:-1]) / dt
            return S_values, discharge_values, None


class RungeKutta4Solver:
    def __init__(self):
        """
        Initializes the Runge-Kutta 4 solver.
        """
        pass

    def solve(self, f, input, y0, dt, ET_func=None, ETp=None, **kwargs):
        """
        Solves a first-order ODE of the form dS/dt = f(S, IN, **kwargs)
        using the fourth-order Runge-Kutta method (RK4). The scheme
        estimates the derivative at four points within each timestep
        (start, two midpoints, end) and combines them in a weighted
        average, achieving fourth-order accuracy without requiring a
        nonlinear solve. Output flux is derived from the mass balance
        at each timestep. If an ET function is provided, ET is evaluated
        a posteriori at each timestep from the storage state and returned
        as a separate timeseries.

        Parameters
        ----------
        f : callable
            ODE function returning dS/dt as a float. Must accept (S, IN)
            as positional arguments plus any additional kwargs.
        input : numpy.ndarray
            Net input flux timeseries of length t.
        y0 : float
            Initial storage.
        dt : float
            Timestep size, in the same time units as the input fluxes.
        ET_func : callable or None
            ET function returning ET as a float given (S, ETp). If None,
            no ET timeseries is computed or returned.
        ETp : numpy.ndarray or None
            Potential ET timeseries of length t. Required if ET_func
            is provided.
        **kwargs : optional
            Additional timeseries or scalar arguments passed to f at each
            timestep. Timeseries (arrays of length t) are indexed per
            timestep; scalars are passed as-is. These are only needed for
            ODE functions that require arguments beyond S and IN that
            cannot be captured via closure — for example, time-varying
            parameters that must be passed as external timeseries rather
            than defined inside _define_ode. In most cases, reservoir
            parameters are captured via closure in _define_ode and kwargs
            will be empty.

        Returns
        -------
        S_values : numpy.ndarray
            Storage timeseries of length t+1 (includes initial condition).
        discharge_values : numpy.ndarray
            Output flux timeseries of length t, derived from mass balance.
        ET_values : numpy.ndarray or None
            ET timeseries of length t if ET_func was provided, else None.
        """
        n_steps = len(input)
        S_values = np.zeros(n_steps + 1)
        S_values[0] = y0

        # Separate timeseries kwargs from scalar kwargs
        ts_kwargs = {k: v for k, v in kwargs.items() if hasattr(v, '__len__')}
        scalar_kwargs = {k: v for k, v in kwargs.items() if not hasattr(v, '__len__')}

        compute_ET_internal = ET_func is not None
        if compute_ET_internal:
            ET_values = np.zeros(n_steps)

        for i in range(n_steps):
            step_kwargs = {k: v[i] for k, v in ts_kwargs.items()}
            step_kwargs.update(scalar_kwargs)

            if compute_ET_internal:
                step_kwargs['ETp'] = ETp[i]

            S_i = S_values[i]
            IN_i = input[i]

            # Four RK4 slope estimates — input flux and kwargs are held
            # constant within the timestep (piecewise constant assumption)
            k1 = f(S_i, IN_i, **step_kwargs)
            k2 = f(S_i + dt/2 * k1, IN_i, **step_kwargs)
            k3 = f(S_i + dt/2 * k2, IN_i, **step_kwargs)
            k4 = f(S_i + dt * k3, IN_i, **step_kwargs)

            # Weighted average of slopes
            S_values[i+1] = S_i + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

            # Evaluate ET a posteriori from the mean storage over the timestep,
            # consistent with the RK4 intermediate states
            if compute_ET_internal:
                ET_values[i] = (
                    (ET_func(S_i, ETp[i]) +
                    2*ET_func(S_i + dt/2*k1, ETp[i]) +
                    2*ET_func(S_i + dt/2*k2, ETp[i]) +
                    ET_func(S_i + dt*k3, ETp[i]))
                    / 6
                )

        # Derive output flux from mass balance
        if compute_ET_internal:
            discharge_values = input - ET_values - (S_values[1:] - S_values[:-1]) / dt
            return S_values, discharge_values, ET_values
        else:
            discharge_values = input - (S_values[1:] - S_values[:-1]) / dt
            return S_values, discharge_values, None


class ScipySolver:
    def __init__(self, method='Radau', rtol=1e-6, atol=1e-6):
        """
        Initializes the Scipy ODE solver.

        Parameters
        ----------
        method : str
            Integration method passed to scipy.integrate.solve_ivp.
            Recommended options:
            - 'Radau' : implicit, 5th order, unconditionally stable,
              best for stiff problems (default).
            - 'RK45'  : explicit, 4th-5th order adaptive, best for
              smooth non-stiff problems.
            - 'LSODA' : automatic stiffness detection, switches between
              explicit and implicit methods.
        rtol : float
            Relative tolerance for the adaptive step size control.
        atol : float
            Absolute tolerance for the adaptive step size control.
        """
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def solve(self, f, input, y0, dt, ET_func=None, ETp=None, **kwargs):
        """
        Solves a first-order ODE of the form dS/dt = f(S, IN, **kwargs)
        using scipy.integrate.solve_ivp with adaptive stepping. The solver
        internally uses variable substeps to meet the prescribed tolerances,
        then returns storage values at the fixed output times defined by dt.
        Output flux is derived from the mass balance at each timestep. If
        an ET function is provided, ET is evaluated a posteriori at each
        timestep and returned as a separate timeseries.

        Parameters
        ----------
        f : callable
            ODE function returning dS/dt as a float. Must accept (S, IN)
            as positional arguments plus any additional kwargs.
        input : numpy.ndarray
            Net input flux timeseries of length t.
        y0 : float
            Initial storage.
        dt : float
            Timestep size, in the same time units as the input fluxes.
            Defines the output times only — internal adaptive substeps
            are determined by the solver to meet rtol and atol.
        ET_func : callable or None
            ET function returning ET as a float given (S, ETp). If None,
            no ET timeseries is computed or returned.
        ETp : numpy.ndarray or None
            Potential ET timeseries of length t. Required if ET_func
            is provided.
        **kwargs : optional
            Additional timeseries or scalar arguments passed to f at each
            timestep. Timeseries (arrays of length t) are indexed per
            timestep; scalars are passed as-is. These are only needed for
            ODE functions that require arguments beyond S and IN that
            cannot be captured via closure — for example, time-varying
            parameters that must be passed as external timeseries rather
            than defined inside _define_ode. In most cases, reservoir
            parameters are captured via closure in _define_ode and kwargs
            will be empty.

        Returns
        -------
        S_values : numpy.ndarray
            Storage timeseries of length t+1 (includes initial condition).
        discharge_values : numpy.ndarray
            Output flux timeseries of length t, derived from mass balance.
        ET_values : numpy.ndarray or None
            ET timeseries of length t if ET_func was provided, else None.

        Raises
        ------
        RuntimeError
            If solve_ivp fails to integrate a timestep successfully.
        """
        n_steps = len(input)
        S_values = np.zeros(n_steps + 1)
        S_values[0] = y0

        # Separate timeseries kwargs from scalar kwargs
        ts_kwargs = {k: v for k, v in kwargs.items() if hasattr(v, '__len__')}
        scalar_kwargs = {k: v for k, v in kwargs.items() if not hasattr(v, '__len__')}

        compute_ET_internal = ET_func is not None
        if compute_ET_internal:
            ET_values = np.zeros(n_steps)

        for i in range(n_steps):
            step_kwargs = {k: v[i] for k, v in ts_kwargs.items()}
            step_kwargs.update(scalar_kwargs)

            if compute_ET_internal:
                step_kwargs['ETp'] = ETp[i]

            def ode_ivp(t, S):
                return [f(S[0], input[i], **step_kwargs)]

            # Request dense output to get the continuous S(t) interpolant
            result = solve_ivp(
                fun=ode_ivp,
                t_span=(0, dt),
                y0=[S_values[i]],
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
                dense_output=True  # enables the continuous interpolant
            )

            if not result.success:
                raise RuntimeError(
                    f'solve_ivp failed at timestep {i}: {result.message}'
                )

            S_values[i+1] = result.y[0, -1]

            # Evaluate average ET over the timestep by integrating ET_func(S(t))
            # over [0, dt] using the dense output interpolant, then dividing by dt
            if compute_ET_internal:
                def ET_integrand(t):
                    S_t = result.sol(t)[0]  # continuous S(t) from dense output
                    return ET_func(S_t, ETp[i])

                ET_integrated, _ = quad(ET_integrand, 0, dt)
                ET_values[i] = ET_integrated / dt  # average rate over the timestep

        # Derive output flux from mass balance
        if compute_ET_internal:
            discharge_values = input - ET_values - (S_values[1:] - S_values[:-1]) / dt
            return S_values, discharge_values, ET_values
        else:
            discharge_values = input - (S_values[1:] - S_values[:-1]) / dt
            return S_values, discharge_values, None
