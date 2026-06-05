import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# Different numerical solvers for solving
# the first-order ordinary differential equation
# dS/dt = P - ET - Q


class ForwardEulerSolver:
    def __init__(self):
        """
        Initialize the solver.
        """
        pass

    def solve(self, f, input, params, y0, dt, t0, t_end, compute_ET_internal, **kwargs):
        """
        Solve the ODE using the Forward Euler method.

        Parameters:
        f (function): The function defining the ODE (dy/dt = f(t, y))
        input (numpy.ndarray): The input timeseries to the storage.
        params (dict): The parameters of the discharge law (I_bar, S_ref, b)
        y0 (float): The initial condition at t0
        t0 (float): The initial time
        t_end (float): The end time
        dt (float): The time step size
        compute_ET_internal (bool) : Defines if ET should be computed
        kwargs : Additional arguments for ET computation
        """

        t_values = np.arange(t0, t_end + 1, dt)
        S_values = np.zeros(len(t_values))
        discharge_values = np.zeros(len(t_values))

        I_bar = params['I_bar']
        S_ref = params['S_ref']
        b = params['b']

        S_values[0] = y0

        if compute_ET_internal is False:
            # No internal ET computation
            for i in range(t_end):
                S_values[i+1] = S_values[i] + dt * f(S_values[i], input[i], I_bar, S_ref, b)
                discharge_values[i] = I_bar * (S_values[i] / S_ref)**b

            return S_values, discharge_values[:-1], None

        else:
            # Internal ET computation
            ET_values = np.zeros(len(t_values))
            ETp = kwargs['ETp']
            sw = kwargs['sw']
            s_star = kwargs['s_star']

            for i in range(t_end):
                S_values[i+1] = (
                    S_values[i]
                    + dt * f(S=S_values[i], IN_=input[i],
                             ETp_=ETp[i], sw=sw, s_star=s_star,
                             I_bar_=I_bar, S_ref_=S_ref, b_=b)
                )

                discharge_values[i] = I_bar * (S_values[i] / S_ref)**b
                ET_values[i] = (
                    ETp[i] * np.minimum(1, np.maximum((S_values[i] - sw) / (s_star - sw), 0))
                )

            return S_values, discharge_values[:-1], ET_values[:-1]


class BackwardEulerSolver:
    def __init__(self):
        """
        Initialize the solver.
        """
        pass

    def solve(self, f, input, params, y0, dt, t0, t_end, compute_ET_internal, **kwargs):
        """
        Solve the ODE using the Backward Euler method.

        Parameters:
        f (function): The function defining the ODE (dy/dt = f(t, y))
        input (numpy.ndarray): The input timeseries to the storage.
        params (dict): The parameters of the discharge law (I_bar, S_ref, b)
        y0 (float): The initial condition at t0
        t0 (float): The initial time
        t_end (float): The end time
        dt (float): The time step size
        compute_ET_internal (bool) : Defines if ET should be computed
        kwargs : Additional arguments for ET computation
        """

        t_values = np.arange(t0, t_end + 1, dt)
        S_values = np.zeros(len(t_values))
        discharge_values = np.zeros(len(t_values))

        _I_bar = params['I_bar']
        _S_ref = params['S_ref']
        _b = params['b']

        S_values[0] = y0

        if compute_ET_internal is False:
            # No internal ET computation
            for i in range(t_end):
                def backward_euler_eq(S_next):
                    return S_next - S_values[i] - dt * f(S_next, input[i], _I_bar, _S_ref, _b)

                S_next = fsolve(backward_euler_eq, S_values[i])
                S_values[i + 1] = S_next
                discharge_values[i] = _I_bar * (S_next / _S_ref)**_b

            return S_values, discharge_values[:-1]

        else:
            # Internal ET computation
            ET_values = np.zeros(len(t_values))
            ETp = kwargs['ETp']
            sw = kwargs['sw']
            s_star = kwargs['s_star']

            for i in range(t_end):
                def backward_euler_eq(S_next):
                    return (S_next - S_values[i]
                            - dt * f(S=S_next, IN_=input[i],
                                     ETp_=ETp[i], sw=sw, s_star=s_star,
                                     I_bar_=_I_bar, S_ref_=_S_ref, b_=_b))

                S_next = fsolve(backward_euler_eq, S_values[i])
                S_values[i + 1] = S_next

                discharge_values[i] = _I_bar * (S_next / _S_ref)**_b
                ET_values[i] = (
                        ETp[i] * np.minimum(1, np.maximum((S_next - sw) / (s_star - sw), 0))
                    )

            return S_values, discharge_values[:-1], ET_values[:-1]


class RungeKutta4Solver:
    def __init__(self):
        """
        Initialize the solver.
        """
        pass

    def solve(self, f, input, params, y0, dt, t0, t_end, compute_ET_internal, **kwargs):
        """
        Solve the ODE using the Runge-Kutta 4 (RK4) method.

        Parameters:
        f (function): The function defining the ODE (dy/dt = f(t, y))
        input (numpy.ndarray): The input timeseries to the storage.
        params (dict): The parameters of the discharge law (I_bar, S_ref, b)
        y0 (float): The initial condition at t0
        t0 (float): The initial time
        t_end (float): The end time
        dt (float): The time step size
        compute_ET_internal (bool) : Defines if ET should be computed
        kwargs : Additional arguments for ET computation
        """

        t_values = np.arange(t0, t_end + 1, dt)
        S_values = np.zeros(len(t_values))
        discharge_values = np.zeros(len(t_values))

        _I_bar = params['I_bar']
        _S_ref = params['S_ref']
        _b = params['b']

        S_values[0] = y0

        if compute_ET_internal is False:
            # No internal ET computation
            for i in range(t_end):
                S = S_values[i]

                k1 = dt * f(S, input[i], _I_bar, _S_ref, _b)
                k2 = dt * f(S + 0.5 * k1, input[i], _I_bar, _S_ref, _b)
                k3 = dt * f(S + 0.5 * k2, input[i], _I_bar, _S_ref, _b)
                k4 = dt * f(S + k3, input[i], _I_bar, _S_ref, _b)

                S_values[i + 1] = S + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                discharge_values[i] = input[i] + (S_values[i] - S_values[i+1]) / dt

            return S_values, discharge_values[:-1]

        else:
            # Internal ET computation
            ET_values = np.zeros(len(t_values))
            ETp = kwargs['ETp']
            sw = kwargs['sw']
            s_star = kwargs['s_star']

            for i in range(t_end):
                S = S_values[i]

                k1 = dt * f(S, input[i], ETp[i], sw, s_star, _I_bar, _S_ref, _b)
                k2 = dt * f(S + 0.5 * k1, input[i], ETp[i], sw, s_star, _I_bar, _S_ref, _b)
                k3 = dt * f(S + 0.5 * k2, input[i], ETp[i], sw, s_star, _I_bar, _S_ref, _b)
                k4 = dt * f(S + k3, input[i], ETp[i], sw, s_star, _I_bar, _S_ref, _b)

                S_values[i + 1] = S + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

                # Get ET from averaging intermediate S(t) values
                S_avg = (S + 2 * (S + k1/2) + 2 * (S + k2/2) + S + k3) / 6
                ET_values[i] = (
                    ETp[i] * np.minimum(1, np.maximum((S_avg - sw) / (s_star - sw), 0))
                )

                # Get discharge from mass balance
                discharge_values[i] = input[i] - ET_values[i] + (S_values[i] - S_values[i+1]) / dt

            return S_values, discharge_values[:-1], ET_values[:-1]


class ScipySolver:
    def __init__(self):
        """
        Initialize the solver.
        """
        pass

    def solve(self, f, input, params, y0, dt, t0, t_end, compute_ET_internal, **kwargs):
        """
        Solve the ODE using SciPy's ODE solver (RK45).

        Parameters:
        f (function): The function defining the ODE (dy/dt = f(t, y))
        (NOT USED BUT HARDCODED IN THIS CLASS iN COMPATIBLE FORMAT FO SCIPY)
        input (numpy.ndarray): The input timeseries to the storage.
        params (dict): The parameters of the discharge law (I_bar, S_ref, b)
        y0 (float): The initial condition at t0
        t0 (float): The initial time
        t_end (float): The end time
        dt (float): The time step size
        compute_ET_internal (bool) : Defines if ET should be computed
        kwargs : Additional arguments for ET computation
        """

        t_values = np.arange(t0, t_end + 1, dt)
        S_values = np.zeros(len(t_values))
        discharge_values = np.zeros(len(t_values))

        _I_bar = params['I_bar']
        _S_ref = params['S_ref']
        _b = params['b']

        if compute_ET_internal is False:  # No internal ET computation
            # Define the ODE function
            def differential_equation(t, S, input_, I_bar_, S_ref_, b_):
                dS_dt = input_[t] - I_bar_ * (S / S_ref_)**b_
                return dS_dt

            t_span = np.arange(t0, t_end, dtype=int)

            sol = solve_ivp(differential_equation, t_span, [y0], args=(input, _I_bar, _S_ref, _b), t_eval=t_span)

            S_values = sol.y[0]
            discharge_values = input + (S_values[:-1] - S_values[1:]) / dt

            return S_values, discharge_values

        elif compute_ET_internal is True:  # Internal ET computation
            raise ValueError('This numerical solver has not been implemented with internal ET computation.')
            # ET_values = np.zeros(len(t_values))
            # ETp = kwargs['ETp']
            # sw = kwargs['sw']
            # s_star = kwargs['s_star']

            # # Define the ODE function
            # def differential_equation(t, S, input_, I_bar_, S_ref_, b_, ETp_, sw_, s_star_):
            #     dS_dt = input_[t] - ETp_[t] * np.minimum(1, np.maximum((S - sw_) / (s_star_ - sw_), 0)) - I_bar_ * (S / S_ref_)**b_
            #     return dS_dt

            # t_span = np.arange(t0, t_end, dtype=int)

            # sol = solve_ivp(differential_equation, t_span, [y0], args=(input, _I_bar, _S_ref, _b, ETp, sw, s_star), t_eval=t_span)

            # S_values = sol.y[0]
            # discharge_values =  # need to analyse solving scheme
            # ET_values =  # need to analyse solving scheme

            # return S_values, discharge_values, ET_values


class TrapezoidalBackwardEulerSolver:
    def __init__(self):
        """
        Initialize the solver.
        """
        pass

    def solve(self, f, input, params, y0, dt, t0, t_end, compute_ET_internal, **kwargs):
        """
        Solve the ODE using the Trapezoidal-Backward Euler method from Kirchner (2016).

        Parameters:
        f (function): The function defining the ODE (dy/dt = f(t, y))
                      (NOT USED BUT HARDCODED IN THIS CLASS)
        input (numpy.ndarray): The input timeseries to the storage.
        params (dict): The parameters of the discharge law (I_bar, S_ref, b)
        y0 (float): The initial condition at t0
        t0 (float): The initial time
        t_end (float): The end time
        dt (float): The time step size
        compute_ET_internal (bool) : Defines if ET should be computed
        kwargs : Additional arguments for ET computation
        """

        t_values = np.arange(t0, t_end + 1, dt)
        S_values = np.zeros(len(t_values))
        discharge_values = np.zeros(len(t_values))

        _I_bar = params['I_bar']
        _S_ref = params['S_ref']
        _b = params['b']

        S_values[0] = y0

        if compute_ET_internal is False:  # No internal ET computation

            for i in range(t_end):

                rho = np.min(0.5 + 0.5 * (input[i] - _I_bar * (S_values[i] / _S_ref)**_b) / ((input[i] / (_I_bar * _S_ref**_b)) - S_values[i]), 1)

                def trapezoidal_backward_euler_eq(S_next):
                    zero_objective = (
                        S_next - S_values[i]
                        - dt * (input[i] - rho * _I_bar * (S_next / _S_ref)**_b - (1 - rho) * _I_bar * (S_values[i] / _S_ref)**_b)
                    )
                    return zero_objective

                S_next = fsolve(trapezoidal_backward_euler_eq, S_values[i])
                S_values[i + 1] = S_next
                discharge_values[i] = input[i] + (S_values[i] - S_values[i+1]) / dt

            return S_values, discharge_values[:-1]

        elif compute_ET_internal is True:  # Internal ET computation
            raise ValueError('This numerical solver has not been implemented with internal ET computation.')
