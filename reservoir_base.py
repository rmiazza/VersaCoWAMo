import numpy as np
from base_elements import BaseElement
from abc import ABC, abstractmethod


class BaseReservoir(BaseElement, ABC):
    num_upstream = 1

    def __init__(self, id, initial_storage, numerical_solver, dt=1,
                 concentration_computation=False, reaction_type='Non-reactive',
                 initial_concentration=None, reaction_rate=None,
                 equilibrium_concentration=None,
                 age_computation=False, num_ages_tracked=None, initial_TTD=None
                 ):
        """
        This is the initializer of the class BaseReservoir.

        Parameters
        ----------
        - Flow :
        id : str
            Identifier of the element.
        initial_storage : float
            Initial storage in the reservoir.
        numerical_solver : object
            Instance of chosen numerical solver class.
        dt : int
            Size of the timestep in comparison to the input flux units.

        - Concentration :
        concentration_computation : Bool
            Defines if concentration should be computed.
        reaction_type : str
            Defines which type of reactive processes the tracer undergoes.
            Should be either 'Non-reactive', 'FOK' or 'ED'.
        initial_concentration : float
            Initial concentration in the reservoir.
        reaction_rate : float
            Reaction rate for first order kinetics or exponential decay.
        equilibrium_concentration : float
            Equilibrium concentration for first order kinetics of
            geogenic solutes.

        - Water age tracking :
        age_computation : Bool
            Defines if ages should be tracked (as ranked storage).
        num_ages_tracked : int
            Maximal age tracked in the age distribution (i.e. length of the
            TTD), the rest is stored as "old" water.
        initial_TTD : numpy.array
            Initial residence time distribution in storage if a specific TTD
            should be defined (no-steady-state spinup; for example for
            setting the TTD after a manual spinup period). It must be the size
            of num_age_tracked.
        """

        BaseElement.__init__(self, id)

        # Flow initialization
        self.initial_storage = initial_storage  # storage relevant for water age

        # Numerical solver initialization
        self.numerical_solver = numerical_solver
        self.dt = dt

        # Concentration initialization
        self.concentration_computation = concentration_computation

        if concentration_computation:
            self.reaction_type = reaction_type
            self.initial_concentration = initial_concentration

            if reaction_type == 'Non-reactive':
                pass

            elif reaction_type == 'FOK':
                if reaction_rate is None or equilibrium_concentration is None:
                    raise ValueError('Missing first order kinetics parameter.')
                else:
                    self.reaction_rate = reaction_rate
                    self.C_eq = equilibrium_concentration

            elif reaction_type == 'ED':
                if reaction_rate is None:
                    raise ValueError('Missing exponential decay parameter.')
                else:
                    self.reaction_rate = reaction_rate

            else:
                raise ValueError(f'Unknown reaction type: {reaction_type}.')

        # Ranked storage initialization
        self.age_computation = age_computation

        if age_computation:
            self.initial_TTD = initial_TTD
            if num_ages_tracked is None:
                raise ValueError('Missing maximal age to be tracked.')
            else:
                self.num_ages_tracked = num_ages_tracked
        
        # Evapotranspiration initialization (to be overridden by subclasses)
        self.compute_ET = False
        self.alpha_ET = 1

    def set_input(self, input_tuple):
        """
        The method sets the inputs of the element.

        Parameters
        ----------
        input_tuple : list(Tuple)
            List containing a single Tuple with:
            - the input flux timeseries (index 0) as numpy.array;
            - the input concentration timeseries (index 1) as numpy.array;
            - the input water TTDs (index 2, timeseries of input TTD) as numpy.ndarray;
        """
        self.input_flux = input_tuple[0][0]
        self.input_concentration = input_tuple[0][1]
        self.input_TTD = input_tuple[0][2]

    def delete_inputs(self):
        """
        The method sets all inputs to 'None' to free some memory.
        It is called after the outputs have been computed.
        """
        self.input_flux = None
        self.input_concentration = None
        self.input_TTD = None
    
    def solve_ranked_storage(self):
        """
        Solves the water age balance only.
        """
        # Initialize ranked storage matrix
        self.ranked_storage = np.zeros([len(self.storage), self.num_ages_tracked], dtype=float)

        # Initialize old storage
        self.S_old = np.zeros(len(self.storage))

        # Set initial values for t=0
        self.ranked_storage[0, :], self.S_old[0] = self.initialize_ranked_storage()  # reservoir-specific

        # Define local ET in case no evapotranspiration is taken into account
        if self.compute_ET:
            ET_flux = self.output_ET
        else:
            ET_flux = np.zeros_like(self.input_flux)

        # Update ranked and old storage for each timestep
        for i, (IN, ET, OUT) in enumerate(zip(self.input_flux, ET_flux, self.output_flux)):
            # Aging
            ranked_storage_next = np.roll(self.ranked_storage[i, :], shift=1)
            ranked_storage_next[0] = 0
            self.S_old[i+1] = self.S_old[i] + self.ranked_storage[i, -1]

            # Input water
            ranked_storage_next = ranked_storage_next + IN * self.dt * self.input_TTD[i, :]
            self.S_old[i+1] = self.S_old[i+1] + IN * self.dt * (1 - np.sum(self.input_TTD[i, :]))

            # Output water
            ranked_storage_next = ranked_storage_next / (1 + (OUT + ET) * self.dt / self.storage[i+1])
            self.S_old[i+1] = self.S_old[i+1] / (1 + (OUT + ET) * self.dt / self.storage[i+1])

            self.ranked_storage[i+1, :] = ranked_storage_next

    def solve_storage_and_output(self):
        """
        Solves the storage ODE and computes the output flux timeseries.
        Delegates the numerical integration to the external solver, passing
        the ODE and parameters defined by the specific reservoir subclass.
        """
        # Retrieve the ODE and parameters from the subclass implementation
        ode = self._define_storage_ode()
        params = self._get_parameters()

        # Solve the ODE and unpack storage, output flux and ET timeseries
        self.storage, self.output_flux, self.output_ET = (
            self.numerical_solver.solve(
                f=ode, input=self.input_flux,
                params=params, y0=self.initial_storage,
                dt=self.dt
            )
        )
    
    def initialize_concentration(self):
        """
        Initializes the storage concentration. Based on the analytical solution
        for storage concentration at steady-state.
        """
        if self.initial_concentration is not None:
            # Implies initial concentration was manually set -> do not change it
            return

        # Define constants
        if self.compute_ET:
            ET_bar = np.mean(self.output_ET)
            alpha_ET_ = self.alpha_ET
        else:
            ET_bar = 0
            alpha_ET_ = 1

        IN_bar = np.mean(self.input_flux)
        OUT_bar = IN_bar - ET_bar  # or np.mean(self.output_flux)
        Cin_bar = np.mean(self.input_concentration)

        # Compute initial storage concentration
        if self.reaction_type == 'Non-reactive':
            self.initial_concentration = (
                (IN_bar * Cin_bar) / (ET_bar * alpha_ET_ + OUT_bar)
            )

        elif self.reaction_type == 'ED':
            self.initial_concentration = (
                (IN_bar * Cin_bar)
                / (ET_bar * alpha_ET_ + OUT_bar + self.reaction_rate * self.initial_storage)
            )

        elif self.reaction_type == 'FOK':
            self.initial_concentration = (
                (IN_bar * Cin_bar + self.reaction_rate * self.C_eq * self.initial_storage)
                / (ET_bar * alpha_ET_ + OUT_bar + self.reaction_rate * self.initial_storage)
            )

    def solve_concentration(self):
        """
        Solves the concentration in storage with a hardcoded Runge-Kutta 4 approach.

        ODE to be solved:
        dC/dt = (
            (IN_flux * (C_in - C(t)) + ET_flux * (1 - a) * C(t))
            / (S(t) + (IN_flux - OUT_flux) * (t - t0)) + f(C(t))
            )

        where f is the reaction law for the tracers.
        """
        # Define reaction law
        if self.reaction_type == 'Non-reactive':
            def reaction_law(C_s):
                return 0

        elif self.reaction_type == 'FOK':
            def reaction_law(C_s):
                return self.reaction_rate * (self.C_eq - C_s)

        elif self.reaction_type == 'ED':
            def reaction_law(C_s):
                return -self.reaction_rate * C_s

        # Define differential equation for concentration
        if self.compute_ET:
            alpha_ET_ = self.alpha_ET
        else:
            alpha_ET_ = 1

        def differential_equation(C_s, IN_, ET_, OUT_, Cin_, S_i_, dt_):
            result = (
                (IN_ * (Cin_ - C_s) + ET_ * (1 - alpha_ET_) * C_s)
                / (S_i_ + (IN_ - ET_ - OUT_) * dt_)
                + reaction_law(C_s)
            )
            return result

        # Initialize delta mass through reaction component
        delta_m_reaction = np.zeros_like(self.input_flux, dtype=float)

        # Initialize evapotranspiration concentration over each timestep
        C_ET = np.zeros_like(self.input_flux, dtype=float)

        # Initialize the storage concentration
        self.storage_concentration = np.zeros_like(self.storage, dtype=float)
        self.initialize_concentration()
        self.storage_concentration[0] = self.initial_concentration

        # Define ET in case evapotranspiration is or isn't taken into account
        if self.compute_ET:
            ET_flux = self.output_ET
        else:
            ET_flux = np.zeros_like(self.input_flux)

        # Solve differential equation
        dt = self.dt
        for i, (IN, ET, OUT, Cin) in enumerate(zip(self.input_flux, ET_flux, self.output_flux, self.input_concentration)):
            C_s_i = self.storage_concentration[i]
            S_i = self.storage[i]

            k1 = differential_equation(
                C_s=C_s_i, IN_=IN, ET_=ET, OUT_=OUT, Cin_=Cin, S_i_=S_i, dt_=0
            )
            k2 = differential_equation(
                C_s=C_s_i+dt/2*k1, IN_=IN, ET_=ET, OUT_=OUT, Cin_=Cin, S_i_=S_i, dt_=dt/2
            )
            k3 = differential_equation(
                C_s=C_s_i+dt/2*k2, IN_=IN, ET_=ET, OUT_=OUT, Cin_=Cin, S_i_=S_i, dt_=dt/2
            )
            k4 = differential_equation(
                C_s=C_s_i+dt*k3, IN_=IN, ET_=ET, OUT_=OUT, Cin_=Cin, S_i_=S_i, dt_=dt
            )

            self.storage_concentration[i + 1] = (
                C_s_i + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            )

            delta_m_reaction[i] = (
                (reaction_law(C_s_i) * S_i
                 + 2 * reaction_law(C_s_i + dt/2*k1) * (S_i + dt/2 * (IN - ET - OUT))
                 + 2 * reaction_law(C_s_i + dt/2*k2) * (S_i + dt/2 * (IN - ET - OUT))
                 + reaction_law(C_s_i + dt*k3) * (S_i + dt * (IN - ET - OUT)))
                / 6
            )

            C_ET[i] = (
                alpha_ET_ * (C_s_i
                            + 2*(C_s_i + dt/2*k1)
                            + 2*(C_s_i + dt/2*k2)
                            +   (C_s_i + dt  *k3))
                / 6
            )

        # Compute output concentration by mass balance
        self.output_concentration = (
            (self.input_flux * self.input_concentration * dt
             - ET_flux * C_ET * dt
             + self.storage_concentration[:-1] * self.storage[:-1]
             - self.storage_concentration[1:] * self.storage[1:]
             + delta_m_reaction * dt)
            / (self.output_flux * dt)
        )

    def get_output(self):
        """
        General method calling all methods to solve the states of the reservoir
        and returns the output flux, concentration and the flux TTD as a Tuple.
        """
        # Flow is computed by default
        self.solve_storage_and_output()

        if self.concentration_computation:
            self.solve_concentration()

        if self.age_computation:
            self.solve_ranked_storage()

        self.delete_inputs()

        if self.concentration_computation and self.age_computation:
            # Case concentration and ranked storage
            return [tuple((self.output_flux, self.output_concentration,
                        self.ranked_storage[1:, :] / self.storage[1:, np.newaxis]))]
        
        if self.concentration_computation:
            # Case concentration only
            return [tuple((self.output_flux, self.output_concentration, None))]
        
        return [tuple((self.output_flux, None, None))]

    @abstractmethod
    def _define_storage_ode(self):
        """Method that reservoir subclasses need to define"""
        pass
