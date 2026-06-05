import numpy as np
from scipy.signal import fftconvolve
from base_elements import BaseElement


class Reservoir(BaseElement):

    num_upstream = 1

    def __init__(self, parameters, id, initial_storage, numerical_solver, dt=1,
                 concentration_computation=False, reaction_type='Non-reactive',
                 initial_concentration=None, reaction_rate=None,
                 equilibrium_concentration=None,
                 compute_ET=False, alpha_ET=1,
                 ET_timeseries=None, ETp_timeseries=None,
                 wilting_point=None, s_star=None,
                 age_computation=False, initial_TTD=None,
                 num_ages_tracked=None, mass_tracking_computation=False,
                 initial_ranked_mass=None
                 ):
        """
        This is the initializer of the class Reservoir.

        Parameters
        ----------
        - Flow :
        parameters : dict
            Flow parameters controlling the storage-discharge relationship.
            Each parameters is a float (constant in time).
            -> ('IN_bar', 'S_ref', 'b')
        id : str
            Identifier of the element.
        initial_storage : float
            Initial storage in the reservoir.
        numerical_solver : object
            Instance of chosen numerical solver class.
        dt : int
            Size of the timestep (not rigorously implemented).

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

        - Evapotranspiration :
        compute_ET : Bool
            Defines if evapotranspiration should be taken into account.
            For the moment needs to provide ET timeseries, later will be
            implemented in a method.
        alpha_ET : float
            Evapoconcentration factor. Should be between 0 (maximal
            evapoconcentration) and 1 (no evapoconcentration, default).
        ET_timeseries : numpy.array
            Timeseries of evapotranspiration from the reservoir.
        ETp_timeseries : numpy.array
            Timeseries of potential evapotranspiration for internal ET
            computation.
        wilting_point: float
            Wilting point storage for internal ET compuation.
        s_star: float
            Storage at which ET is independent of S for internal ET
            compuattion.

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

        - Mass age tracking :
        mass_tracking_computation : Bool
            Defines if mass should be tracked (as ranked mass).
        initial_ranked_mass : numpy.array
            Initial ranked mass in storage if a specific ranked mass
            should be defined (no-steady-state spinup; for example for
            setting the ranked mass after a manual spinup period). It must be
            the size of num_age_tracked.
        """

        BaseElement.__init__(self, id)

        # Flow initialization
        self._parameters = parameters
        self.initial_storage = initial_storage

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

        # Ranked mass initialization
        self.mass_tracking_computation = mass_tracking_computation

        if mass_tracking_computation is True and age_computation is False:
            raise ValueError('Mass cannot be tracked without tracking water age.')

        if mass_tracking_computation:
            if initial_ranked_mass is not None and initial_storage is None:
                raise ValueError('If an initial ranked mass is given, the initial storage concentration must also be given.')
            if initial_ranked_mass is not None and initial_TTD is None:
                raise ValueError('If an initial ranked mass is given, the initial storage TTD should also be given for consistency.')

            self.initial_ranked_mass = initial_ranked_mass

        # Evapotranspiration initialization
        self.compute_ET = compute_ET
        self.compute_ET_internal = False

        if compute_ET:
            if ET_timeseries is None and ETp_timeseries is None:
                raise ValueError('Missing either ETp or ET timeseries input.')
            elif ETp_timeseries is not None:
                self.compute_ET_internal = True
                self.output_ETp = ETp_timeseries
                self.sw = wilting_point
                self.s_star = s_star
            else:  # to be erased if only computed internally
                self.output_ET = ET_timeseries

            self.alpha_ET = alpha_ET

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
            - the input mass TTDs (index 3, timeseries of input TTD) as numpy.ndarray.
        """

        self.input_flux = input_tuple[0][0]
        self.input_concentration = input_tuple[0][1]
        self.input_TTD = input_tuple[0][2]
        self.input_mass_TTD = input_tuple[0][3]

    def delete_inputs(self):
        """
        The method sets all inputs to 'None' to free some memory.
        It is called after the outputs have been computed.
        """
        self.input_flux = None
        self.input_concentration = None
        self.input_TTD = None
        self.input_mass_TTD = None

    def solve_storage_and_output(self):
        """
        Solves the differential equation for storage, computes the output flux and
        updates the states.

        ODE to be solved:
        dS(t)/dt = input - output = P - ET - I_bar * (S(t)/S_ref)^b

        ODE if ET is computed internally:
        dS(t)/dt = input - output = P - ET(S(t)) - I_bar * (S(t)/S_ref)^b
        """
        # Define the I_bar parameter based on input flux if None was given
        if self._parameters['I_bar'] is None:
            self._parameters['I_bar'] = np.mean(self.input_flux)

        # Prepare the net (or "effective") input flux
        if self.compute_ET is False or self.compute_ET_internal is True:
            net_input_flux = self.input_flux
        else:
            # Subtract ET from input flux
            net_input_flux = self.input_flux - self.output_ET

        # Solve the ODE
        if self.compute_ET is False or self.compute_ET_internal is False:
            # Case ET is not considered or not computed internally
            def differential_equation(S, IN_, I_bar_, S_ref_, b_):
                delta_S = IN_ - I_bar_ * (S / S_ref_)**b_
                return delta_S

            self.storage, self.output_flux, _ = (
                self.numerical_solver.solve(
                    f=differential_equation, input=net_input_flux,
                    params=self._parameters, y0=self.initial_storage,
                    dt=self.dt, t0=0, t_end=len(self.input_flux),
                    compute_ET_internal=self.compute_ET_internal
                    )
            )

        else:
            # Case ET is computed internally
            def differential_equation(S, IN_, ETp_, sw, s_star, I_bar_, S_ref_, b_):
                delta_S = (
                    IN_
                    - ETp_ * np.minimum(1, np.maximum((S - sw) / (s_star - sw), 0))
                    - I_bar_ * (S / S_ref_)**b_
                    )
                return delta_S

            self.storage, self.output_flux, self.output_ET = (
                self.numerical_solver.solve(
                    f=differential_equation, input=net_input_flux,
                    params=self._parameters, y0=self.initial_storage,
                    dt=self.dt, t0=0, t_end=len(self.input_flux),
                    compute_ET_internal=self.compute_ET_internal,
                    ETp=self.output_ETp, sw=self.sw, s_star=self.s_star
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
            alpha_ET_ = 0

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
                / (ET_bar * alpha_ET_ + OUT_bar + self.reaction_rate * self._parameters['S_ref'])
            )

        elif self.reaction_type == 'FOK':
            self.initial_concentration = (
                (IN_bar * Cin_bar + self.reaction_rate * self.C_eq * self._parameters['S_ref'])
                / (ET_bar * alpha_ET_ + OUT_bar + self.reaction_rate * self._parameters['S_ref'])
            )

    def solve_concentration(self):
        """
        Solves the differential equation for storage concentration of
        non-reactive tracers in the absence of evapotranspiration, computes
        the output concentration and updates the states.

        Based on the semi-analytical solution for non-reactive transport presented in
        Kirchner (2016) - Aggregation in environmental systems – Part 2 (Appendix A)

        ODE to be solved:
        dC/dt = IN_flux * (C_in - C(t)) / (S(t0) + (IN_flux - OUT_flux) * (t - t0))
        """
        # Stop if the tracer is reactive or if ET is considered
        if self.reaction_type != 'Non-reactive' or self.compute_ET is True:
            return

        # Initialize the storage concentration
        self.storage_concentration = np.zeros(len(self.storage))
        self.initialize_concentration()
        self.storage_concentration[0] = self.initial_concentration

        # Compute storage concentration
        for i, (IN_flux, OUT_flux, C_IN) in enumerate(zip(self.input_flux, self.output_flux, self.input_concentration)):
            if np.abs(1 - IN_flux / OUT_flux) > 1/1000:
                self.storage_concentration[i+1] = (
                    C_IN + (self.storage_concentration[i] - C_IN)
                    * (self.storage[i] / self.storage[i+1])**(IN_flux / (IN_flux - OUT_flux))
                    )
            else:
                self.storage_concentration[i+1] = (
                    C_IN + (self.storage_concentration[i] - C_IN) * np.exp(-IN_flux * self.dt / self.storage[i])
                    )

        # Compute output concentration by mass balance
        self.output_concentration = (
            (self.input_concentration * self.input_flux
             + self.storage_concentration[:-1] * self.storage[:-1]
             - self.storage_concentration[1:] * self.storage[1:])
            / self.output_flux
            )

    def solve_reactive_or_with_ET_concentration(self):
        """
        Solves the concentration in storage in case evapotranspiration is
        taken into account and/or for reactive tracers with a hardcoded
        Runge-Kutta 4 approach.

        ODE to be solved:
        dC/dt = (IN_flux * (C_in - C(t)) + ET_flux * (1 - a) * C(t))
                / (S(t) + (IN_flux - OUT_flux) * (t - t0)) + f(C(t))

        where f is the reaction law for the tracers.
        """
        # Stop if the tracer is non-reactive and evapotranspiration is not present
        if self.reaction_type == 'Non-reactive' and self.compute_ET is False:
            return

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
            alpha_ET_ = 0

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
        for i, (IN, ET, OUT, Cin) in enumerate(zip(self.input_flux, ET_flux, self.output_flux, self.input_concentration)):
            C_s_i = self.storage_concentration[i]
            S_i = self.storage[i]

            k1 = differential_equation(
                C_s=C_s_i, IN_=IN, ET_=ET, OUT_=OUT, Cin_=Cin, S_i_=S_i, dt_=0
            )
            k2 = differential_equation(
                C_s=C_s_i+.5*k1, IN_=IN, ET_=ET, OUT_=OUT, Cin_=Cin, S_i_=S_i, dt_=.5
            )
            k3 = differential_equation(
                C_s=C_s_i+.5*k2, IN_=IN, ET_=ET, OUT_=OUT, Cin_=Cin, S_i_=S_i, dt_=.5
            )
            k4 = differential_equation(
                C_s=C_s_i+k3, IN_=IN, ET_=ET, OUT_=OUT, Cin_=Cin, S_i_=S_i, dt_=1
            )

            self.storage_concentration[i + 1] = (
                C_s_i + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            )

            delta_m_reaction[i] = (
                (reaction_law(C_s_i) * S_i
                 + 2 * reaction_law(C_s_i + .5*k1) * (S_i + .5 * (IN - ET - OUT))
                 + 2 * reaction_law(C_s_i + .5*k2) * (S_i + .5 * (IN - ET - OUT))
                 + reaction_law(C_s_i + k3) * (S_i + 1 * (IN - ET - OUT)))
                / 6
            )

            C_ET[i] = (
                alpha_ET_ * (C_s_i + 2*(C_s_i + .5*k1) + 2*(C_s_i + .5*k2) + (C_s_i + k3))
                / 6
            )

        # Store ET concentration if computed (needed for mass tracking)
        if self.compute_ET and self.mass_tracking_computation:
            self.ET_concentration = C_ET

        # Compute output concentration by mass balance
        self.output_concentration = (
            (self.input_flux * self.input_concentration
             - ET_flux * C_ET
             + self.storage_concentration[:-1] * self.storage[:-1]
             - self.storage_concentration[1:] * self.storage[1:]
             + delta_m_reaction)
            / self.output_flux
            )

    def initialize_ranked_storage(self):
        """
        Computes the initial ranked storage and "old" storage based on the
        solution of the steady-state eqation for randomly sampled reservoirs.
        The transit time distribution (TTD) is obtained by convolving the
        local TTD since entry in the reservoir with the TTD of the input flux.

        If an initial TTD was given via the parameter initial_TTD, it will be
        assigned here.
        """
        if self.initial_TTD is not None:
            initial_S_old = self.initial_storage * (1 - np.sum(self.initial_TTD))
            return self.initial_storage * self.initial_TTD, initial_S_old

        else:
            local_steady_state_TTD = (
                self._parameters['I_bar'] / self.initial_storage
                * np.exp(-self._parameters['I_bar'] / self.initial_storage * np.arange(self.num_ages_tracked))
            )

            initial_TTD = fftconvolve(self.input_TTD[0, :], local_steady_state_TTD, mode='full')[:self.num_ages_tracked]

            # Make sure the TTD is a pdf (avoid potential numerical errors)
            if np.sum(initial_TTD) > 1:
                initial_TTD = initial_TTD / np.sum(initial_TTD)

            initial_S_old = self.initial_storage * (1 - np.sum(initial_TTD))

            return self.initial_storage * initial_TTD, initial_S_old

    def solve_ranked_storage(self):
        """
        Solves the water age balance only.
        """
        # Initialize ranked storage matrix
        self.ranked_storage = np.zeros([len(self.storage), self.num_ages_tracked], dtype=float)

        # Initialize old storage
        self.S_old = np.zeros(len(self.storage))

        # Set initial values for t=0
        self.ranked_storage[0, :], self.S_old[0] = self.initialize_ranked_storage()

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

    def initialize_ranked_mass(self):
        # TO DO : correction/check of method
        """
        Computes the initial ranked mass and "old" mass in storage based on the
        solution of the steady-state equation for randomly sampled reservoirs.

        If an initial ranked mass was given via the parameter
        initial_ranked_mass, it will be assigned here.
        """
        # Case where an initial ranked mass was given as input
        if self.initial_ranked_mass is not None:
            initial_old_mass = (
                self.initial_storage * self.initial_concentration
                - np.sum(self.initial_ranked_mass)
                )
            return self.initial_ranked_mass, initial_old_mass

        # Define evapotranspiration rates locally in case there is none
        if self.compute_ET:
            ET_bar = np.mean(self.output_ET)
            alpha_ET_ = self.alpha_ET
        else:
            ET_bar = 0
            alpha_ET_ = 0

        # Define constants
        Cin_bar = np.mean(self.input_concentration)
        IN_bar = self._parameters['I_bar']  # could also be np.mean(self.input_flux)
        OUT_bar = IN_bar - ET_bar  # could also be np.mean(self.output_flux)
        S_bar = self._parameters['S_ref']
        age_steps = np.arange(self.num_ages_tracked)
        initial_storage_mass = self.initial_concentration * self.initial_storage

        # Compute initial ranked mass and old mass
        if self.reaction_type == 'Non-reactive':
            initial_ranked_mass = (
                np.exp(-(ET_bar * alpha_ET_ + OUT_bar) / S_bar * age_steps)
                * np.cumsum(np.exp((ET_bar * alpha_ET_ + OUT_bar) / S_bar * age_steps)
                            * Cin_bar * IN_bar * self.input_mass_TTD[0])
            )
            print(initial_ranked_mass)

        elif self.reaction_type == 'ED':
            initial_ranked_mass = (
                np.exp(-((ET_bar * alpha_ET_ + OUT_bar) / S_bar + self.reaction_rate) * age_steps)
                * np.cumsum(np.exp(((ET_bar * alpha_ET_ + OUT_bar) / S_bar + self.reaction_rate) * age_steps)
                            * Cin_bar * IN_bar * self.input_mass_TTD[0])
            )

        elif self.reaction_type == 'FOK':
            initial_ranked_mass = (
                np.exp(-((ET_bar * alpha_ET_ + OUT_bar) / S_bar + self.reaction_rate) * age_steps)
                * np.cumsum(np.exp(((ET_bar * alpha_ET_ + OUT_bar) / S_bar + self.reaction_rate) * age_steps)
                            * (Cin_bar * IN_bar * self.input_mass_TTD[0] + self.reaction_rate * self.C_eq * self.ranked_storage[0]))
            )

        # Compute initial old storage
        if np.sum(initial_ranked_mass) < 0:
            # To delete when sure about anlytical solution
            raise ValueError('Summed initial ranked mass is negative.')

        elif np.sum(initial_ranked_mass) <= initial_storage_mass:
            initial_old_mass = (
                initial_storage_mass - np.sum(initial_ranked_mass)
            )

        elif np.sum(initial_ranked_mass) > initial_storage_mass:
            # Normalize to ensure mass conservation
            initial_ranked_mass = (
                initial_ranked_mass * initial_storage_mass / np.sum(initial_ranked_mass)
            )
            initial_old_mass = 0

        return initial_ranked_mass, initial_old_mass

    def solve_ranked_storage_and_mass(self):
        """
        Computes the ranked storage and ranked mass together.
        Implements first order reactions.
        """
        # Initialize ranked storage matrix and old storage
        self.ranked_storage = np.zeros([len(self.storage), self.num_ages_tracked], dtype=float)
        self.S_old = np.zeros(len(self.storage))
        self.ranked_storage[0, :], self.S_old[0] = self.initialize_ranked_storage()

        # Define reaction rates
        if self.reaction_type == 'FOK':
            def reaction_rate_FOK(S_r, m_r):
                # Computes the reaction rate m_dot for first order kinetics
                return S_r * self.reaction_rate * self.C_eq - self.reaction_rate * m_r

        elif self.reaction_type == 'ED':
            def reaction_rate_ED(m_r):
                # Computes the reaction rate m_dot for exponential decay
                return -self.reaction_rate * m_r

        # Initialize ranked mass matrix and old mass
        self.ranked_mass = np.zeros([len(self.storage), self.num_ages_tracked], dtype=float)
        self.m_old = np.zeros(len(self.storage))
        self.ranked_mass[0, :], self.m_old[0] = self.initialize_ranked_mass()

        # Define local ET in case no evapotranspiration is taken into account
        if self.compute_ET:
            ET_flux = self.output_ET
            ET_concentration = self.ET_concentration
        else:
            ET_flux = np.zeros_like(self.input_flux)
            ET_concentration = ET_flux  # can be any value anyways

        # Update ranked and old storage and mass for each timestep
        for i, (IN, ET, OUT) in enumerate(zip(self.input_flux, ET_flux, self.output_flux)):
            # 1) Ranked Storage
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

            # 2) Ranked Mass
            # Aging
            ranked_mass_next = np.roll(self.ranked_mass[i, :], shift=1)
            ranked_mass_next[0] = 0
            self.m_old[i+1] = self.m_old[i] + self.ranked_mass[i, -1]

            # Reaction (forward Euler approach)
            if self.reaction_type == 'Non-reactive':
                m_dot = 0
                m_dot_old = 0

            elif self.reaction_type == 'FOK':
                m_dot = reaction_rate_FOK(S_r=self.ranked_storage[i, :-1], m_r=ranked_mass_next[1:])
                m_dot_old = reaction_rate_FOK(S_r=self.S_old[i], m_r=self.m_old[i+1])

            elif self.reaction_type == 'ED':
                m_dot = reaction_rate_ED(m_r=ranked_mass_next[1:])
                m_dot_old = reaction_rate_ED(m_r=self.m_old[i+1])

            ranked_mass_next[1:] = ranked_mass_next[1:] + m_dot
            self.m_old[i+1] = self.m_old[i+1] + m_dot_old

            # Input mass
            ranked_mass_next = (
                ranked_mass_next
                + IN * self.dt * self.input_concentration[i] * self.input_mass_TTD[i, :]
            )
            self.m_old[i+1] = (
                self.m_old[i+1]
                + IN * self.dt * self.input_concentration[i] * (1 - np.sum(self.input_mass_TTD[i, :]))
            )

            # Output mass (TO DO : add ET)
            ranked_mass_next = (
                ranked_mass_next
                / (1 + self.dt * (ET * ET_concentration[i] + OUT * self.output_concentration[i]) / (self.storage[i+1] * self.storage_concentration[i+1]))
            )
            self.m_old[i+1] = (
                self.m_old[i+1]
                / (1 + self.dt * (ET * ET_concentration[i] + OUT * self.output_concentration[i]) / (self.storage[i+1] * self.storage_concentration[i+1]))
            )

            # Assign ranked mass
            self.ranked_mass[i+1, :] = ranked_mass_next

    def get_output(self):
        """
        General method calling all methods to solve the states of the reservoir
        and returns the output flux, concentration, the TTD of fluxes and the
        TTD of mass as a Tuple.
        """
        # Flow is computed by default
        self.solve_storage_and_output()

        if self.concentration_computation is True and self.age_computation is False:
            # Case concentration only
            self.solve_concentration()
            self.solve_reactive_or_with_ET_concentration()
            self.delete_inputs()
            return [tuple((self.output_flux, self.output_concentration, None, None))]

        if self.concentration_computation is True and self.age_computation is True and self.mass_tracking_computation is False:
            # Case concentration and ranked storage only
            self.solve_concentration()
            self.solve_reactive_or_with_ET_concentration()
            self.solve_ranked_storage()
            self.delete_inputs()
            # Very slow to add new axis...
            return [tuple((self.output_flux, self.output_concentration, self.ranked_storage[1:, :] / self.storage[1:, np.newaxis], None))]

        if self.concentration_computation is True and self.mass_tracking_computation is True:
            # Case concentration, ranked storage and ranked mass
            self.solve_concentration()
            self.solve_reactive_or_with_ET_concentration()
            self.solve_ranked_storage_and_mass()
            self.delete_inputs()
            # Very slow to add new axis...
            return [tuple((self.output_flux,
                           self.output_concentration,
                           self.ranked_storage[1:, :] / self.storage[1:, np.newaxis],
                           self.ranked_mass[1:, :] / (self.storage * self.storage_concentration)[1:, np.newaxis]
                           ))
                    ]

        self.delete_inputs()

        return [tuple((self.output_flux, None, None, None))]
