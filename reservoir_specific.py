import numpy as np
from scipy.signal import fftconvolve
from reservoir_base import BaseReservoir


class JKPowerReservoir(BaseReservoir):
    """
    Specific reservoir class, implementing the Power Reservoir type described in:

    Kirchner, J. W. (2016). Aggregation in environmental systems–Part 2: Catchment
    mean transit times and young water fractions under hydrologic nonstationarity.
    Hydrology and Earth System Sciences, 20(1), 299-328.
    """
    def __init__(self, parameters, compute_ET=False, alpha_ET=1,
                 ET_timeseries=None, ETp_timeseries=None,
                 wilting_point=None, s_star=None,
                 **kwargs):
        """
        Parameters
        ----------
        parameters : dict
            Flow parameters: 'IN_bar', 'S_ref', 'b'.
        compute_ET : bool
            Whether to consider/compute evapotranspiration.
        alpha_ET : float
            Evapoconcentration factor (0=max evapoconc., 1=none).
        ET_timeseries : numpy.array
            External ET timeseries (used if ETp not provided).
        ETp_timeseries : numpy.array
            Potential ET timeseries for internal ET computation.
        wilting_point : float
            Wilting point storage for internal ET computation.
        s_star : float
            Storage at which ET becomes supply-independent.
        **kwargs
            All remaining parameters are passed to BaseReservoir:
            id, initial_storage, numerical_solver, dt,
            concentration_computation, reaction_type,
            initial_concentration, reaction_rate,
            equilibrium_concentration, age_computation,
            num_ages_tracked, initial_TTD.
        """
        # Forward base attributes to BaseReservoir
        super().__init__(**kwargs)

        # Store subclass-specific attributes first
        self._parameters = parameters
        self.compute_ET = compute_ET
        self.alpha_ET = alpha_ET
        self.compute_ET_internal = False

        # Evapotranspiration initialization
        if compute_ET:
            if ET_timeseries is None and ETp_timeseries is None:
                raise ValueError('Missing either ETp or ET timeseries input.')
            elif ETp_timeseries is not None:
                self.compute_ET_internal = True
                self.output_ETp = ETp_timeseries
                self.sw = wilting_point
                self.s_star = s_star
            else:
                self.output_ET = ET_timeseries

    def _define_ET(self):
        """
        Returns the internal ET function, which is computed as a piecewise linear
        function of storage, bounded between 0 at the wilting point and ETp
        above s_star.

        Returns
        -------
        callable
            ET_function(S, ETp) -> float
        """
        sw = self.sw
        s_star = self.s_star

        def ET_function(S, ETp):
            return ETp * np.minimum(1, np.maximum((S - sw) / (s_star - sw), 0))

        return ET_function


    def _define_storage_ode(self):
        """
        Returns the ODE function dS/dt = f(S, IN, ETp) for the Kirchner
        power-law reservoir. The ODE only returns dS/dt — ET evaluation
        is handled separately by the numerical solver via _define_ET.

        ODE without ET:
            dS/dt = IN - IN_bar * (S/S_ref)^b

        ODE with external ET (pre-subtracted from input before solving):
            dS/dt = IN_net - IN_bar * (S/S_ref)^b

        ODE with internal ET:
            dS/dt = IN - ET(S, ETp) - IN_bar * (S/S_ref)^b

        Returns
        -------
        callable
            ODE function returning dS/dt as a float.
        """
        IN_bar = self._parameters['IN_bar']
        S_ref = self._parameters['S_ref']
        b = self._parameters['b']

        if self.compute_ET_internal:
            ET_func = self._define_ET()

            def ode(S, IN, ETp):
                return IN - ET_func(S, ETp) - IN_bar * (S / S_ref)**b

        else:
            def ode(S, IN):
                return IN - IN_bar * (S / S_ref)**b

        return ode

    def solve_storage_and_output(self):
        """
        Solves the storage ODE and computes the output flux timeseries
        for the Kirchner power-law reservoir. Overrides the base class
        method to handle ET preprocessing before delegating to the solver.

        If ET is provided externally it is subtracted from the input flux
        before solving. If ET is computed internally, the ET function is
        passed to the solver which evaluates it at each timestep.
        """
        # Resolve IN_bar from mean input if not provided
        if self._parameters['IN_bar'] is None:
            self._parameters['IN_bar'] = np.mean(self.input_flux)

        ode = self._define_storage_ode()

        if self.compute_ET and not self.compute_ET_internal:
            # External ET: subtract from input before solving
            net_input = self.input_flux - self.output_ET
            self.storage, self.output_flux, _ = (
                self.numerical_solver.solve(
                    f=ode, input=net_input,
                    y0=self.initial_storage, dt=self.dt
                )
            )

        elif self.compute_ET_internal:
            # Internal ET: pass ODE, ET function, and ETp timeseries to solver
            ET_func = self._define_ET()
            self.storage, self.output_flux, self.output_ET = (
                self.numerical_solver.solve(
                    f=ode, input=self.input_flux,
                    y0=self.initial_storage, dt=self.dt,
                    ET_func=ET_func, ETp=self.output_ETp
                )
            )

        else:
            # No ET
            self.storage, self.output_flux, self.output_ET = (
                self.numerical_solver.solve(
                    f=ode, input=self.input_flux,
                    y0=self.initial_storage, dt=self.dt
                )
            )
    
    def initialize_ranked_storage(self):
        """
        Computes the initial ranked storage and "old" storage based on the
        solution of the steady-state eqation for randomly sampled reservoirs
        (considering average input rates and the average storage value).
        The transit time distribution (TTD) is obtained by convolving the
        local TTD since entry in the reservoir with the TTD of the input flux.

        If an initial TTD was given via the parameter initial_TTD, it will be
        assigned here.
        """
        if self.initial_TTD is not None:
            initial_S_old = self.initial_storage * (1 - np.sum(self.initial_TTD))
            return self.initial_storage * self.initial_TTD, initial_S_old

        else:
            IN_bar = np.mean(self.input_flux)
            S_bar = np.mean(self.storage)

            local_steady_state_TTD = (
                IN_bar / S_bar
                * np.exp(-IN_bar / S_bar * np.arange(self.num_ages_tracked))
            )

            initial_TTD = fftconvolve(self.input_TTD[0, :], local_steady_state_TTD, mode='full')[:self.num_ages_tracked]

            # Make sure the TTD is a pdf (avoid potential numerical errors)
            if np.sum(initial_TTD) > 1:
                initial_TTD = initial_TTD / np.sum(initial_TTD)

            initial_S_old = self.initial_storage * (1 - np.sum(initial_TTD))

            return self.initial_storage * initial_TTD, initial_S_old


class LinearReservoir(BaseReservoir):
    """
    Specific reservoir class, implementing a simple linear resrvoir.
    """
    def __init__(self, parameters, compute_ET=False, alpha_ET=1,
                 ET_timeseries=None, ETp_timeseries=None,
                 wilting_point=None, s_star=None,
                 **kwargs):
        """
        Parameters
        ----------
        parameters : dict
            Flow parameters: 'a'.
        compute_ET : bool
            Whether to consider/compute evapotranspiration.
        alpha_ET : float
            Evapoconcentration factor (0=max evapoconc., 1=none).
        ET_timeseries : numpy.array
            External ET timeseries (used if ETp not provided).
        ETp_timeseries : numpy.array
            Potential ET timeseries for internal ET computation.
        wilting_point : float
            Wilting point storage for internal ET computation.
        s_star : float
            Storage at which ET becomes supply-independent.
        **kwargs
            All remaining parameters are passed to BaseReservoir:
            id, initial_storage, numerical_solver, dt,
            concentration_computation, reaction_type,
            initial_concentration, reaction_rate,
            equilibrium_concentration, age_computation,
            num_ages_tracked, initial_TTD.
        """
        
        # Forward base attributes to BaseReservoir
        super().__init__(**kwargs)

        # Store subclass-specific attributes first
        self._parameters = parameters
        self.compute_ET = compute_ET
        self.alpha_ET = alpha_ET
        self.compute_ET_internal = False

        # Evapotranspiration initialization
        if compute_ET:
            if ET_timeseries is None and ETp_timeseries is None:
                raise ValueError('Missing either ETp or ET timeseries input.')
            elif ETp_timeseries is not None:
                self.compute_ET_internal = True
                self.output_ETp = ETp_timeseries
                self.sw = wilting_point
                self.s_star = s_star
            else:
                self.output_ET = ET_timeseries
    
    def _define_ET(self):
        """
        Returns the internal ET function, which is computed as a piecewise linear
        function of storage, bounded between 0 at the wilting point and ETp
        above s_star.

        Returns
        -------
        callable
            ET_function(S, ETp) -> float
        """
        sw = self.sw
        s_star = self.s_star

        def ET_function(S, ETp):
            return ETp * np.minimum(1, np.maximum((S - sw) / (s_star - sw), 0))

        return ET_function
    
    def _define_storage_ode(self):
        """
        Returns the ODE function dS/dt = f(S, IN, ETp) for the linear
        reservoir. The ODE only returns dS/dt — ET evaluation is handled
        separately by the numerical solver via _define_ET.

        ODE without ET:
            dS/dt = IN - a * S

        ODE with external ET (pre-subtracted from input before solving):
            dS/dt = IN_net - a * S

        ODE with internal ET:
            dS/dt = IN - ET(S, ETp) - a * S

        Returns
        -------
        callable
            ODE function returning dS/dt as a float.
        """
        a = self._parameters['a']

        if self.compute_ET_internal:
            ET_func = self._define_ET()

            def ode(S, IN, ETp):
                return IN - ET_func(S, ETp) - a * S

        else:
            def ode(S, IN):
                return IN - a * S

        return ode
    
    def solve_storage_and_output(self):
        """
        Solves the storage ODE and computes the output flux timeseries
        for the linear reservoir. Overrides the base class method to handle
        ET preprocessing before delegating to the solver.

        If ET is provided externally it is subtracted from the input flux
        before solving. If ET is computed internally, the ET function is
        passed to the solver which evaluates it at each timestep.
        """

        ode = self._define_storage_ode()

        if self.compute_ET and not self.compute_ET_internal:
            # External ET: subtract from input before solving
            net_input = self.input_flux - self.output_ET
            self.storage, self.output_flux, _ = (
                self.numerical_solver.solve(
                    f=ode, input=net_input,
                    y0=self.initial_storage, dt=self.dt
                )
            )

        elif self.compute_ET_internal:
            # Internal ET: pass ODE, ET function, and ETp timeseries to solver
            ET_func = self._define_ET()
            self.storage, self.output_flux, self.output_ET = (
                self.numerical_solver.solve(
                    f=ode, input=self.input_flux,
                    y0=self.initial_storage, dt=self.dt,
                    ET_func=ET_func, ETp=self.output_ETp
                )
            )

        else:
            # No ET
            self.storage, self.output_flux, self.output_ET = (
                self.numerical_solver.solve(
                    f=ode, input=self.input_flux,
                    y0=self.initial_storage, dt=self.dt
                )
            )

    def initialize_ranked_storage(self):
        """
        Computes the initial ranked storage and "old" storage based on the
        solution of the steady-state eqation for randomly sampled reservoirs
        (considering average input rates and the average storage value).
        The transit time distribution (TTD) is obtained by convolving the
        local TTD since entry in the reservoir with the TTD of the input flux.

        If an initial TTD was given via the parameter initial_TTD, it will be
        assigned here.
        """
        if self.initial_TTD is not None:
            initial_S_old = self.initial_storage * (1 - np.sum(self.initial_TTD))
            return self.initial_storage * self.initial_TTD, initial_S_old

        else:
            IN_bar = np.mean(self.input_flux)
            S_bar = np.mean(self.storage)

            local_steady_state_TTD = (
                IN_bar / S_bar
                * np.exp(-IN_bar / S_bar * np.arange(self.num_ages_tracked))
            )

            initial_TTD = fftconvolve(self.input_TTD[0, :], local_steady_state_TTD, mode='full')[:self.num_ages_tracked]

            # Make sure the TTD is a pdf (avoid potential numerical errors)
            if np.sum(initial_TTD) > 1:
                initial_TTD = initial_TTD / np.sum(initial_TTD)

            initial_S_old = self.initial_storage * (1 - np.sum(initial_TTD))

            return self.initial_storage * initial_TTD, initial_S_old


class PBPowerReservoir(BaseReservoir):
    """
    Specific reservoir class, implementing a power law reservoir with root zone
    pore volume normalization, and residual storage that doesn't contribute to
    runoff generation but influences tracer concentrations.
    """
    def __init__(self, parameters, compute_ET=False, alpha_ET=1,
                 ET_timeseries=None, ETp_timeseries=None,
                 wilting_point=None, s_star=None,
                 **kwargs):
        """
        Parameters
        ----------
        parameters : dict
            Flow parameters: 'a', 'b', 'W', 'n', 'Zr'.
        compute_ET : bool
            Whether to consider/compute evapotranspiration.
        alpha_ET : float
            Evapoconcentration factor (0=max evapoconc., 1=none).
        ET_timeseries : numpy.array
            External ET timeseries (used if ETp not provided).
        ETp_timeseries : numpy.array
            Potential ET timeseries for internal ET computation.
        wilting_point : float
            Wilting point soil moisture for internal ET computation.
        s_star : float
            Soil moisture at which ET becomes supply-independent.            
        **kwargs
            All remaining parameters are passed to BaseReservoir:
            id, initial_storage, numerical_solver, dt,
            concentration_computation, reaction_type,
            initial_concentration, reaction_rate,
            equilibrium_concentration, age_computation,
            num_ages_tracked, initial_TTD.
            The "storage" state variable is the sum of both the
            dynamic and residual storage.
        """
        
        # Forward base attributes to BaseReservoir
        super().__init__(**kwargs)

        # Store subclass-specific attributes first
        self._parameters = parameters
        self.compute_ET = compute_ET
        self.alpha_ET = alpha_ET
        self.compute_ET_internal = False

        # Evapotranspiration initialization
        if compute_ET:
            if ET_timeseries is None and ETp_timeseries is None:
                raise ValueError('Missing either ETp or ET timeseries input.')
            elif ETp_timeseries is not None:
                self.compute_ET_internal = True
                self.output_ETp = ETp_timeseries
                self.sw = wilting_point
                self.s_star = s_star
            else:
                self.output_ET = ET_timeseries
    
    def _define_ET(self):
        """
        Returns the internal ET function, which is computed as a piecewise linear
        function of storage (/!\ not saturation /!\, for consistency with the numerical
        solvers), bounded between 0 at the wilting point and ETp above s_star.

        Returns
        -------
        callable
            ET_function(S, ETp) -> float
        """
        sw = self.sw
        s_star = self.s_star
        W = self._parameters['W']
        n = self._parameters['n']
        Zr = self._parameters['Zr']

        def ET_function(S, ETp):
            return ETp * np.minimum(1, np.maximum(((S-W)/(n*Zr) - sw) / (s_star - sw), 0))

        return ET_function
    
    def _define_storage_ode(self):
        """
        Returns the ODE function dS/dt = f(S, IN, ETp) for the power law
        reservoir. The ODE only returns dS/dt — ET evaluation is handled
        separately by the numerical solver via _define_ET.

        ODE without ET:
            dS/dt = IN - a * ((S-W)/(nZr))^b

        ODE with external ET (pre-subtracted from input before solving):
            dS/dt = IN_net - a * ((S-W)/(nZr))^b

        ODE with internal ET:
            dS/dt = IN - ET(S, ETp) - a * ((S-W)/(nZr))^b

        Returns
        -------
        callable
            ODE function returning dS/dt as a float.
        """
        a = self._parameters['a']
        b = self._parameters['b']
        W = self._parameters['W']
        n = self._parameters['n']
        Zr = self._parameters['Zr']

        if self.compute_ET_internal:
            ET_func = self._define_ET()

            def ode(S, IN, ETp):
                return IN - ET_func(S, ETp) - a * ((S-W)/(n*Zr))**b

        else:
            def ode(S, IN):
                return IN - a * ((S-W)/(n*Zr))**b

        return ode
    
    def solve_storage_and_output(self):
        """
        Solves the (dynamic) storage ODE and computes the output flux timeseries
        for the power reservoir. Overrides the base class method to handle
        ET preprocessing before delegating to the solver.

        If ET is provided externally it is subtracted from the input flux
        before solving. If ET is computed internally, the ET function is
        passed to the solver which evaluates it at each timestep.
        """

        ode = self._define_storage_ode()

        if self.compute_ET and not self.compute_ET_internal:
            # External ET: subtract from input before solving
            net_input = self.input_flux - self.output_ET
            self.storage, self.output_flux, _ = (
                self.numerical_solver.solve(
                    f=ode, input=net_input,
                    y0=self.initial_storage, dt=self.dt
                )
            )

        elif self.compute_ET_internal:
            # Internal ET: pass ODE, ET function, and ETp timeseries to solver
            ET_func = self._define_ET()
            self.storage, self.output_flux, self.output_ET = (
                self.numerical_solver.solve(
                    f=ode, input=self.input_flux,
                    y0=self.initial_storage, dt=self.dt,
                    ET_func=ET_func, ETp=self.output_ETp
                )
            )

        else:
            # No ET
            self.storage, self.output_flux, self.output_ET = (
                self.numerical_solver.solve(
                    f=ode, input=self.input_flux,
                    y0=self.initial_storage, dt=self.dt
                )
            )

    def initialize_ranked_storage(self):
        """
        Computes the initial ranked storage and "old" storage based on the
        solution of the steady-state eqation for randomly sampled reservoirs
        (considering average input rates and the average storage value).
        The transit time distribution (TTD) is obtained by convolving the
        local TTD since entry in the reservoir with the TTD of the input flux.

        If an initial TTD was given via the parameter initial_TTD, it will be
        assigned here.
        """
        if self.initial_TTD is not None:
            initial_S_old = self.initial_storage * (1 - np.sum(self.initial_TTD))
            return self.initial_storage * self.initial_TTD, initial_S_old

        else:
            IN_bar = np.mean(self.input_flux)
            S_bar = np.mean(self.storage)

            local_steady_state_TTD = (
                IN_bar / S_bar
                * np.exp(-IN_bar / S_bar * np.arange(self.num_ages_tracked))
            )

            initial_TTD = (
                fftconvolve(
                    self.input_TTD[0, :],
                    local_steady_state_TTD,
                    mode='full'
                    )[:self.num_ages_tracked]
            )

            # Make sure the TTD is a pdf (avoid potential numerical errors)
            if np.sum(initial_TTD) > 1:
                initial_TTD = initial_TTD / np.sum(initial_TTD)

            initial_S_old = self.initial_storage * (1 - np.sum(initial_TTD))

            return self.initial_storage * initial_TTD, initial_S_old


class PBLinearReservoir(BaseReservoir):
    """
    Specific reservoir class, implementing a simple linear resrvoir with
    no evapotranspiration, and with residual storage that doesn't contribute
    to runoff generation but influences tracer concentrations.
    """
    def __init__(self, parameters, **kwargs):
        """
        Parameters
        ----------
        parameters : dict
            Flow parameters: 'a', 'W'.
        **kwargs
            All remaining parameters are passed to BaseReservoir:
            id, initial_storage, numerical_solver, dt,
            concentration_computation, reaction_type,
            initial_concentration, reaction_rate,
            equilibrium_concentration, age_computation,
            num_ages_tracked, initial_TTD.
        """
        
        # Forward base attributes to BaseReservoir
        super().__init__(**kwargs)

        # Store subclass-specific attributes first
        self._parameters = parameters
    
    def _define_storage_ode(self):
        """
        Returns the ODE function dS/dt = f(S, IN) for the linear
        reservoir.

        ODE:
            dS/dt = IN - a * (S-W)

        Returns
        -------
        callable
            ODE function returning dS/dt as a float.
        """
        a = self._parameters['a']
        W = self._parameters['W']
        
        def ode(S, IN):
            return IN - a * (S - W)

        return ode
    
    def solve_storage_and_output(self):
        """
        Solves the storage ODE and computes the output flux timeseries
        for the linear reservoir. Overrides the base class method to handle
        ET preprocessing before delegating to the solver.
        """

        ode = self._define_storage_ode()

        self.storage, self.output_flux, self.output_ET = (
            self.numerical_solver.solve(
                f=ode, input=self.input_flux,
                y0=self.initial_storage, dt=self.dt
            )
        )

    def initialize_ranked_storage(self):
        """
        Computes the initial ranked storage and "old" storage based on the
        solution of the steady-state eqation for randomly sampled reservoirs
        (considering average input rates and the average storage value).
        The transit time distribution (TTD) is obtained by convolving the
        local TTD since entry in the reservoir with the TTD of the input flux.

        If an initial TTD was given via the parameter initial_TTD, it will be
        assigned here.
        """
        if self.initial_TTD is not None:
            initial_S_old = self.initial_storage * (1 - np.sum(self.initial_TTD))
            return self.initial_storage * self.initial_TTD, initial_S_old

        else:
            IN_bar = np.mean(self.input_flux)
            S_bar = np.mean(self.storage)

            local_steady_state_TTD = (
                IN_bar / S_bar
                * np.exp(-IN_bar / S_bar * np.arange(self.num_ages_tracked))
            )

            initial_TTD = fftconvolve(self.input_TTD[0, :], local_steady_state_TTD, mode='full')[:self.num_ages_tracked]

            # Make sure the TTD is a pdf (avoid potential numerical errors)
            if np.sum(initial_TTD) > 1:
                initial_TTD = initial_TTD / np.sum(initial_TTD)

            initial_S_old = self.initial_storage * (1 - np.sum(initial_TTD))

            return self.initial_storage * initial_TTD, initial_S_old
