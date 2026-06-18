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
