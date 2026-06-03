import numpy as np
from unit import Unit

class Node():
    """
    This class defines a Node. A node can be part of a network and it is a
    collection of Units. It's task is to sum the outputs of the Units,
    applying, if present, a routing.
    """

    def __init__(
            self, units, weights, area, id
            ):
        """
        This is the initializer of the class Node.

        Parameters
        ----------
        units : list(unit.Unit)
            List of Units contained in the Node.
        weights : list
            List of weights to be applied to the Units when putting together
            their outputs (handles the fact that all units are in normalized discharge I think RM).
            The order must be the same used in the units list.
            If a weight is a list, then different fluxes coming from the same
            unit have a different weight.
        area : float
            Influence area of the node. It is the net value: if a node has
            other nodes upstream, their area is not counted.
        id : str
            Identifier of the node. All the nodes of the framework must have an
            identifier.
        """

        self.id = id

        self._error_message = 'module : superflexPy, Node : {},'.format(id)
        self._error_message += ' Error message : '

        self.units = []
        for h in units:
            if not isinstance(h, Unit):
                message = '{}units must be instance of the Unit class'.format(self._error_message)
                raise TypeError(message)
            else:
                self.units.append(h)

        self.area = area  # why do we need it? RM

        if len(weights) != len(units):
            message = '{}number of weights does not match number of units'.format(self._error_message)
            raise TypeError(message)
        else:
            self._weights = weights

    def set_input(self, rain_input_flux, rain_input_concentration=None,
                  rain_TTDs=None, rain_mass_TTD=None):
        """
        This method sets the rain inputs to the model.

        Parameters
        ----------
        rain_input_flux : numpy.array
            Timeserie of rain input flux.
            Evapotranspiration should be given directly to the reservoirs
            undergoing ET or modeled directly in them.
        rain_input_concentration : numpy.array (optional)
            Timeserie of rain concentration. Set to None if not given/computed.
        rain_TTDs : numpy.ndarray
            Timeseries of rain water TTD. Should be given for age computation.
            Should always be represented as dirac delta distributions.
        rain_mass_TTDs : numpy.ndarray
            Timeseries of rain tracer mass TTD. Should be given for mass tracking
            computation. Should always be represented as dirac delta
            distributions.
        """

        self.rain_input = rain_input_flux
        self.rain_input_concentration = rain_input_concentration

        self.rain_TTDs = rain_TTDs
        self.rain_mass_TTDs = rain_mass_TTD
    
    def get_output(self):
        """
        This method solves the Node, solving each Unit and putting together
        their outputs according to the weight.

        Returns
        -------
        Output : Tuple(numpy.array, numpy.array, numpy.ndarray, numpy.ndarray)
            Tuple with:
            - the output flux timeseries (index 0) as numpy.array of length t;
            - the output concentration timeseries (index 1) as numpy.array of length t;
            - the output water TTDs (index 2, timeseries of output TTD) as numpy.ndarray of dimension t x T;
            - the output mass TTDs (index 3, timeseries of output TTD) as numpy.ndarray of dimension t x T.
        """

        # Set the inputs
        for h in self.units:
            h.set_unit_input(
                self.rain_input, self.rain_input_concentration,
                self.rain_TTDs, self.rain_mass_TTDs
                )
        
        if len(self._weights) == 1:  # case single unit in node
            return h.run_unit()  # gives back tuple with outputs from sigle unit
        
        # Case multiple units in node -> apply weights based on area
        for i, (h, w) in enumerate(zip(self.units, self._weights)):
            q, c, ttd_q, ttd_m = h.run_unit()

            if i == 0:
                out_q = q * w
                out_c = c * q * w if c is not None else None
                out_ttd_q = ttd_q * q[:, np.newaxis] * w if ttd_q is not None else None
                out_mflux = c * q * w if c is not None else None
                out_ttd_m = ttd_m * c[:, np.newaxis] * q[:, np.newaxis] * w if ttd_m is not None else None
            else:
                out_q += q * w
                out_c += c * q * w if out_c is not None else None
                out_ttd_q += ttd_q * q[:, np.newaxis] * w if out_ttd_q is not None else None
                out_mflux += c * q * w if out_mflux is not None else None
                out_ttd_m += ttd_m * c[:, np.newaxis] * q[:, np.newaxis] * w if out_ttd_m is not None else None
        
        if out_c is not None: out_c /= out_q
        if out_ttd_q is not None: out_ttd_q /= out_q[:, np.newaxis]
        if out_ttd_m is not None: out_ttd_m /= out_mflux[:, np.newaxis]

        return tuple((out_q, out_c, out_ttd_q, out_ttd_m))
    

    def external_routing(self, flux):
        """
        This methods applies the external routing to the fluxes. External
        routing is the one that affects the fluxes moving from the outflow of
        this node to the outflow of the one downstream. This function is used
        by the Network.

        Parameters
        ----------
        flux : list(numpy.ndarray)
            List of fluxes on which the routing has to be applied.
        """

        # TO DO: implement routing function to delay outputs to downstream node
        return flux