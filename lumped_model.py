class LumpedModel():
    """
    This class defines a lumped hydrological model (i.e. a Unit in SuperflexPy).
    A unit is a collection of elements. It's task is to build the basic structure,
    connecting different elements.
    """

    def __init__(self, layers):
        """
        This is the initializer of the class LumpedModel.

        Parameters
        ----------
        layers : list(list(superflexpy.framework.element.BaseElement))
            This list defines the structure of the model. The elements are
            arranged in layers (upstream to downstream) and each layer can
            contain multiple elements.
        """

        self._layers = layers

    def set_rain_input(self, rain_input_flux, rain_input_concentration=None,
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
            Timeseries of rain mass TTD. Should be given for mass tracking
            computation. Should always be represented as dirac delta
            distributions.
        """

        self.rain_input = rain_input_flux

        if rain_input_concentration is not None:
            self.rain_input_concentration = rain_input_concentration
        else:
            self.rain_input_concentration = None

        self.rain_TTDs = rain_TTDs
        self.rain_mass_TTDs = rain_mass_TTD

    def run(self):
        """
        This method solves the Unit, solving each Element and putting together
        their outputs according to the structure.

        Returns
        -------
        Output : Tuple(numpy.array, numpy.array, numpy.ndarray, numpy.ndarray)
            Tuple with:
            - the output flux timeseries (index 0) as numpy.array;
            - the output concentration timeseries (index 1) as numpy.array;
            - the output water TTDs (index 2, timeseries of output TTD) as numpy.ndarray;
            - the output mass TTDs (index 3, timeseries of output TTD) as numpy.ndarray.
        """

        # Set the input flux and concentration from rain to the first layer
        # (the first layer must contain a single element)
        self._layers[0][0].set_input([tuple((self.rain_input, self.rain_input_concentration,
                                             self.rain_TTDs, self.rain_mass_TTDs))])

        # Loop over every layer and solve states and fluxes
        for i in range(1, len(self._layers)):

            # Case single Element in the layer
            if len(self._layers[i - 1]) == 1:
                outputs = self._layers[i - 1][0].get_output()

            # Case multiple Elements in the layer
            else:
                outputs = []

                # Loop over every element in the current layer (i-1)
                for el in self._layers[i - 1]:
                    # Produce outputs of the element for every timestep and update its states (if reservoir)
                    outputs.extend(el.get_output())

            # Fill the inputs of the elements in the next layer (i)
            ind_output = 0
            for el in self._layers[i]:
                el.set_input(outputs[ind_output:(ind_output+el.num_upstream)])
                ind_output += el.num_upstream

        # Return the output of the last layer
        # (the last layer must contain a single element)
        return self._layers[-1][0].get_output()[0]
