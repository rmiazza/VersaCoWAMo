import numpy as np
from base_elements import BaseElement


class Splitter(BaseElement):
    """
    This class implements a Splitter. A Splitter is used to connect one element
    to multiple elements downstream.
    """
    num_upstream = 1

    def __init__(self, weight, id):
        """
        This is the initializer of the class Splitter.

        Parameters
        ----------
        weight : list(float)
            The weight defines the fraction (between 0 and 1) of a flux that goes
            into a downstream element (e.g. [.5, .2, .3]). It should sum to 1.
        id : str
            Itentifier of the element.
        """

        BaseElement.__init__(self, id)
        self._weight = weight

    def set_input(self, input_tuple):
        """
        The method sets the input fluxes, concentration and TTDs of the element.

        Parameters
        ----------
        input_tuple : list(Tuple)
            List containing a single Tuple with:
            - the input flux timeseries (index 0) as numpy.array;
            - the input concentration timeseries (index 1) as numpy.array;
            - the input water TTDs (index 2, timeseries of input TTD) as
              numpy.ndarray.
        """
        self.input_flux = input_tuple[0][0]
        self.input_concentration = input_tuple[0][1]
        self.input_TTD = input_tuple[0][2]

    def get_output(self):
        """
        This method returns the output of the splitter to the downstream
        elements.

        Returns
        -------
        list(list(Tuple))
            List of lists containing a single Tuple (one for each split) with:
            - the output flux timeseries (index 0) as numpy.array;
            - the output concentration timeseries (index 1) as numpy.array;
            - the output water TTDs (index 2, timeseries of output TTD) as
              numpy.ndarray.
        """

        output = []

        for w in self._weight:
            output.append(tuple((self.input_flux * w, self.input_concentration, self.input_TTD)))

        # Reassign attributes to save memory
        self.input_flux = None
        self.input_concentration = None
        self.input_TTD = None

        return output


class Junction(BaseElement):
    """
    This class implements a Junction. A Junction is used to connect multiple
    element to a single element downstream.
    """

    def __init__(self, id, num_upstream):
        """
        This is the initializer of the class Junction.

        Parameters
        ----------
        id : str
            id of the element
        """

        BaseElement.__init__(self, id)
        self.num_upstream = num_upstream

    def set_input(self, input_tuple):
        """
        The method sets the input fluxes, concentration and TTDs of the element.

        Parameters
        ----------
        list(list(Tuple))
            List of lists containing a single Tuple (one for each input flux) with:
            - the input flux timeseries (index 0) as numpy.array;
            - the input concentration timeseries (index 1) as numpy.array;
            - the input water TTDs (index 2, timeseries of input TTD) as numpy.ndarray.
        """
        self.input_flux = np.array([i[0] for i in input_tuple])
        self.input_concentration = np.array([i[1] for i in input_tuple])
        self.input_TTDs = np.array([i[2] for i in input_tuple])

    def get_output(self):
        """
        This method returns the output of the junction to the downstream
        element.

        The input fluxes are summed up to produce the output flux, while
        the input concentrations are weighted by their respective input
        flux to produce the output concentration.

        Returns
        -------
        list(Tuple)
            List containing a single Tuple with:
            - the output flux timeseries (index 0) as numpy.array;
            - the output concentration timeseries (index 1) as numpy.array;
            - the output water TTDs (index 2, timeseries of output TTD) as numpy.ndarray.
        """
        def contains_none(array):
            return np.any(array == None)

        # Merge fluxes
        output_flux = np.sum(self.input_flux, axis=0)

        if contains_none(self.input_concentration) and contains_none(self.input_TTDs):
            # Flux only
            output = [tuple((output_flux, None, None))]

        elif contains_none(self.input_concentration) == False and contains_none(self.input_TTDs):
            # Flux and concentration only
            output_concentration = (
                np.sum(self.input_flux * self.input_concentration, axis=0)
                / output_flux
                )
            output = [tuple((output_flux, output_concentration, None))]

        else:
            # Flux, concentration and flux TTD
            output_concentration = (
                np.sum(self.input_flux * self.input_concentration, axis=0)
                / output_flux
                )
            # Very slow to add new axis...
            output_TTD = (
                np.sum(self.input_flux[:, :, np.newaxis] * self.input_TTDs, axis=0)
                / output_flux[:, np.newaxis]
                )
            output = [tuple((output_flux, output_concentration, output_TTD))]

        # Reassign attributes to save memory
        self.input_flux = None
        self.input_concentration = None
        self.input_TTDs = None

        return output


class Transparent(BaseElement):
    """
    This class implements a Transparent element. A Transparent element is used
    to fill gaps in the model structure. It just transfers the incoming fluxes
    to the output.
    """

    num_upstream = 1

    def __init__(self, id):
        """
        This is the initializer of the class Transparent.

        Parameters
        ----------
        id : str
            id of the element
        """
        BaseElement.__init__(self, id)

    def set_input(self, input_tuple):
        """
        The method sets the input fluxes and concentration of the element.

        Parameters
        ----------
        input_tuple : list(Tuple)
            List containing a single Tuple with:
            - the input flux timeseries (index 0) as numpy.array;
            - the input concentration timeseries (index 1) as numpy.array;
            - the input water TTDs (index 2, timeseries of input TTD) as numpy.ndarray.
        """
        self.input_flux = input_tuple[0][0]
        self.input_concentration = input_tuple[0][1]
        self.input_TTD = input_tuple[0][2]

    def get_output(self):
        """
        The method returns the output fluxes of the element.

        Returns
        ----------
        list(Tuple)
            List containing a single Tuple with:
            - the output flux timeseries (index 0) as numpy.array;
            - the output concentration timeseries (index 1) as numpy.array;
            - the output water TTDs (index 2, timeseries of output TTD) as numpy.ndarray.
        """
        output = [tuple((self.input_flux, self.input_concentration, self.input_TTD))]

        # Reassign up attributes to save memory
        self.input_flux = None
        self.input_concentration = None
        self.input_TTD = None

        return output
