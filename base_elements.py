class BaseElement():
    def __init__(self, id):
        """
        This is the initializer of the class BaseElement. This is
        an abstract class from which each element inherits. Allows
        to implement shared attributes and methods in elements.

        Parameters
        ----------
        id : str
            Identifier of the element (not used).
        """

        self._id = id