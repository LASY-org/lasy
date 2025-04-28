class Propagator(object):
    r"""
    Base class for all propagators.
    """

    def __init__(self):
        self.update()
        return

    def update(self, dim=None, omega0=None):
        """
        Update the propagator parameters.

        Parameters
        ----------
        dim : str
            The dimension of the laser grid. Can be 'rt' or 'xyt'.
        omega0 : float
            The central frequency of the laser.
        """
        self.dim = dim
        self.omega0 = omega0
        return

    def propagate(self, grid, distance=None, abcd=None):
        """
        Propagate the input grid using the specified method.
        """
        return