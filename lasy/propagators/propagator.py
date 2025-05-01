class Propagator(object):
    r"""
    Base class for all propagators.
    """

    def __init__(self):
        return

    def update(self, dim, omega0):
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

    def propagate(self, distance, grid_in, dim, omega0):
        # Update is called only in this step, to reinitialize the propagator
        # if needed.
        self.update(dim, omega0)

        # This function explicitly returns a grid. This would let
        # laser.propagate have both grids, and potentially do some check there.
        # Can be rediscussed.
        grid_out = deepcopy(grid_in)

        return grid_out
