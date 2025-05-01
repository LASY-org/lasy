from copy import deepcopy


class Propagator(object):
    """
    Base class for all propagators.
    """

    def __init__(self):
        return

    def update(self, dim, omega0):
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
        grid_out = copy.deepcopy(grid_in)

        return grid_out
