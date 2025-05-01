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

    def propagate(self, distance, grid_in, grid_out=None, abcd=None):
        self.update()

        return
