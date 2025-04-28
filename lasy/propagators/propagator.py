class Propagator(object):
    """
    Base class for all propagators.
    """

    def __init__(self):
        self.update()
        return

    def update(self, dim=None, omega0=None):
        self.dim = dim
        self.omega0 = omega0
        return

    def propagate(self, grid, distance=None, abcd=None):
        return
