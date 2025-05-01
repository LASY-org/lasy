from copy import deepcopy


class Propagator(object):
    """
    Base class for all propagators.
    """

    def __init__(self):

        return

    def update(self):

        return

    def propagate(self, distance, grid_in, grid_out=None, abcd=None):
        self.update()

        return deepcopy(grid_in)
