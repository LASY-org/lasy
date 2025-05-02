from copy import deepcopy


class Propagator(object):
    """
    Base class for all propagators.
    """

    def __init__(self):
        return

    def update(self, dim, omega0):
        """
        Create the propagator or check consistancy if already exists

        Parameters
        ----------
        dim : string
            Dimensionality of the array. Options are:

            - ``'xyt'``: The laser pulse is represented on a 3D grid:
                        Cartesian (x,y) transversely, and temporal (t) longitudinally.
            - ``'rt'`` : The laser pulse is represented on a 2D grid:
                        Cylindrical (r) transversely, and temporal (t) longitudinally.

        omega0 : float (in s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        """
        self.dim = dim
        self.omega0 = omega0
        return

    def propagate(self, distance, grid_in, dim, omega0, grid_out=None):
        """
        Method to propagate field from the grid.

        Parameters
        ----------
        distance : float (in meters)
            Distance over which the field will be propagated.

        grid_in: :class:`lasy.utils.Grid`
            Grid lasy object that contanins the input field.

        dim : string
            Dimensionality of the array. Options are:

            - ``'xyt'``: The laser pulse is represented on a 3D grid:
                        Cartesian (x,y) transversely, and temporal (t) longitudinally.
            - ``'rt'`` : The laser pulse is represented on a 2D grid:
                        Cylindrical (r) transversely, and temporal (t) longitudinally.

        omega0 : float (in s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        grid_out: :class:`lasy.utils.Grid` (optional)
            Grid lasy object where the output field will be written.
        """

        # Update is called only in this step, to reinitialize the propagator
        # if needed.
        self.update(dim, omega0)

        # This function explicitly returns a grid. This would let
        # laser.propagate have both grids, and potentially do some check there.
        # Can be rediscussed.
        grid_out = deepcopy(grid_in)

        return grid_out
