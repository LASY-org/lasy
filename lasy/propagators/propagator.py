from abc import ABC, abstractmethod
from copy import deepcopy


class Propagator(ABC):
    """
    Base class for all propagators.

    A propagator advances the laser pulse in the z direction by some distance.
    """

    def __init__(self):
        pass

    @abstractmethod
    def update(self, dim, omega0):
        r"""
        Update the propagator.

        Typically called at the beginning of the propagate function, to update the propagator itself if required.
        If the update is costly, some machinery should be in place to do it only if required.
        Some propagators need to be updated if the distance to propagate changes.
        Others only need to be updated if the grid properties change, etc.

        Parameters
        ----------
        dim : string
            Dimensionality of the array. Options are:

            - ``'xyt'``: The laser pulse is represented on a 3D Cartesian grid.
            - ``'rt'`` : The laser pulse is represented on a 2D cylindrical grid.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.
        """
        self.dim = dim
        self.omega0 = omega0

    @abstractmethod
    def propagate(self, grid_in, dim, omega0, distance=None, grid_out=None):
        r"""
        Propagate field in the grid along axis z, for a certain distance.

        Parameters
        ----------
        grid_in: :class:`lasy.utils.Grid`
            Grid lasy object that contains the input field.

        dim : string
            Dimensionality of the array. Options are:

            - ``'xyt'``: The laser pulse is represented on a 3D Cartesian grid.
            - ``'rt'`` : The laser pulse is represented on a 2D cylindrical grid.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        distance : float (optional)
            Distance (in meters) over which the field will be propagated.

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
