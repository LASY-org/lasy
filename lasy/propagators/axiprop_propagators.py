from copy import deepcopy

import numpy as np
from axiprop.containers import ScalarFieldEnvelope
from axiprop.lib import (
    PropagatorFFT2,
    PropagatorFFT2Fresnel,
    PropagatorResampling,
    PropagatorResamplingFresnel,
)
from axiprop.utils import import_from_lasy_grid
from scipy.constants import c

from .propagator import Propagator


class AxipropPropagator(Propagator):
    """
    Axiprop's non-paraxial propagator.

    This class wraps around Axiprop's PropagatorFFT2 and PropagatorResampling, for 3D cartesian and 2D cylindrical, respectively.
    """

    def update(self, dim, omega0, containers_in, grid_out=None, verbose=False):
        r"""
        Initialize or update the propagator if needed.

        Parameters
        ----------
        dim : string
            Dimensionality of the array. Options are:
            - ``'xyt'``: Laser pulse represented on a 3D Cartesian grid.
            - ``'rt'`` : Laser pulse represented on a 2D cylindrical grid.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        containers_in : Axiprop container(s)
            An Axiprop container (dim='xyt'), or list of containers (dim='rt', 1 element per mode), with the data of laser to propagate.

        grid_out : Grid object (optional)
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.
            Only supported for rt geometry.

        verbose : boolean (optional)
            Whether to print intermediate steps.
        """
        self.dim = dim
        self.omega0 = omega0
        self.make_propagator = True

        if self.dim == "rt":
            self._update_mrt(omega0, containers_in, grid_out, verbose)
        else:
            assert grid_out is None, "grid_out not supported in xyt, use None"
            self._update_xyt(omega0, containers_in, verbose)

    def _update_mrt(self, omega0, containers_in, grid_out, verbose):
        r"""
        Initialize or update the propagator if needed.

        Parameters
        ----------
        omega0 : float (in s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        containers_in : Axiprop container(s)
            A list of Axiprop containers, with the data of laser to propagate.

        grid_out : Grid object (optional)
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.

        verbose : boolean (optional)
            Whether to print intermediate steps.
        """
        if hasattr(self, "props_rt"):
            grid_changed = False
            for im in range(self.m_axis.size):
                container_in = containers_in[im]
                prop_rt = self.props_rt[im]
                try:
                    assert np.allclose(container_in.r, prop_rt.r)
                    assert np.allclose(grid_out.axes[0], prop_rt.r_new)
                except AssertionError:
                    grid_changed = True

            if not grid_changed:
                self.make_propagator = False

        if self.make_propagator:
            self.props_rt = []
            for im in range(self.m_axis.size):
                m = self.m_axis[im]
                container_in = containers_in[im]
                self.props_rt.append(
                    PropagatorResampling(
                        r_axis=container_in.r,
                        kz_axis=container_in.k_freq,
                        r_axis_new=grid_out.axes[0],
                        mode=m,
                        verbose=verbose,
                    )
                )

    def _update_xyt(self, omega0, container_in, verbose):
        r"""
        Initialize or update the propagator if needed.

        Parameters
        ----------
        omega0 : float (in s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        containers_in : Axiprop container(s)
            An Axiprop container, with the data of laser to propagate.

        grid_out : Grid object (optional)
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.
            Only supported for rt geometry.

        verbose : boolean (optional)
            Whether to print intermediate steps.
        """
        if hasattr(self, "prop_xyt"):
            grid_changed = False
            try:
                assert np.allclose(container_in.x, self.prop_xyt.x)
                assert np.allclose(container_in.y, self.prop_xyt.y)
            except AssertionError:
                grid_changed = True

            if not grid_changed:
                self.make_propagator = False

        if self.make_propagator:
            self.prop_xyt = PropagatorFFT2(
                x_axis=container_in.x,
                y_axis=container_in.y,
                kz_axis=container_in.k_freq,
                verbose=verbose,
            )

    def propagate(
        self,
        grid_in,
        dim,
        omega0,
        distance=None,
        grid_out=None,
        verbose=True,
        nr_boundary=0,
    ):
        r"""
        Propagate laser pulse in z direction by a given distance.

        Currently, the propagation is assumed to take place in vacuum.
        This propagator is non-paraxial.

        Parameters
        ----------
        grid_in : Grid
            Grid object containing the laser to propagate.

        dim : string
            Dimensionality of the array. Options are:
            - ``'xyt'``: Laser pulse represented on a 3D Cartesian grid.
            - ``'rt'`` : Laser pulse represented on a 2D cylindrical grid.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        distance : scalar (optional)
            Distance by which the laser is propagated.

        grid_out : Grid object (optional)
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.
            Only supported for 'rt' geometry.

        verbose : boolean (optional, default False)
            Whether to print intermediate steps.

        nr_boundary : int (optional, default 0)
            Number of grid points for absorbing boundary condition.

        Returns
        -------
        Grid object with laser data after propagation.
        """
        assert distance is not None
        if dim == "xyt":
            assert grid_out is None, (
                "grid_out not yet supported for xyt, please use None"
            )
        if grid_out is None:
            grid_out = deepcopy(grid_in)

        if dim == "rt":
            field = self._propagate_mrt(
                distance, grid_in, omega0, grid_out, verbose, nr_boundary
            )
        else:
            field = self._propagate_xyt(distance, grid_in, omega0, verbose, nr_boundary)
        grid_out.position += distance
        grid_out.set_temporal_field(field)

        return grid_out

    def _propagate_mrt(
        self, distance, grid_in, omega0, grid_out, verbose=True, nr_boundary=0
    ):
        r"""
        Propagate laser pulse in z direction by a given distance.

        Currently, the propagation is assumed to take place in vacuum.
        This propagator is non-paraxial.

        Parameters
        ----------
        distance : scalar
            Distance by which the laser is propagated.

        grid_in : Grid
            Grid object containing the laser to propagate.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        grid_out : Grid object (optional)
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.

        verbose : boolean (optional, default False)
            Whether to print intermediate steps.

        nr_boundary : int (optional, default 0)
            Number of grid points for absorbing boundary condition.

        Returns
        -------
        field : ndarray with laser envelope in temporal representation.
        """
        containers_in, self.m_axis = import_from_lasy_grid(
            grid_in, "rt", omega0, nr_boundary
        )

        field_3d = np.zeros_like(grid_out.temporal_field)

        self.update("rt", omega0, containers_in, grid_out, verbose)

        for im in range(self.m_axis.size):
            prop_rt = self.props_rt[im]

            container_in = containers_in[im]
            Field_ft_new = prop_rt.step(
                container_in.Field_ft, distance, overwrite=False, show_progress=verbose
            )

            laser_loc = ScalarFieldEnvelope(
                container_in.k0, container_in.t + distance / c, nr_boundary
            )

            laser_loc.import_field_ft(
                Field_ft_new, r_axis=prop_rt.r_new, transform=True, make_copy=False
            )

            field_3d[im] = laser_loc.Field.T

        return field_3d

    def _propagate_xyt(self, distance, grid_in, omega0, verbose=True, nr_boundary=0):
        r"""
        Propagate laser pulse in z direction by a given distance.

        Currently, the propagation is assumed to take place in vacuum.
        This propagator is non-paraxial.

        Parameters
        ----------
        distance : scalar
            Distance by which the laser is propagated.

        grid_in : Grid
            Grid object containing the laser to propagate.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        verbose : boolean (optional, default False)
            Whether to print intermediate steps.

        nr_boundary : int (optional, default 0)
            Number of grid points for absorbing boundary condition.

        Returns
        -------
        field : ndarray with laser envelope in temporal representation.
        """
        container_in = import_from_lasy_grid(grid_in, "xyt", omega0, nr_boundary)

        self.update("xyt", omega0, container_in, verbose=verbose)

        Field_ft_new = self.prop_xyt.step(
            container_in.Field_ft, distance, overwrite=False, show_progress=verbose
        )

        laser_loc = ScalarFieldEnvelope(
            container_in.k0, container_in.t + distance / c, nr_boundary
        ).import_field_ft(
            Field_ft_new,
            r_axis=(self.prop_xyt.r, self.prop_xyt.x, self.prop_xyt.y),
            transform=True,
            make_copy=False,
        )

        return np.moveaxis(laser_loc.Field, 0, -1)


class AxipropFresnelPropagator(Propagator):
    """
    Axiprop's paraxial Fresnel propagator.

    This class wraps around Axiprop's PropagatorFFT2Fresnel and PropagatorResamplingFresnel, for 3D cartesian and 2D cylindrical, respectively.
    """

    def update(self, distance, dim, omega0, containers_in, grid_out, verbose=False):
        r"""
        Initialize or update the propagator if needed.

        Parameters
        ----------
        distance : scalar
            Distance over which to propagate the laser pulse.

        dim : string
            Dimensionality of the array. Options are:
            - ``'xyt'``: Laser pulse represented on a 3D Cartesian grid.
            - ``'rt'`` : Laser pulse represented on a 2D cylindrical grid.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        containers_in : Axiprop container(s)
            An Axiprop container (dim='xyt'), or list of containers (dim='rt', 1 element per mode), with the data of laser to propagate.

        grid_out : Grid object
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.

        verbose : boolean (optional)
            Whether to print intermediate steps.
        """
        self.dim = dim
        self.omega0 = omega0
        self.make_propagator = True

        if self.dim == "rt":
            self._update_mrt(distance, omega0, containers_in, grid_out, verbose)
        else:
            self._update_xyt(distance, omega0, containers_in, grid_out, verbose)

    def _update_mrt(self, distance, omega0, containers_in, grid_out, verbose):
        r"""
        Initialize or update the propagator if needed.

        Parameters
        ----------
        distance : scalar
            Distance over which to propagate the laser pulse.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        containers_in : Axiprop container(s)
            A list of Axiprop containers (1 element per mode), with the data of laser to propagate.

        grid_out : Grid object
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.

        verbose : boolean (optional)
            Whether to print intermediate steps.
        """
        if hasattr(self, "props_rt"):
            grid_changed = False
            for im in range(self.m_axis.size):
                container_in = containers_in[im]
                prop_rt = self.props_rt[im]
                try:
                    assert distance == self.distance
                    assert np.allclose(container_in.r, prop_rt.r)
                    assert np.allclose(grid_out.axes[0], prop_rt.r_new)
                except AssertionError:
                    grid_changed = True

            if not grid_changed:
                self.make_propagator = False

        if self.make_propagator:
            self.props_rt = []
            self.distance = distance
            for im in range(self.m_axis.size):
                m = self.m_axis[im]
                container_in = containers_in[im]
                self.props_rt.append(
                    PropagatorResamplingFresnel(
                        dz=distance,
                        r_axis=container_in.r,
                        kz_axis=container_in.k_freq,
                        r_axis_new=grid_out.axes[0],
                        mode=m,
                        verbose=verbose,
                    )
                )

    def _update_xyt(self, distance, omega0, container_in, grid_out, verbose):
        r"""
        Initialize or update the propagator if needed.

        Parameters
        ----------
        distance : scalar
            Distance over which to propagate the laser pulse.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        containers_in : Axiprop container(s)
            An Axiprop container with the data of laser to propagate.

        grid_out : Grid object
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.

        verbose : boolean (optional)
            Whether to print intermediate steps.
        """
        if hasattr(self, "prop_xyt"):
            grid_changed = False
            try:
                assert np.allclose(self.distance, distance)
                assert np.allclose(container_in.x, self.prop_xyt.x0)
                assert np.allclose(container_in.y, self.prop_xyt.y0)
                assert np.allclose(grid_out.axes[0], self.prop_xyt.x)
                assert np.allclose(grid_out.axes[1], self.prop_xyt.y)
            except AssertionError:
                grid_changed = True

            if not grid_changed:
                self.make_propagator = False

        if self.make_propagator:
            self.distance = distance
            self.prop_xyt = PropagatorFFT2Fresnel(
                dz=distance,
                x_axis=container_in.x,
                y_axis=container_in.y,
                x_axis_new=grid_out.axes[0],
                y_axis_new=grid_out.axes[1],
                kz_axis=container_in.k_freq,
                verbose=verbose,
            )

    def propagate(
        self,
        grid_in,
        dim,
        omega0,
        distance=None,
        grid_out=None,
        verbose=True,
        nr_boundary=0,
    ):
        r"""
        Propagate laser pulse in z direction by a given distance.

        Currently, the propagation is assumed to take place in vacuum.
        This is a Fresnel, paraxial propagator.

        Parameters
        ----------
        distance : scalar
            Distance by which the laser is propagated.

        grid_in : Grid
            Grid object containing the laser to propagate.

        dim : string
            Dimensionality of the array. Options are:
            - ``'xyt'``: Laser pulse represented on a 3D Cartesian grid.
            - ``'rt'`` : Laser pulse represented on a 2D cylindrical grid.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        grid_out : Grid object (optional)
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.

        verbose : boolean (optional, default False)
            Whether to print intermediate steps.

        nr_boundary : int (optional, default 0)
            Number of grid points for absorbing boundary condition.

        Returns
        -------
        Grid object with laser data after propagation.
        """
        if grid_out is None:
            print("`grid_out` is required for this propagator")
            return grid_in

        if dim == "rt":
            field = self._propagate_mrt(
                distance, grid_in, omega0, grid_out, verbose, nr_boundary
            )
        else:
            field = self._propagate_xyt(
                distance, grid_in, omega0, grid_out, verbose, nr_boundary
            )

        grid_out.set_temporal_field(field)
        grid_out.position += distance

        return grid_out

    def _propagate_mrt(
        self, distance, grid_in, omega0, grid_out, verbose=True, nr_boundary=0
    ):
        r"""
        Propagate laser pulse in z direction by a given distance.

        Currently, the propagation is assumed to take place in vacuum.
        This propagator is a Fresnel, paraxial propagator.

        Parameters
        ----------
        distance : scalar
            Distance by which the laser is propagated.

        grid_in : Grid
            Grid object containing the laser to propagate.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        grid_out : Grid object
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.

        verbose : boolean (optional, default False)
            Whether to print intermediate steps.

        nr_boundary : int (optional, default 0)
            Number of grid points for absorbing boundary condition.

        Returns
        -------
        field : ndarray with laser envelope in temporal representation.
        """
        containers_in, self.m_axis = import_from_lasy_grid(
            grid_in, "rt", omega0, nr_boundary
        )

        field_3d = np.zeros_like(grid_out.temporal_field)

        self.update(distance, "rt", omega0, containers_in, grid_out, verbose)

        for im in range(self.m_axis.size):
            prop_rt = self.props_rt[im]

            container_in = containers_in[im]
            Field_ft_new = prop_rt.step(
                container_in.Field_ft, distance, overwrite=False, show_progress=verbose
            )

            laser_loc = ScalarFieldEnvelope(
                container_in.k0, container_in.t + distance / c, nr_boundary
            )

            laser_loc.import_field_ft(
                Field_ft_new, r_axis=prop_rt.r_new, transform=True, make_copy=False
            )

            field_3d[im] = laser_loc.Field.T

        return field_3d

    def _propagate_xyt(
        self, distance, grid_in, omega0, grid_out, verbose=True, nr_boundary=0
    ):
        r"""
        Propagate laser pulse in z direction by a given distance.

        Currently, the propagation is assumed to take place in vacuum.
        This propagator is a Fresnel, paraxial propagator.

        Parameters
        ----------
        distance : scalar
            Distance by which the laser is propagated.

        grid_in : Grid
            Grid object containing the laser to propagate.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        grid_out : Grid object
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.

        verbose : boolean (optional, default False)
            Whether to print intermediate steps.

        nr_boundary : int (optional, default 0)
            Number of grid points for absorbing boundary condition.

        Returns
        -------
        field : ndarray with laser envelope in temporal representation.
        """
        container_in = import_from_lasy_grid(grid_in, "xyt", omega0, nr_boundary)

        self.update(distance, "xyt", omega0, container_in, grid_out, verbose)

        Field_ft_new = self.prop_xyt.step(
            container_in.Field_ft, distance, overwrite=False, show_progress=verbose
        )

        laser_loc = ScalarFieldEnvelope(
            container_in.k0, container_in.t + distance / c, nr_boundary
        ).import_field_ft(
            Field_ft_new,
            r_axis=(self.prop_xyt.r, self.prop_xyt.x, self.prop_xyt.y),
            transform=True,
            make_copy=False,
        )

        return np.moveaxis(laser_loc.Field, 0, -1)
