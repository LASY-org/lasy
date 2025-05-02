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


class MRTPropagator(Propagator):
    """
    Wrapper for PropagatorResampling
    """

    def update(self, dim, omega0, containers_in, grid_out, verbose):
        self.dim = dim
        self.omega0 = omega0

        make_propagator = True

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
                make_propagator = False

        if make_propagator:
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

    def propagate(
        self, distance, grid_in, dim, omega0, grid_out=None, verbose=True, nr_boundary=0
    ):
        containers_in, self.m_axis = import_from_lasy_grid(
            grid_in, dim, omega0, nr_boundary
        )

        if grid_out is None:
            grid_out = deepcopy(grid_in)

        field_3d = np.zeros_like(grid_out.temporal_field)

        self.update(dim, omega0, containers_in, grid_out, verbose)

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

        grid_out.set_temporal_field(field_3d)
        grid_out.axes[-1] = laser_loc.t
        grid_out.hi[-1] = laser_loc.t.max()
        grid_out.lo[-1] = laser_loc.t.min()

        return grid_out


class MRTFresnelPropagator(Propagator):
    """
    Wrapper for PropagatorResamplingFresnel
    """

    def update(self, distance, dim, omega0, containers_in, grid_out, verbose):
        self.dim = dim
        self.omega0 = omega0

        make_propagator = True

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
                make_propagator = False

        if make_propagator:
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

    def propagate(
        self, distance, grid_in, dim, omega0, grid_out=None, verbose=True, nr_boundary=0
    ):
        containers_in, self.m_axis = import_from_lasy_grid(
            grid_in, dim, omega0, nr_boundary
        )

        if grid_out is None:
            print("`grid_out` is required for this propagator")
            return grid_in

        field_3d = np.zeros_like(grid_out.temporal_field)

        self.update(distance, dim, omega0, containers_in, grid_out, verbose)

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

        grid_out.set_temporal_field(field_3d)
        grid_out.axes[-1] = laser_loc.t
        grid_out.hi[-1] = laser_loc.t.max()
        grid_out.lo[-1] = laser_loc.t.min()

        return grid_out


class XYTPropagator(Propagator):
    """
    Wrapper for PropagatorFFT2
    """

    def update(self, dim, omega0, container_in, verbose):
        self.dim = dim
        self.omega0 = omega0

        make_propagator = True

        if hasattr(self, "prop_xyt"):
            grid_changed = False
            try:
                assert np.allclose(container_in.x, self.prop_xyt.x)
                assert np.allclose(container_in.y, self.prop_xyt.y)
            except AssertionError:
                grid_changed = True

            if not grid_changed:
                make_propagator = False

        if make_propagator:
            self.prop_xyt = PropagatorFFT2(
                x_axis=container_in.x,
                y_axis=container_in.y,
                kz_axis=container_in.k_freq,
                verbose=verbose,
            )

    def propagate(self, distance, grid_in, dim, omega0, verbose=True, nr_boundary=0):
        container_in = import_from_lasy_grid(grid_in, dim, omega0, nr_boundary)
        grid_out = deepcopy(grid_in)

        self.update(dim, omega0, container_in, verbose)

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

        grid_out.set_temporal_field(np.moveaxis(laser_loc.Field, 0, -1))
        grid_out.axes[-1] = laser_loc.t
        grid_out.hi[-1] = laser_loc.t.max()
        grid_out.lo[-1] = laser_loc.t.min()

        return grid_out


class XYTFresnelPropagator(Propagator):
    """
    Wrapper for PropagatorFFT2Fresnel
    """

    def update(self, distance, dim, omega0, container_in, grid_out, verbose):
        self.dim = dim
        self.omega0 = omega0

        make_propagator = True

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
                make_propagator = False

        if make_propagator:
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
        self, distance, grid_in, dim, omega0, grid_out=None, verbose=True, nr_boundary=0
    ):
        container_in = import_from_lasy_grid(grid_in, dim, omega0, nr_boundary)

        if grid_out is None:
            print("`grid_out` is required for this propagator")
            return grid_in

        self.update(distance, dim, omega0, container_in, grid_out, verbose)

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

        grid_out.set_temporal_field(np.moveaxis(laser_loc.Field, 0, -1))
        grid_out.axes[-1] = laser_loc.t
        grid_out.hi[-1] = laser_loc.t.max()
        grid_out.lo[-1] = laser_loc.t.min()

        return grid_out
