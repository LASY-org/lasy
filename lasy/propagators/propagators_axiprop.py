import numpy as np
from scipy.constants import c

from lasy.propagators import Propagator

from axiprop.containers import ScalarFieldEnvelope
from axiprop.lib import (
    PropagatorFFT2,
    PropagatorFFT2Fresnel,
    PropagatorResampling,
    PropagatorResamplingFresnel,
)

from axiprop.utils import import_from_lasy_grid


class MRTPropagator(Propagator):
    """
    Wrapper for PropagatorResampling
    """

    def propagate(self, grid, distance, grid_out=None, verbose=True):

        containers_in, m_axis = import_from_lasy_grid(grid, self.dim, self.omega0)

        if grid_out is None:
            grid_out = grid

        self.props_rt = []
        for im in range(m_axis.size):
            m = m_axis[im]
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

        field_3d = np.zeros_like(grid_out.temporal_field)

        for im in range(m_axis.size):
            prop_rt = self.props_rt[im]

            container_in = containers_in[im]
            Field_ft_new = prop_rt.step(
                container_in.Field_ft, distance, overwrite=False, show_progress=verbose
            )

            laser_loc = ScalarFieldEnvelope(
                container_in.k0, t_axis=container_in.t + distance / c
            )

            laser_loc.import_field_ft(
                Field_ft_new, r_axis=prop_rt.r_new, transform=True, make_copy=False
            )

            field_3d[im] = laser_loc.Field.T

        grid_out.set_temporal_field(field_3d)
        grid_out.axes[-1] = laser_loc.t
        grid_out.hi[-1] = laser_loc.t.max()
        grid_out.lo[-1] = laser_loc.t.min()


class MRTFresnelPropagator(Propagator):
    """
    Wrapper for PropagatorResamplingFresnel
    """

    def propagate(self, grid, distance, grid_out=None, verbose=True):

        containers_in, m_axis = import_from_lasy_grid(grid, self.dim, self.omega0)

        if grid_out is None:
            grid_out = grid

        self.props_rt = []
        for im in range(m_axis.size):
            m = m_axis[im]
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

        field_3d = np.zeros_like(grid_out.temporal_field)

        for im in range(m_axis.size):
            prop_rt = self.props_rt[im]

            container_in = containers_in[im]
            Field_ft_new = prop_rt.step(
                container_in.Field_ft, distance, overwrite=False, show_progress=verbose
            )

            laser_loc = ScalarFieldEnvelope(
                container_in.k0, t_axis=container_in.t + distance / c
            )

            laser_loc.import_field_ft(
                Field_ft_new, r_axis=prop_rt.r_new, transform=True, make_copy=False
            )

            field_3d[im] = laser_loc.Field.T

        grid_out.set_temporal_field(field_3d)
        grid_out.axes[-1] = laser_loc.t
        grid_out.hi[-1] = laser_loc.t.max()
        grid_out.lo[-1] = laser_loc.t.min()


class XYTPropagator(Propagator):
    """
    Wrapper for PropagatorFFT2
    """

    def propagate(self, grid, distance, grid_out=None, verbose=True):
        container_in = import_from_lasy_grid(grid, self.dim, self.omega0)

        if grid_out is None:
            grid_out = grid

        prop_xyt = PropagatorFFT2(
            x_axis=container_in.x,
            y_axis=container_in.y,
            kz_axis=container_in.k_freq,
            verbose=verbose,
        )

        Field_ft_new = prop_xyt.step(
            container_in.Field_ft, distance,
            overwrite=False, show_progress=verbose
        )

        laser_loc = ScalarFieldEnvelope(
            container_in.k0, t_axis=container_in.t + distance / c
        ).import_field_ft(
            Field_ft_new,
            r_axis=(prop_xyt.r, prop_xyt.x, prop_xyt.y),
            transform=True,
            make_copy=False,
        )

        grid_out.set_temporal_field(np.moveaxis(laser_loc.Field, 0, -1))
        grid_out.axes[-1] = laser_loc.t
        grid_out.hi[-1] = laser_loc.t.max()
        grid_out.lo[-1] = laser_loc.t.min()


class XYTFresnelPropagator(Propagator):
    """
    Wrapper for PropagatorFFT2Fresnel
    """

    def propagate(self, grid, distance, grid_out=None, verbose=True):

        container_in = import_from_lasy_grid(grid, self.dim, self.omega0)

        if grid_out is None:
            grid_out = grid

        x_axis_new = grid_out.axes[0]
        y_axis_new = grid_out.axes[1]

        prop_xyt = PropagatorFFT2Fresnel(
            dz=distance,
            x_axis=container_in.x,
            y_axis=container_in.y,
            x_axis_new=x_axis_new,
            y_axis_new=y_axis_new,
            kz_axis=container_in.k_freq,
            verbose=verbose,
        )

        Field_ft_new = prop_xyt.step(
            container_in.Field_ft, distance,
            overwrite=False, show_progress=verbose
        )

        laser_loc = ScalarFieldEnvelope(
            container_in.k0, t_axis=container_in.t + distance / c
        ).import_field_ft(
            Field_ft_new,
            r_axis=(prop_xyt.r, prop_xyt.x, prop_xyt.y),
            transform=True,
            make_copy=False,
        )

        grid_out.set_temporal_field(np.moveaxis(laser_loc.Field, 0, -1))
        grid_out.axes[-1] = laser_loc.t
        grid_out.hi[-1] = laser_loc.t.max()
        grid_out.lo[-1] = laser_loc.t.min()
