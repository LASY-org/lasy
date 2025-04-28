import numpy as np
from scipy.constants import c

from lasy.propagators import Propagator

from axiprop.lib import PropagatorResampling
from axiprop.lib import PropagatorResamplingFresnel
from axiprop.lib import PropagatorFFT2
from axiprop.lib import PropagatorFFT2Fresnel
from axiprop.utils import import_from_lasy
from axiprop.containers import ScalarFieldEnvelope


class MRTPropagator(Propagator):
    """
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        return

    def update(self, laser_in):
        self.laser_in = laser_in
        return

    def propagate(self, distance, laser_out=None):
        laser_in = self.laser_in
        verbose = self.verbose

        containers_in, m_axis = import_from_lasy(laser_in)

        if laser_out is None:
            laser_out = laser_in

        self.props_rt = []
        for im in range(m_axis.size):
            m = m_axis[im]
            container_in = containers_in[im]
            self.props_rt.append(
                PropagatorResampling(
                    r_axis=container_in.r,
                    kz_axis=container_in.k_freq,
                    r_axis_new=laser_out.grid.axes[0],
                    mode=m, verbose=verbose
                )
            )

        field_3d = np.zeros_like(laser_out.grid.temporal_field)

        for im in range(m_axis.size):
            prop_rt = self.props_rt[im]

            m = m_axis[im]
            container_in = containers_in[im]
            Field_ft_new = prop_rt.step(
                container_in.Field_ft,
                distance, overwrite=False,
                show_progress=verbose
            )

            laser_loc = ScalarFieldEnvelope(
                container_in.k0,
                t_axis=container_in.t + distance/c
            )

            laser_loc.import_field_ft(
                Field_ft_new,
                r_axis=prop_rt.r_new,
                transform=True, make_copy=False
            )

            field_3d[im] = laser_loc.Field.T

        laser_out.grid.set_temporal_field(field_3d)
        laser_out.grid.axes[-1] = laser_loc.t
        laser_out.grid.hi[-1] = laser_loc.t.max()
        laser_out.grid.lo[-1] = laser_loc.t.min()

        return laser_out

class MRTFresnelPropagator(Propagator):
    """
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        return

    def update(self, laser_in):
        self.laser_in = laser_in
        return

    def propagate(self, distance, laser_out=None):
        laser_in = self.laser_in
        verbose = self.verbose

        containers_in, m_axis = import_from_lasy(laser_in)

        if laser_out is None:
            laser_out = laser_in

        self.props_rt = []
        for im in range(m_axis.size):
            m = m_axis[im]
            container_in = containers_in[im]
            self.props_rt.append(
                PropagatorResamplingFresnel(
                    dz=distance,
                    r_axis=container_in.r,
                    kz_axis=container_in.k_freq,
                    r_axis_new=laser_out.grid.axes[0],
                    mode=m, verbose=verbose
                )
            )

        field_3d = np.zeros_like(laser_out.grid.temporal_field)

        for im in range(m_axis.size):
            prop_rt = self.props_rt[im]

            m = m_axis[im]
            container_in = containers_in[im]
            Field_ft_new = prop_rt.step(
                container_in.Field_ft,
                distance, overwrite=False,
                show_progress=verbose
            )

            laser_loc = ScalarFieldEnvelope(
                container_in.k0,
                t_axis=container_in.t + distance/c
            )

            laser_loc.import_field_ft(
                Field_ft_new,
                r_axis=prop_rt.r_new,
                transform=True, make_copy=False
            )

            field_3d[im] = laser_loc.Field.T

        laser_out.grid.set_temporal_field(field_3d)
        laser_out.grid.axes[-1] = laser_loc.t
        laser_out.grid.hi[-1] = laser_loc.t.max()
        laser_out.grid.lo[-1] = laser_loc.t.min()

        return laser_out


class XYTPropagator(Propagator):
    """
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        return

    def update(self, laser_in):
        self.laser_in = laser_in
        return

    def propagate(self, distance, laser_out=None):
        laser_in = self.laser_in
        verbose = self.verbose

        container_in = import_from_lasy(laser_in)

        if laser_out is None:
            laser_out = laser_in

        prop_xyt = PropagatorFFT2(
            x_axis=container_in.x,
            y_axis=container_in.y,
            kz_axis=container_in.k_freq,
            verbose=verbose
        )

        Field_ft_new = prop_xyt.step(
            container_in.Field_ft,
            distance, overwrite=False,
            show_progress=verbose
        )


        laser_loc = ScalarFieldEnvelope(
            container_in.k0,
            t_axis=container_in.t + distance/c
        ).import_field_ft(
            Field_ft_new,
            r_axis=(prop_xyt.r, prop_xyt.x, prop_xyt.y),
            transform=True, make_copy=False)

        laser_out.grid.set_temporal_field(np.moveaxis(laser_loc.Field, 0, -1))
        laser_out.grid.axes[-1] = laser_loc.t
        laser_out.grid.hi[-1] = laser_loc.t.max()
        laser_out.grid.lo[-1] = laser_loc.t.min()

        return laser_out



class XYTFresnelPropagator(Propagator):
    """
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        return

    def update(self, laser_in):
        self.laser_in = laser_in
        return

    def propagate(self, distance, laser_out=None):
        laser_in = self.laser_in
        verbose = self.verbose

        container_in = import_from_lasy(laser_in)

        if laser_out is None:
            laser_out = laser_in

        x_axis_new = laser_out.grid.axes[0]
        y_axis_new = laser_out.grid.axes[1]

        prop_xyt = PropagatorFFT2Fresnel(
            dz=distance,
            x_axis=container_in.x,
            y_axis=container_in.y,
            x_axis_new=x_axis_new,
            y_axis_new=y_axis_new,
            kz_axis=container_in.k_freq,
            verbose=verbose
        )

        Field_ft_new = prop_xyt.step(
            container_in.Field_ft,
            distance, overwrite=False,
            show_progress=verbose
        )


        laser_loc = ScalarFieldEnvelope(
            container_in.k0,
            t_axis=container_in.t + distance/c
        ).import_field_ft(
            Field_ft_new,
            r_axis=(prop_xyt.r, prop_xyt.x, prop_xyt.y),
            transform=True, make_copy=False)

        laser_out.grid.set_temporal_field(np.moveaxis(laser_loc.Field, 0, -1))
        laser_out.grid.axes[-1] = laser_loc.t
        laser_out.grid.hi[-1] = laser_loc.t.max()
        laser_out.grid.lo[-1] = laser_loc.t.min()

        return laser_out
