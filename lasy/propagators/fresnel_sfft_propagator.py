from .single_fft_propagator import SingleFFTPropagator


class FresnelSFFTPropagator(SingleFFTPropagator):
    """Fresnel's propagator."""

    def __init__(self):
        super().__init__()
        print("empty init for FresnelSFFTPropagator")
