from .propagator import Propagator


class SingleFFTPropagator(Propagator):
    """Single FFT propagators."""

    def __init__(self):
        super().__init__()
        print("empty init for SingleFFTPropagator")
