from .single_fft_propagator import SingleFFTPropagator


class CollinsSFFTPropagator(SingleFFTPropagator):
    """Collin's propagator."""

    def __init__(self):
        super().__init__()
        print("empty init for CollinsSFFTPropagator")
