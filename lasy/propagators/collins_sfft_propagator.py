from .single_fft_propagator import SingleFFTPropagator


class CollinsSFFTPropagator(SingleFFTPropagator):
    def __init__(self):
        super().__init__()
        print("empty init for CollinsSFFTPropagator")
