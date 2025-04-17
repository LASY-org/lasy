from .propagator import Propagator


class SingleFFTPropagator(Propagator):
    def __init__(self):
        super().__init__()
        print("empty init for SingleFFTPropagator")
