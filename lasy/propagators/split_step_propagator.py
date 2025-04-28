from .propagator import Propagator


class SplitStepPropagator(Propagator):
    def __init__(self, propagators):
        """
        Initialize the SplitStepPropagator.

        Parameters
        ----------
        propagators : list
            List of propagators to be used in the split-step method.
        """
        super().__init__()
        self.propagators = propagators

    def propagate(self, grid, distance, nsteps=1):
        """
        Propagate the input grid using a split-step method.

        Parameters
        ----------
        grid : Grid
            Input grid to be propagated.
        distance : float
            Distance over withh to propagate the field.
        nsteps : int, optional
            Number of steps to take during the propagation, by default 1
        """
        step_distance = distance / nsteps

        for _ in range(nsteps):
            for step_propagator in self.propagators:
                step_propagator.propagate(grid=grid, distance=step_distance)
