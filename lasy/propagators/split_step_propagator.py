from .propagator import Propagator


class SplitStepPropagator(Propagator):
    """Class that represents a split step propagator.

    The propagator takes a list of propagator or nonlinear steppers as inputs 
    and propagates an input grid by iterating through each of these sub-steps.

    Parameters
        ----------
        propagators : list
            List of propagators to be used as the sub-steps of the propagation.
    """
    def __init__(self, propagators):
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
            for prop in self.propagators:
                prop.propagate(grid=grid, distance=step_distance)
