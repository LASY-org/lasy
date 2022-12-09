
class TransverseProfile(object):
    """
    Base class for all transverse profiles.

    Any new transverse profile should inherit from this class, and define its own
    `evaluate` method, using the same signature as the method below.
    """
    def __init__( self ):
        """
        Initialize the transverse profile.
        """

    def evaluate( self, dim, envelope, *axes ):
        """
        Fills the envelope field of the laser
        Usage: evaluate(dim, envelope, x, y) (3D Cartesian) or
               evaluate(dim, envelope, r) (2D cylindrical)

        Parameters
        -----------
        dim: string
            'rt' or 'xyt'

        envelope: ndarrays (V/m)
            Contains the values of the transverse envelope field, to be filled
            (1D for cylindrical coordinate, 2D for Cartesian coordinates)

        axes: Coordinates at which the envelope should be evaluated.
            Can be 1 elements in cylindrical geometry (r) or
            3 elements in Cartesian geometry (x,y).
        """
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        pass
