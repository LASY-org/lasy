import numpy as np


class ABCD:
    r"""
    Class that represents an ABCD optical matrix.

    The ABCD matrix is defined in the following way for propagation in vacuum:

    .. math::

            O =
            \begin{pmatrix}
            A & B \\
            C & D
            \end{pmatrix}=
            \begin{pmatrix}
            1 & 0 \\
            0 & 1
            \end{pmatrix}.

    where :math:`E_{i} (x,y,\omega)` is the initial/propagated fields complex field envelope
    and :math:`S` is the propagator term

    :math:
        S = \bigg\{\frac{1}{2B}\Big[A(x_0^2+y_0^2)+D(x^2+y^2)-2(xx_0+yy_0)\Big]\bigg\},

    defined in terms of the elements of the ``'ABCD'`` optical ray matrix.

    Parameters
    ----------
    omega0 : float (in rad/s)
        The center frequency of the laser field.

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    abcd : 2d array
        The 2D ray matrix of the optical system through which the beam propagates.
        By default, this is initialised to be the unitary matrix:

        .. math::

    """

    def __init__(self, abcd=np.array([[1, 0], [0, 1]])):
        super().__init__()
        self.update(abcd=abcd)

    def update(self, abcd):
        r"""
        Initialize or update the ABCD matrix if needed.

        Parameters
        ----------
        abcd : 2d array
            The 2D ray matrix of the optical system through which the beam propagates.
            By default, this is initialised to be the unitary matrix:

            .. math::

                O =
                \begin{pmatrix}
                A & B \\
                C & D
                \end{pmatrix}=
                \begin{pmatrix}
                1 & 0 \\
                0 & 1
                \end{pmatrix}.
        """
        self.abcd = abcd  # optical ray matrix

    def add_vacuum(self, distance):
        vacuum = np.array([[1, distance], [0, 1]])
        self.abcd = np.matmul(vacuum, self.abcd)
        return

    def add_lens(self, focal_length):
        lens = np.array([[1, 0], [-1.0 / focal_length, 1]])
        self.abcd = np.matmul(lens, self.abcd)
        return

    def reset_matrix(self):
        self.abcd = np.array([[1, 0], [0, 1]])
        return
