import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interpolate_complex_field_XY(spectral_field, X, Y, OM, X_new, Y_new):
    """
    Fast interpolation of complex 3D spectral field from varying regular XY grids to new regular XY grids.

    The input field is assumed to be defined on a regular grid in the XY plane for each frequency OM.
    The output field is interpolated to a new regular grid in the XY plane for each frequency OM.
    The input and output grids are assumed to be consistent with the frequency axis OM.


    Parameters
    ----------
    spectral_field : 3darray of complex numbers
        The spectral field to be interpolated.
        It is assumed to be defined on a regular grid in the XY plane for each frequency OM.

    X, Y : 3darray of real numbers
        The regular spatial grids per frequency for the input field.

    OM : 3darray of real numbers
        The frequemcy axis of the spectral field

    X_new, Y_new : 3darray of real numbers
        The regular output spatial grids per frequency

    Returns
    -------
    field_interp : 3darray of complex numbers
        The interpolated spectral field on the new regular grid.
    """
    Nx_new, Ny_new, Nom = X_new.shape
    field_interp = np.zeros((Nx_new, Ny_new, Nom), dtype=complex)

    for k in range(Nom):
        # Extract 1D vectors from meshgrid
        xk = X[:, 0, k]
        yk = Y[0, :, k]

        if not (
            np.allclose(X[:, :, k], np.meshgrid(xk, yk, indexing="ij")[0])
            and np.allclose(Y[:, :, k], np.meshgrid(xk, yk, indexing="ij")[1])
        ):
            raise ValueError(f"X and Y slice {k} are not regular grids.")

        # Create fast interpolators
        interp = RegularGridInterpolator(
            (xk, yk),
            spectral_field[:, :, k],
            method="nearest",
            bounds_error=False,
            fill_value=0,
        )

        # Interpolate on target grid
        target_points = np.stack(
            [X_new[:, :, k].ravel(), Y_new[:, :, k].ravel()], axis=-1
        )

        field_interp[:, :, k] = interp(target_points).reshape(Nx_new, Ny_new)

    return field_interp
