import numpy as np


def find_center_of_mass(img):
    """Finds the center of mass of an image.

    Parameters:
    -----------
    img: 2Darray of floats
        The image on which to calculate the COM

    Returns:
    --------
    x0 , y0: floats
        The center of mass of the image along the horizontal
        and the vertical. The units are in pixels.
    """
    rows, cols = np.shape(img)
    x_data = np.linspace(0, cols - 1, cols)
    y_data = np.linspace(0, rows - 1, rows)

    # find the beam center using COM
    img_tot = np.sum(img)
    x0 = np.sum(np.dot(img, x_data)) / img_tot
    y0 = np.sum(np.dot(img.T, y_data)) / img_tot

    return x0, y0
