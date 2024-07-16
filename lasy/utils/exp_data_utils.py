from lasy.backend import xp


def find_center_of_mass(img):
    """
    Find the center of mass of an image.

    Parameters
    ----------
    img: 2Darray of floats
        The image on which to calculate the COM

    Returns
    -------
    x0 , y0: floats
        The center of mass of the image along the horizontal
        and the vertical. The units are in pixels.

    """
    rows, cols = xp.shape(img)
    x = xp.linspace(0, cols - 1, cols)
    y = xp.linspace(0, rows - 1, rows)

    # find the beam center using COM
    img_tot = xp.sum(img)
    x0 = xp.sum(xp.dot(img, x)) / img_tot
    y0 = xp.sum(xp.dot(img.T, y)) / img_tot

    return x0, y0


def find_d4sigma(img):
    """
    Find the D4Sigma measurement of the spot size in x and y in pixels.

    https://en.wikipedia.org/wiki/Beam_diameter#D4%CF%83_or_second-moment_width

    Parameters
    ----------
    img : A numpy array containing the spatial intensity profile of a laser pulse.

    Returns
    -------
    D4sigX : The D4sigma along the first (x) axis
    D4sigY : The D4sigma along the second (y) axis
    """
    rows, cols = xp.shape(img)
    x = xp.linspace(0, cols - 1, cols)
    y = xp.linspace(0, rows - 1, rows)

    x0, y0 = find_center_of_mass(img)

    img_tot = xp.sum(img)
    D4sigX = 4 * xp.sqrt(xp.sum(xp.dot(img, (x - x0) ** 2)) / img_tot)
    D4sigY = 4 * xp.sqrt(xp.sum(xp.dot(img.T, (y - y0) ** 2)) / img_tot)

    return D4sigX, D4sigY
