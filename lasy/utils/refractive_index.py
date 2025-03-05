"""
refractiveindex.info database parser/client.

Inspired somewhat by https://github.com/toftul/refractiveindex/tree/master
"""

import os
import warnings

import numpy as np
import scipy.constants as ct
import yaml
from scipy.interpolate import CubicSpline

try:
    import numdifftools as nd

    have_nd = True
except ImportError:
    warnings.warn(
        "numdifftools not available! "
        "Using fixed numerical expressions for spectral "
        "phase expansion calculations."
    )
    have_nd = False

known_materials = {
    "fused silica": ("glass", "fused_silica", "Malitson"),
    "BK7": ("popular_glass", "BK7", "SCHOTT"),
    "air": ("other", "air", "Ciddor"),
}


class RefractiveIndexDatabase:
    """
    Refractive index database for various materials.

    Class that opens and stores the refractiveindex.info
    YAML database. The entire database will be downloaded
    on the first time it is run.
    """

    __database_version = "2025-02-23"

    def __init__(self, database_path=None, auto_download=True):
        """
        Initialise the refractive index database.

        Data will be downloaded, if it does not exist
        and download is requested.

        Parameters
        ----------
        database_path: str or None
            Is None, defaults to user home directory. If passed,
            should be the directory containing the database
            structure.

        auto_download: bool, default is True
            If True, database will be downloaded. If False and
            no database found, an error will be thrown.
        """
        if database_path is None:
            database_path = os.path.join(
                os.path.expanduser("~"), ".refractiveindex.info-database"
            )

        if not os.path.exists(database_path) and auto_download:
            import shutil
            import tempfile
            import urllib.request
            import zipfile

            with tempfile.TemporaryDirectory() as tempdir:
                zip_filename = os.path.join(tempdir, "db.zip")

                print("Downloading refractiveindex.info database...", end="")
                url = (
                    "https://github.com/polyanskiy/"
                    "refractiveindex.info-database/archive/"
                    f"refs/tags/v{self.__database_version}.zip"
                )
                urllib.request.urlretrieve(url, zip_filename)

                print(" extracting zip file...", end="")
                with zipfile.ZipFile(zip_filename, "r") as zf:
                    zf.extractall(tempdir)
                tempdb = os.path.join(
                    tempdir,
                    "refractiveindex.info-database-" + self.__database_version,
                    "database",
                )
                shutil.move(tempdb, database_path)
                print(" Done!")

        self.database_path = os.path.normpath(database_path)
        self.database_file = os.path.join(database_path, "catalog-nk.yml")

        # Clean the file of the 'DIVIDER' items
        clean_text = []
        with open(self.database_file) as f:
            for line in f:
                if "DIVIDER" not in line:
                    clean_text.append(line)
        clean_text = "".join(clean_text)

        self.database = yaml.load(clean_text, Loader=yaml.BaseLoader)


class Material:
    """
    Description of material and its optical properties.

    Class that contains material specific data:
    its refractive index and extinction coefficient.
    """

    def __init__(self, shelf=None, book=None, page=None, name=None, db=None):
        """
        Initialise the Material container.

        Initialise the Material. Input arguments can either be a known
        name defined in the dict above or a combination of shelf, book
        and page. The latter follow the definitions on
        refractiveindex.info website.

        Parameters
        ----------
        shelf: str or None
            refractiveindex.info shelf name.

        book: str or None
            refractiveindex.info book name.

        page: str or None
            refractiveindex.info page name.

        name: str or None
            A known name, defined in the dict above.

        db: RefractiveIndexDatabase instance or None
            An instance of RefractiveIndexDatabase can be
            given, which speeds up material initialisation.
        """
        self.db = db
        if name is not None:
            if name in known_materials.keys():
                shelf, book, page = known_materials[name]
            else:
                raise RuntimeError(f'Name "{name}" not known!')

        self._get_filename(shelf, book, page)

        self._load_data()

    def _get_filename(self, shelf_name, book_name, page_name):
        """Iterate through the database to get filename."""
        if self.db is None:
            self.db = RefractiveIndexDatabase()
        db = self.db.database

        shelf = next(iter(s for s in db if s["SHELF"] == shelf_name), None)
        if shelf is None:
            raise RuntimeError(f"Shelf {shelf_name} not in database!")

        book = next(iter(b for b in shelf["content"] if b["BOOK"] == book_name), None)
        if book is None:
            raise RuntimeError(f"Book {book_name} not on shelf {shelf_name}!")

        page = next(iter(p for p in book["content"] if p["PAGE"] == page_name), None)
        if page is None:
            raise RuntimeError(f"Page {page_name} not in book {book_name}!")

        self.filename = os.path.join(self.db.database_path, "data", page["data"])

    def _load_data(self):
        with open(self.filename) as f:
            mat_dict = yaml.load(f, Loader=yaml.BaseLoader)

        self.reference = mat_dict.get("REFERENCES")
        self.conditions = mat_dict.get("CONDITIONS")
        self.properties = mat_dict.get("PROPERTIES")
        self.comments = mat_dict.get("COMMENTS")

        data_list = mat_dict.get("DATA")
        if data_list is None:
            raise f"No usable data found in {self.filename}"
        for data in data_list:
            type = data.get("type").replace(" ", "")

            # Parse different types of data we know about
            if "formula" in type:
                self.type_n = type
                self.wavelength_range_n = np.fromstring(
                    data.get("wavelength_range", "nan nan"), sep=" "
                )
                self.coefficients_n = np.fromstring(
                    data.get("coefficients", "0 0"), sep=" "
                )
                self.equation_n = globals().get("_" + self.type_n)
            else:
                self.data_raw = np.fromstring(data.get("data", "0 0\n0 0"), sep=" ")
                n_cols = 3 if "nk" in type else 2
                self.data_raw = np.reshape(
                    self.data_raw, (len(self.data_raw) // n_cols, n_cols)
                )
                interp_kw = {}  # dict(bounds_error=False, fill_value=0.)

                if "n" in type:
                    self.type_n = "interp"
                    self.wavelengths_n = self.data_raw[:, 0]
                    self.wavelength_range_n = [
                        min(self.wavelengths_n),
                        max(self.wavelengths_n),
                    ]
                    self.data_n = self.data_raw[:, 1]
                    self.interp_n = CubicSpline(
                        self.wavelengths_n, self.data_n, **interp_kw
                    )
                if "k" in type:
                    self.wavelengths_k = self.data_raw[:, 0]
                    self.wavelength_range_k = [
                        min(self.wavelengths_k),
                        max(self.wavelengths_k),
                    ]
                    self.data_k = (
                        self.data_raw[:, 2] if "nk" in type else self.data_raw[:, 1]
                    )
                    self.interp_k = CubicSpline(
                        self.wavelengths_k, self.data_k, **interp_kw
                    )

    def calc_n(self, lambda_um):
        """
        Calculate refractive index for this material.

        Performs the calculation and checks for wavelength
        being in the required range.

        Parameters
        ----------
        lambda_um: float or iterable
            Wavelength(s) at which to evaluate the refractive
            index. Must be in microns.

        Returns
        -------
        n: float or np.array
            Refractive index value, same shape as `lambda_mu`. 0 is
            returned for wavelengths outside the applicable range
        """
        # Make inputs into a proper array
        if isinstance(lambda_um, (list, set)):
            lambda_um = np.array(lambda_um)

        mask = (self.wavelength_range_n[0] < lambda_um) & (
            lambda_um < self.wavelength_range_n[1]
        )

        if "formula" in self.type_n:
            n = self.equation_n(lambda_um, *self.coefficients_n)
        else:
            n = self.interp_n(lambda_um)

        if isinstance(mask, (bool, np.bool_)):
            return n * int(mask)
        else:
            n[np.logical_not(mask)] = 0.0
            return n

    def calc_k(self, lambda_um):
        """
        Calculate extinction coefficient for this material.

        Performs the calculation and checks for wavelength
        being in the required range.

        Parameters
        ----------
        lambda_um: float or iterable
            Wavelength(s) at which to evaluate the extinction
            coefficient. Must be in microns.

        Returns
        -------
        k: float or np.array
            Extinction coefficient, same shape as `lambda_mu`. 0 is
            returned for wavelengths outside the applicable range
        """
        # Check we have some data for this!
        if not hasattr(self, "interp_k"):
            print("No extinction data for this material!")
            return None

        # Make inputs into a proper array
        if isinstance(lambda_um, (list, set)):
            lambda_um = np.array(lambda_um)

        mask = (self.wavelength_range_k[0] < lambda_um) & (
            lambda_um < self.wavelength_range_k[1]
        )

        k = self.interp_k(lambda_um)

        if isinstance(mask, (bool, np.bool_)):
            return k * int(mask)
        else:
            k[np.logical_not(mask)] = 0.0
            return k

    def calc_spectral_phase_expansion(self, omega0):
        """
        Calculate spectral phase expansion terms.

        More precisely, the first three terms of the Taylor
        expansion of the spectral phase around :math:`omega0`
        are calculated:

        .. math::

            \frac{\mathrm{d}\phi}{\mathrm{d}/omega},
            \frac{\mathrm{d}^2\phi}{\mathrm{d}/omega^2},
            \frac{\mathrm{d}^3\phi}{\mathrm{d}/omega^3}

        Definitions can be found at
        https://www.newport.com/n/the-effect-of-dispersion-on-ultrashort-pulses

        Parameters
        ----------
        omega0: float (in rad/s)
            Central frequency at which to evaluate the
            spectral phase expansion terms.

        Returns
        -------
        dphi_dw: float
            First term, in units s/m

        d2phi_dw2: float
            Second term (GVD), in units s^2/m

        d3phi_dw3: float
            Third term (TOD), in units s^3/m
        """
        lam = 2 * np.pi * ct.c / omega0  # Sellmeier and everything uses dn/dlambda!
        lam_mu = 1e6 * lam
        dphi = (self.calc_n(lam_mu) - lam * self._dn_dw(lam_mu, 1)) / ct.c
        ddphi = lam**3 / (2 * np.pi * ct.c**2) * self._dn_dw(lam_mu, 2)
        dddphi = (
            -1
            / (omega0**2 * ct.c)
            * (
                3 * lam_mu**2 * self._dn_dw(lam_mu, 2)
                + lam_mu**3 * self._dn_dw(lam_mu, 3)
            )
        )

        # Returns in s^n/m
        return dphi, ddphi * 1e12, dddphi

    def _dn_dw(self, lambda_mu, order=1):
        if have_nd:
            dn_dw = nd.Derivative(self.calc_n, n=order)
            return 1.0 * dn_dw(lambda_mu)

        else:
            h = lambda_mu * 1e-4
            l0 = lambda_mu
            f = self.calc_n
            if order == 1:
                return (f(l0 + h) - f(l0 - h)) / (2 * h)
            elif order == 2:
                return (f(l0 + h) - 2 * f(l0) + f(l0 - h)) / (h**2)
            elif order == 3:
                return (
                    f(l0 + 2 * h) - 2 * f(l0 + h) + 2 * f(l0 - h) - f(l0 - 2 * h)
                ) / (2 * h**3)


def _formula1(lam, c1, c2, c3, c4, c5, c6, c7):
    # eg specs/vitron/infrared/IG6.yml
    l2 = lam**2
    n2 = (
        1
        + c1
        + c2 * l2 / (l2 - c3**2)
        + c4 * l2 / (l2 - c5**2)
        + c6 * l2 / (l2 - c7**2)
    )
    return np.sqrt(n2)


def _formula2(lam, c1, c2, c3, c4, c5, c6, c7):
    # eg specs/ohara/optical/LAH78.yml
    l2 = lam**2
    n2 = 1 + c1 + c2 * l2 / (l2 - c3) + c4 * l2 / (l2 - c5) + c6 * l2 / (l2 - c7)
    return np.sqrt(n2)


def _formula3(lam, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11):
    # eg specs/sumita/optical/K-BOC20.yml
    n2 = c1 + c2 * lam**c3 + c4 * lam**c5 + c6 * lam**c7 + c8 * lam**c9 + c10 * lam**c11
    return np.sqrt(n2)


def _formula4(lam, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10=0, c11=0):
    # eg main/BaGa4Se7/nk/Kato-beta.yml
    l2 = lam**2
    n2 = (
        c1
        + c2 * lam**c3 / (l2 - c4**c5)
        + c6 * lam**c7 / (l2 - c8**c9)
        + c10 * lam**c11
    )
    return np.sqrt(n2)


def _formula5(lam, c1, c2, c3, c4, c5, c6, c7):
    # eg xylene/nk/Li.yml
    n = c1 + c2 * lam**c3 + c4 * lam**c5 + c6 * lam**c7
    return n


def _formula6(lam, c1, c2, c3, c4=0, c5=0):
    # eg main/He/nk/Mansfield.yml
    l2 = lam**-2
    n = 1 + c1 + c2 / (c3 - l2) + c4 / (c5 - l2)
    return n


def _formula7(lam, c1, c2, c3, c4, c5):
    # eg main/Si/nk/Edwards.yml
    l2 = lam**2
    n = c1 + c2 / (l2 - 0.028) + c3 / (l2 - 0.028) ** 2 + c4 * l2 + c5 * lam**4
    return n


def _formula8(lam, c1, c2, c3, c4):
    # eg main/AgBr/nk/Schroter.yml
    l2 = lam**2
    RHS = c1 + c2 * l2 / (l2 - c3) + c4 * l2
    n2 = (2 * RHS + 1) / (1 - RHS)
    return np.sqrt(n2)


def _formula9(lam, c1, c2, c3, c4, c5, c6):
    # eg organic/CH4N2O - urea/nk/Rosker-e.yml
    lc5 = lam - c5
    n2 = c1 + c2 / (lam**2 - c3) + c4 * lc5 / (lc5**2 + c6)
    return np.sqrt(n2)
