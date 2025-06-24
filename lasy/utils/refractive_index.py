"""
refractiveindex.info database parser/client.

Inspired somewhat by https://github.com/toftul/refractiveindex/tree/master
"""

import json
import os
import warnings
from pprint import pprint

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

known_materials_nk = {
    "fused silica": ("main", "SiO2", "Malitson"),
    "BK7": ("popular_glass", "BK7", "SCHOTT"),
    "air": ("other", "air", "Ciddor"),
    "hydrogen": ("main", "H2", "Peck"),
    "helium": ("main", "He", "Mansfield"),
    "argon": ("main", "Ar", "Peck-15C"),
    "SF11": ("popular_glass", "SF11", "SCHOTT"),
    "soda lime glass": ("glass", "soda-lime", "Rubin-clear"),
    "CaF2": ("main", "CaF2", "Malitson"),
    "MgF2": ("main", "MgF2", "Li-o"),
    "Sapphire": ("main", "Al2O3", "Malitson-o"),
}

known_materials_n2 = {
    "fused silica": ("main", "SiO2"),
    "BK7": ("glass", "BK7"),
    "air": ("other", "air"),
    "helium": ("main", "He"),
    "argon": ("main", "Ar"),
    "CaF2": ("main", "CaF2"),
    "MgF2": ("main", "MgF2"),
    "Sapphire": ("main", "Al2O3"),
}


class RefractiveIndexDatabase:
    """
    Refractive index database for various materials.

    Class that opens and stores the refractiveindex.info
    YAML database. The entire database will be downloaded if it does not exist and download is requested (typically on the first time it is run).

    Parameters
    ----------
    database_path : str or None
        Is None, defaults to user home directory. If passed,
        should be the directory containing the database
        structure.

    auto_download : bool, default is True
        If True, database will be downloaded. If False and
        no database found, an error will be thrown.
    """

    __database_version = "2025-02-23"

    def __init__(self, database_path=None, auto_download=False):
        # Load the json database shipped with lasy
        lasy_db_file = os.path.join(
            os.path.dirname(__file__), "refractive_index_db.json"
        )
        if os.path.isfile(lasy_db_file):
            with open(lasy_db_file, "r") as f:
                self.json_db = json.load(f)
        else:
            self.json_db = {"nk": {}, "n2": {}}

        if database_path is None:
            database_path = os.path.join(
                os.path.expanduser("~"), ".refractiveindex.info-database"
            )

        # If the full database does not exist, we can download it
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

        # If user asked to download, the database should exist now
        if os.path.exists(database_path):
            self.database_path = os.path.normpath(database_path)
            database_file_nk = os.path.join(database_path, "catalog-nk.yml")
            database_file_n2 = os.path.join(database_path, "catalog-n2.yml")

            # Clean the nk file of the 'DIVIDER' items
            clean_text_nk = _clean_yaml_file(database_file_nk)
            self.database_nk = yaml.load(clean_text_nk, Loader=yaml.BaseLoader)

            # Clean the n2 file of the 'DIVIDER' items
            clean_text_n2 = _clean_yaml_file(database_file_n2)
            self.database_n2 = yaml.load(clean_text_n2, Loader=yaml.BaseLoader)

    def get_nk_database_entry(
        self, shelf_name=None, book_name=None, page_name=None, name=None
    ):
        """
        Return requested nk database entry.

        This function tries to read the data from the json database
        shipped with lasy. If the data is not in there, tries to read it
        from the full database.

        Parameters
        ----------
        shelf_name : str or None
            refractiveindex.info shelf name.

        book_name : str or None
            refractiveindex.info book name.

        page_name : str or None
            refractiveindex.info page name.

        name : str or None
            A known name, defined in the dict `known_materials_nk` above.

        Returns
        -------
        mat_dict : dict
            Dict with all data available in the database entry, as
            read from the yaml file.
        """
        # If data is in the lasy database, return it
        if name in self.json_db["nk"].keys():
            return self.json_db["nk"][name]

        # Otherwise, try to read it from full database
        if name is not None:
            if name in known_materials_nk.keys():
                shelf_name, book_name, page_name = known_materials_nk[name]
            else:
                raise RuntimeError(f'Name "{name}" not known in local nk database!')

        if not hasattr(self, "database_nk"):
            raise RuntimeError("Full database not available!")

        db = self.database_nk
        shelf = next(iter(s for s in db if s["SHELF"] == shelf_name), None)
        if shelf is None:
            raise RuntimeError(f"Shelf {shelf_name} not in database!")

        book = next(iter(b for b in shelf["content"] if b["BOOK"] == book_name), None)
        if book is None:
            raise RuntimeError(f"Book {book_name} not on shelf {shelf_name}!")

        page = next(iter(p for p in book["content"] if p["PAGE"] == page_name), None)
        if page is None:
            raise RuntimeError(f"Page {page_name} not in book {book_name}!")

        # Load the file and return the contents
        filename_nk = os.path.join(self.database_path, "data", page["data"])
        with open(filename_nk) as f:
            mat_dict = yaml.load(f, Loader=yaml.BaseLoader)
        return mat_dict

    def get_n2_database_entries(self, shelf_name=None, book_name=None, name=None):
        """
        Return requested n2 database entries.

        This function tries to read the data from the json database
        shipped with lasy. If the data is not in there, tries to read it
        from the full database.

        Parameters
        ----------
        shelf_name : str or None
            refractiveindex.info shelf name.

        book_name : str or None
            refractiveindex.info book name.

        name : str or None
            A known name, defined in the dict `known_materials_n2` above.

        Returns
        -------
        data_dict : dict
            Dict containing the yaml database entries with page names as keys.
        """
        # If data is in the lasy database, return it
        if name in self.json_db["n2"].keys():
            return self.json_db["n2"][name]

        # Otherwise, try to read it from full database
        if name is not None:
            if name in known_materials_n2.keys():
                shelf_name, book_name = known_materials_n2[name]
            else:
                warnings.warn(f'Name "{name}" not known in local n2 database!')
                return {}

        if not hasattr(self, "database_n2"):
            raise RuntimeError("Full database not available!")

        db = self.database_n2
        shelf = next(iter(s for s in db if s["SHELF"] == shelf_name), None)
        if shelf is None:
            raise RuntimeError(f"Shelf {shelf_name} not in database!")

        book = next(iter(b for b in shelf["content"] if b["BOOK"] == book_name), None)
        if book is None:
            raise RuntimeError(f"Book {book_name} not on shelf {shelf_name}!")

        # Get all pages in here and load them all
        data_dict = {}
        for item in book["content"]:
            filename = os.path.join(self.database_path, "data", item["data"])
            with open(filename) as f:
                mat_dict = yaml.load(f, Loader=yaml.BaseLoader)
            mat_key = os.path.splitext(os.path.split(filename)[1])[0]
            data_dict[mat_key] = mat_dict
        return data_dict

    def get_shelves(self, db="nk"):
        """
        Get a list of shelves available in a database.

        Parameters
        ----------
        db : str, default is 'nk'
            Either 'nk' or 'n2'

        Returns
        -------
        shelves : list
            List of shelf nambes available
        """
        if db == "nk":
            db = self.database_nk
        elif db == "n2":
            db = self.database_n2

        return [s["SHELF"] for s in db]

    def get_books(self, shelf_name, db="nk"):
        """
        Get a list of books on a shelf in a given database.

        Parameters
        ----------
        shelf_name : str
            A shelf, must be available in the database.

        db : str, default is 'nk'
            Either 'nk' or 'n2'

        Returns
        -------
        books : list
            List of books on the given shelf.
        """
        if db == "nk":
            db = self.database_nk
        elif db == "n2":
            db = self.database_n2

        shelf = next(iter(s for s in db if s["SHELF"] == shelf_name), None)
        if shelf is None:
            raise RuntimeError(f"Shelf {shelf_name} not in {db} database!")

        return [s["BOOK"] for s in shelf["content"]]

    def get_pages(self, shelf_name, book_name, db="nk"):
        """
        Get a list of pages in a book on a shelf.

        Parameters
        ----------
        shelf_name : str
            A shelf, must be available in the database.

        book_name : str
            Name of book to get pages for.

        db : str, default is 'nk'
            Either 'nk' or 'n2'

        Returns
        -------
        pages : list
            List of pages in the requested book on a requested shelf.
        """
        if db == "nk":
            db = self.database_nk
        elif db == "n2":
            db = self.database_n2

        shelf = next(iter(s for s in db if s["SHELF"] == shelf_name), None)
        if shelf is None:
            raise RuntimeError(f"Shelf {shelf_name} not in {db} database!")

        book = next(iter(b for b in shelf["content"] if b["BOOK"] == book_name), None)
        if book is None:
            raise RuntimeError(f"Book {book_name} not on shelf {shelf_name}!")

        return [b["PAGE"] for b in book["content"]]


class Material:
    """
    Description of material and its optical properties.

    Class that contains material specific data: its refractive index and extinction coefficient.
    Input arguments can either be a known name defined in the dict above or a combination of shelf, book and page. The latter follow the definitions on refractiveindex.info website.

    Parameters
    ----------
    shelf : str or None
        refractiveindex.info shelf name.

    book : str or None
        refractiveindex.info book name.

    page : str or None
        refractiveindex.info page name.

    name : str or None
        A known name, defined in the dict above.

    db : RefractiveIndexDatabase instance or None
        An instance of RefractiveIndexDatabase can be
        given, which speeds up material initialisation.
    """

    def __init__(self, shelf=None, book=None, page=None, name=None, db=None):
        if db is None:
            db = RefractiveIndexDatabase()

        try:
            mat_nk = db.get_nk_database_entry(shelf, book, page, name)
            if mat_nk is not None:
                self._load_data_nk(mat_nk)
        except RuntimeError:
            print(f"Error loading nk data for {shelf}, {book}, {page}")
        try:
            mat_n2s = db.get_n2_database_entries(shelf, book, name)
            if mat_n2s is not None:
                self._load_data_n2(mat_n2s)
        except RuntimeError:
            print(f"Error loading n2 data for {shelf}, {book}")

    def _load_data_nk(self, mat_dict):
        self.reference = mat_dict.get("REFERENCES")
        self.conditions = mat_dict.get("CONDITIONS")
        self.properties = mat_dict.get("PROPERTIES")
        self.comments = mat_dict.get("COMMENTS")

        data_list = mat_dict.get("DATA")
        if data_list is None:
            raise RuntimeError("No usable data found")
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

    def _load_data_n2(self, mat_n2s):
        """Load all n2 data available for this material."""
        self.data_n2 = {}
        for mat_key, mat_dict in mat_n2s.items():
            mat = {}
            mat["reference"] = mat_dict.get("REFERENCES")
            mat["comments"] = mat_dict.get("COMMENTS")
            mat["conditions"] = mat_dict.get("CONDITIONS")
            data_list = mat_dict.get("DATA")
            for data in data_list:
                if data["type"] == "tabulated n2":
                    data_raw = np.fromstring(data.get("data", "0 0\n0 0"), sep=" ")
                    data_raw = np.reshape(data_raw, (len(data_raw) // 2, 2))
                    mat["data"] = data_raw
            self.data_n2[mat_key] = mat

    def calc_n(self, lambda_um):
        """
        Calculate refractive index for this material.

        Performs the calculation and checks for wavelength
        being in the required range.

        Parameters
        ----------
        lambda_um : float or iterable
            Wavelength(s) at which to evaluate the refractive
            index. Must be in microns.

        Returns
        -------
        n : float or np.array
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
        lambda_um : float or iterable
            Wavelength(s) at which to evaluate the extinction
            coefficient. Must be in microns.

        Returns
        -------
        k : float or np.array
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
        r"""
        Calculate spectral phase expansion terms.

        More precisely, the first three terms of the Taylor
        expansion of the spectral phase around :math:`\omega_0`
        are calculated:

        .. math::

            \phi_1 = \left.\frac{\mathrm{d}\phi}{\mathrm{d}\omega}\right\vert_{\omega_0}

            \phi_2 = \left.\frac{\mathrm{d}^2\phi}{\mathrm{d}\omega^2}\right\vert_{\omega_0}

            \phi_3 = \left.\frac{\mathrm{d}^3\phi}{\mathrm{d}\omega^3}\right\vert_{\omega_0}

        Definitions can be found at
        https://www.newport.com/n/the-effect-of-dispersion-on-ultrashort-pulses

        Parameters
        ----------
        omega0 : float (in rad/s)
            Central frequency at which to evaluate the
            spectral phase expansion terms.

        Returns
        -------
        dphi_dw : float
            First term (group delay), in units s/m

        d2phi_dw2 : float
            Second term (GVD), in units s^2/m

        d3phi_dw3 : float
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

    def get_n2(self, lambda_mu, page=None, return_mean=False):
        """
        Retrieve n2 from the database.

        If page is not passed, the code will look through all
        the available data and return data, which contain
        the passed wavelength.

        Parameters
        ----------
        lambda_mu : float
            The wavelength at which n2 should be evaluated. Must
            be in microns.

        page : str or None, default is None
            If given, will only return n2 data from the requested
            page, ignoring all other data for the material.

        return_mean : bool, default is False
            If False, all data where n2 can be evaluated will be
            returned. If True, a mean value of all available data
            is returned.

        Returns
        -------
        n2 : float or array
            n2 value (in m^2/W) at the given lambda_mu.
            If `return_mean` is True, a mean of all available data.
            If `return_mean` is False, all available data in an array.

            0 is returned if no data exists or was found.

        pages : list
            List of page names which contain n2 data for the requested
            wavelength.
        """
        if not self.data_n2:
            return 0, []

        if page is not None:
            if page in self.data_n2.keys():
                return self.data_n2[page]["data"]

        # If page was not passed, we loop through all data
        n2_data, pages = [], []
        for page_name, data_dict in self.data_n2.items():
            # If we have only one lambda, check it's correct
            lambdas, n2s = data_dict["data"][:, 0], data_dict["data"][:, 1]
            if len(lambdas) == 1:
                if lambdas[0] == lambda_mu:
                    n2_data.append(n2s[0])
                    pages.append(page_name)
            # Otherwise, get interpolated value if in range
            else:
                n2 = np.interp(lambda_mu, lambdas, n2s, left=0, right=0)
                if n2 != 0:
                    n2_data.append(n2)
                    pages.append(page_name)

        if not n2_data:
            return 0, []

        if return_mean:
            return np.mean(np.array(n2_data)), pages
        else:
            return np.array(n2_data), pages

    def print_n2_data(self):
        """Nicely print out all the known data for n2."""
        if self.data_n2:
            pprint(self.data_n2)

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


def _clean_yaml_file(filename):
    # Clean the nk and n2 file of the 'DIVIDER' items
    clean_text = []
    with open(filename) as f:
        for line in f:
            if "DIVIDER" not in line:
                clean_text.append(line)
    clean_text = "".join(clean_text)

    return clean_text


def _create_lasy_material_database():
    """
    Store properties of certain materials to a lasy database.

    This relies on the database being downloaded. The required
    data is then extracted based on the entries in
    `known_materials`.
    """
    lasy_db_file = os.path.join(os.path.dirname(__file__), "refractive_index_db.json")
    if os.path.isfile(lasy_db_file):
        os.remove(lasy_db_file)

    db = RefractiveIndexDatabase(auto_download=True)
    db_dict = {"nk": {}, "n2": {}}
    for material, entry in known_materials_nk.items():
        print(f"Creating nk record for {material}")
        mat_dict = db.get_nk_database_entry(name=material)
        db_dict["nk"][material] = mat_dict

    for material, entry in known_materials_n2.items():
        print(f"Creating n2 record for {material}")
        mat_n2s = db.get_n2_database_entries(name=material)
        db_dict["n2"][material] = mat_n2s

    with open(lasy_db_file, "a+") as f:
        json.dump(db_dict, f, indent=2)
