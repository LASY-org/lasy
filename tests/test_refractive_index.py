from math import isclose

import scipy.constants as ct

from lasy.utils.refractive_index import Material, RefractiveIndexDatabase


def test_n_formulas():
    # Test that all formulas give correct values of n
    db = RefractiveIndexDatabase(auto_download=True)

    # Formula 1
    m = Material(name="fused silica", db=db)
    n = m.calc_n(0.9)
    assert isclose(n, 1.4518, rel_tol=1e-4)

    # Formula 2
    m = Material("specs", "OHARA-optical", "LAH78", db=db)
    n = m.calc_n(1.05)
    assert isclose(n, 1.8718, rel_tol=1e-4)

    # Formula 3
    m = Material("specs", "SUMITA-optical", "K-BOC20", db=db)
    n = m.calc_n(1.01)
    assert isclose(n, 1.9705, rel_tol=1e-4)

    # Formula 4
    m = Material("main", "Lu2O3", "Kaminskii", db=db)
    n = m.calc_n(0.87)
    assert isclose(n, 1.9156, rel_tol=1e-4)

    # Formula 5
    m = Material("main", "H2O", "Bashkatov", db=db)
    n = m.calc_n(0.81)
    assert isclose(n, 1.3272, rel_tol=1e-4)

    # Formula 6
    m = Material(name="air", db=db)
    n = m.calc_n(0.97)
    assert isclose(n, 1.00027426, rel_tol=1e-8)

    # Formula 7
    m = Material("main", "Si", "Edwards", db=db)
    n = m.calc_n(2.7)
    assert isclose(n, 3.4395, rel_tol=1e-4)

    # Formula 8
    m = Material("main", "AgBr", "Schröter", db=db)
    n = m.calc_n(0.6)
    assert isclose(n, 2.2531, rel_tol=1e-4)

    # Formula 9
    m = Material("organic", "urea", "Rosker-e", db=db)
    n = m.calc_n(0.96)
    assert isclose(n, 1.5915, rel_tol=1e-4)


def test_spectral_phase_expansion():
    db = RefractiveIndexDatabase()
    m = Material(name="fused silica", db=db)
    omega0 = 2 * ct.pi * ct.c / 800e-9
    dphi, dphi2, dphi3 = m.calc_spectral_phase_expansion(omega0)
    assert isclose(dphi, 4.8477e-09, rel_tol=1e-3)
    assert isclose(dphi2, 3.6162e-26, rel_tol=1e-4)
    assert isclose(dphi3, 2.747e-41, rel_tol=5e-3)


def test_k():
    db = RefractiveIndexDatabase()
    m = Material(name="BK7", db=db)
    k = m.calc_k(0.7)
    assert isclose(k, 8.9305e-9, rel_tol=1e-4)
