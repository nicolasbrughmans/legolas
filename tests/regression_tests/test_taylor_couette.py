import pytest

from .regression import RegressionTest

spectrum_limits = [
    {"xlim": (-1100, 1100), "ylim": (-155, 5)},
    {"xlim": (-175, 175), "ylim": (-35, 4)},
    {"xlim": (-18, 18), "ylim": (-3, 0.3)},
    {"xlim": (-1.5, 1.5), "ylim": (-1.25, 0.2)},
]
eigenfunctions = [
    {"eigenvalue": 0.87784 - 0.04820j},
    {"eigenvalue": 0.49861 - 0.08124j},
    {"eigenvalue": 0.34390 - 0.13832j},
    {"eigenvalue": 0.26543 - 0.21655j},
    {"eigenvalue": 0.21295 - 0.31135j},
]


class TaylorCouette(RegressionTest):
    equilibrium = "taylor_couette"
    geometry = "cylindrical"
    x_start = 1
    x_end = 2

    parameters = {
        "k2": 0,
        "k3": 1,
        "cte_rho0": 1.0,
        "alpha": 1.0,
        "beta": 2.0,
    }
    physics_settings = {
        "flow": True,
        "coaxial": True,
        "viscosity": True,
        "viscosity_value": 1e-3,
    }
    eigenfunction_settings = {
        "write_eigenfunctions": True,
        "write_derived_eigenfunctions": True,
        "write_eigenfunction_subset": True,
        "eigenfunction_subset_center": 0.5 - 0.3j,
        "eigenfunction_subset_radius": 0.5,
    }


class TestTaylorCouetteQR(TaylorCouette):
    name = "Taylor Couette k2=0 k3=1 QR"
    filename = "taylor_couette_QR_k2_0_k3_1"

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)

    @pytest.mark.parametrize("eigenfunction", eigenfunctions)
    def test_eigenfunction(self, eigenfunction, ds_test, ds_base):
        super().run_eigenfunction_test(eigenfunction, ds_test, ds_base)

    @pytest.mark.parametrize("derived_eigenfunction", eigenfunctions)
    def test_derived_eigenfunction(self, derived_eigenfunction, ds_test, ds_base):
        super().run_derived_eigenfunction_test(derived_eigenfunction, ds_test, ds_base)


class TestTaylorCouetteQZ(TaylorCouette):
    name = "Taylor Couette k2=0 k3=1 QZ"
    filename = "taylor_couette_QZ_k2_0_k3_1"
    use_custom_baseline = "taylor_couette_QR_k2_0_k3_1"
    solver_settings = {"solver": "QZ-direct"}

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)


class TestTaylorCouetteQRCholesky(TaylorCouette):
    name = "Taylor Couette k2=0 k3=1 QR Cholesky"
    filename = "taylor_couette_QR_cholesky_k2_0_k3_1"
    use_custom_baseline = "taylor_couette_QR_k2_0_k3_1"
    solver_settings = {"solver": "QR-cholesky"}

    eigenfunctions = [
        {"eigenvalue": 0.87784 - 0.04820j, "RMS_TOLERANCE": 11.5},
        {"eigenvalue": 0.49861 - 0.08124j},
        {"eigenvalue": 0.34390 - 0.13832j, "RMS_TOLERANCE": 3},
        {"eigenvalue": 0.26543 - 0.21655j, "RMS_TOLERANCE": 6.5},
        {"eigenvalue": 0.21295 - 0.31135j, "RMS_TOLERANCE": 7},
    ]

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)

    @pytest.mark.parametrize("eigenfunction", eigenfunctions)
    def test_eigenfunction(self, eigenfunction, ds_test, ds_base):
        super().run_eigenfunction_test(eigenfunction, ds_test, ds_base)

    @pytest.mark.parametrize("derived_eigenfunction", eigenfunctions)
    def test_derived_eigenfunction(self, derived_eigenfunction, ds_test, ds_base):
        super().run_derived_eigenfunction_test(derived_eigenfunction, ds_test, ds_base)


class TestTaylorCouetteSI(TaylorCouette):
    name = "Taylor Couette k2=0 k3=1 shift-invert"
    filename = "taylor_couette_SI_k2_0_k3_1"
    solver_settings = {
        "solver": "arnoldi",
        "arpack_mode": "shift-invert",
        "number_of_eigenvalues": 4,
        "which_eigenvalues": "LM",
        "sigma": 0.2 - 0.2j,
    }
    spectrum_limits = [
        {"xlim": (-0.01, 0.35), "ylim": (-0.45, 0.01)},
    ]

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)
