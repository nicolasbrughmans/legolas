from .regression import RegressionTest
import pytest


class RotatingCylinder(RegressionTest):
    equilibrium = "rotating_plasma_cylinder"
    geometry = "cylindrical"

    parameters = {
        "k2": 1.0,
        "k3": 0.0,
        "p1": 8.0,
        "p2": 0.0,
        "p3": 0.0,
        "p4": 1.0,
        "p5": 0.0,
        "p6": 0.0,
        "cte_p0": 0.1,
        "cte_rho0": 1,
    }
    physics_settings = {"flow": True}


class TestRotatingCylinderQR(RotatingCylinder):
    name = "quasimodes k2=1 k3=0 QR"
    filename = "rotating_cylinder_QR_k2_1_k3_0"

    spectrum_limits = [
        {"xlim": (-2400, 2400), "ylim": (-2, 2)},
        {"xlim": (-100, 120), "ylim": (-2, 2)},
        {"xlim": (-5, 20), "ylim": (-2, 2)},
        {"xlim": (4.5, 9.5), "ylim": (-2, 2)},
    ]

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)


class TestRotatingCylinderSI(RotatingCylinder):
    name = "quasimodes k2=1 k3=0 shift-invert"
    filename = "rotating_cylinder_SI_k2_1_k3_0"
    solver_settings = {
        "solver": "arnoldi",
        "arpack_mode": "shift-invert",
        "number_of_eigenvalues": 8,
        "which_eigenvalues": "LM",
        "sigma": 6.5 + 2j,
    }

    spectrum_limits = [
        {"xlim": (4.5, 8), "ylim": (-0.2, 2.2)},
    ]

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)
