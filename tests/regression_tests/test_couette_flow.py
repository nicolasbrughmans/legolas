from .regression import RegressionTest
import pytest


class TestCouetteFlowQR(RegressionTest):
    name = "Couette flow k2=0 k3=1"
    filename = "couette_QR_k2_0_k3_1"
    equilibrium = "couette_flow"
    geometry = "Cartesian"

    parameters = {
        "k2": 0,
        "k3": 1,
        "cte_rho0": 1.0,
        "cte_T0": 1.0,
        "cte_v02": 0.0,
        "cte_v03": 1.0,
    }
    physics_settings = {"flow": True, "viscosity": True, "viscosity_value": 1e-3}
    eigenfunction_settings = {
        "write_eigenfunctions": True,
        "write_eigenfunction_subset": True,
        "eigenfunction_subset_center": 0.5 - 0.5j,
        "eigenfunction_subset_radius": 0.49,
    }

    spectrum_limits = [
        {"xlim": (-430, 430), "ylim": (-155, 5)},
        {"xlim": (-15, 15), "ylim": (-80, 10)},
        {"xlim": (-0.2, 1.2), "ylim": (-40, 10)},
        {"xlim": (-0.05, 1.1), "ylim": (-1.1, 0.2)},
    ]
    eigenfunctions = [
        {"eigenvalue": 0.20249 - 0.11790j},
        {"eigenvalue": 0.79751 - 0.11790j},
        {"eigenvalue": 0.35402 - 0.20539j},
        {"eigenvalue": 0.64598 - 0.20539j},
        {"eigenvalue": 0.48448 - 0.28639j},
        {"eigenvalue": 0.51552 - 0.28639j},
        {"eigenvalue": 0.50000 - 0.44519j},
    ]

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)

    @pytest.mark.parametrize("eigenfunction", eigenfunctions)
    def test_eigenfunction(self, eigenfunction, ds_test, ds_base):
        super().run_eigenfunction_test(eigenfunction, ds_test, ds_base)
