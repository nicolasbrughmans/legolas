import pytest
import numpy as np

hall_harris_sheet = {
    "name": "hall_harris_sheet",
    "config": {
        "geometry": "Cartesian",
        "x_start": -15,
        "x_end": 15,
        "gridpoints": 51,
        "parameters": {
            "k2": 0.155,
            "k3": 0.01,
            "alpha": 1,
            "cte_rho0": 1.0,
            "cte_B02": 1.0,
            "cte_B03": 5.0,
            "eq_bool": False,
        },
        "equilibrium_type": "harris_sheet",
        "resistivity": True,
        "use_fixed_resistivity": True,
        "fixed_eta_value": 10 ** (-4.0),
        "hall_mhd": True,
        "electron_fraction": 0.5,
        "cgs_units": True,
        "unit_density": 1.7e-14,
        "unit_magneticfield": 10,
        "unit_length": 7.534209349981049e-9,
        "incompressible": True,
        "logging_level": 0,
        "show_results": False,
        "write_eigenfunctions": False,
        "write_matrices": False,
    },
    "image_limits": [
        {"xlims": (-375, 375), "ylims": (-11, 0.5)},
        {"xlims": (-30, 30), "ylims": (-3, 0.3)},
        {"xlims": (-0.6, 0.6), "ylims": (-1.2, 0.1)},
        {"xlims": (-0.5, 0.5), "ylims": (-0.2, 0.02)},
    ],
}
parametrisation = dict(
    argnames="setup",
    argvalues=[hall_harris_sheet],
    ids=[hall_harris_sheet["name"]],
)


@pytest.mark.parametrize(**parametrisation)
def test_eta_value(ds_test, setup):
    eta_value = 1e-4
    assert setup["config"]["fixed_eta_value"] == pytest.approx(eta_value)
    assert np.all(ds_test.equilibria.get("eta") == pytest.approx(eta_value))
