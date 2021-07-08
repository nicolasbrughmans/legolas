import pytest
import numpy as np

tc_pinch_setup = {
    "name": "tc_pinch",
    "config": {
        "geometry": "cylindrical",
        "x_start": 1.0,
        "x_end": 2.0,
        "gridpoints": 51,
        "parameters": {
            "k2": 0.0,
            "k3": 1.0,
            "cte_rho0": 10**(3.0),
            "alpha": 10**(-6.0),
            "beta": 1.5 * 10**(-6.0),
            "cte_B02": 10**(-3.0),
            "tau": 4.0 * 10**(-3.0),
        },
        "resistivity": True,
        "use_fixed_resistivity": True,
        "fixed_eta_value": 10**(-4.0),
        "viscosity": True,
        "viscosity_value": 10**(-6.0),
        "equilibrium_type": "tc_pinch",
        "logging_level": 0,
        "show_results": False,
        "write_eigenfunctions": False,
        "write_postprocessed": False,
        "write_matrices": False,
    },
    "image_limits": [
        {"xlims": (-50, 50), "ylims": (-11, 1)},
        {"xlims": (-2e-6, 2e-6), "ylims": (-1e-5, 3e-6)},
        {"xlims": (-5e-7, 5e-7), "ylims": (-2e-7, 3e-6)},
    ],
}
parametrisation = dict(
    argnames="setup",
    argvalues=[tc_pinch_setup],
    ids=[tc_pinch_setup["name"]],
)


@pytest.mark.parametrize(**parametrisation)
def test_eta_value(ds_test, setup):
    eta_value = 1e-4
    assert setup["config"]["fixed_eta_value"] == pytest.approx(eta_value)
    assert np.all(ds_test.equilibria.get("eta") == pytest.approx(eta_value))
