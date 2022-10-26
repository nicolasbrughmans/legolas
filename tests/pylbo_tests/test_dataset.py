import numpy as np
import pytest
from pylbo.exceptions import MatricesNotPresent

ds_v112_ev_guess = -0.14360602 + 0.00688731j
ds_v112_ev_idx = 158


def test_ds_iterable(ds_v112):
    gen = [i for i in ds_v112]
    assert len(gen) == 1
    assert gen.pop() == ds_v112


def test_ds_has_efs(ds_v112):
    assert isinstance(ds_v112.has_efs, bool)
    assert ds_v112.has_efs


def test_ds_ef_names(ds_v112):
    assert isinstance(ds_v112.ef_names, list)


def test_ds_ef_names_not_present(ds_v100):
    assert ds_v100.ef_names is None


def test_ds_ef_grid(ds_v112):
    grid = ds_v112.ef_grid
    assert grid is not None
    # check increasing and monotone
    assert np.all(grid[1:] > grid[:-1])


def test_ds_ef_grid_none(ds_v100):
    grid = ds_v100.ef_grid
    assert grid is None


def test_ds_sound_speed_invalid(ds_v112):
    with pytest.raises(ValueError):
        ds_v112.get_sound_speed(which_values="max")


def test_ds_sound_speed(ds_v112):
    cs = ds_v112.get_sound_speed()
    idxs = [0, 10, 21, -2]
    values = [0.40806667, 0.39936673, 0.37869727, 0.29759189]
    assert np.all(np.isclose(cs[idxs], values))


def test_ds_sound_speed_avg(ds_v112):
    cs_avg = ds_v112.get_sound_speed(which_values="average")
    assert np.isclose(cs_avg, 0.35804914)


def test_ds_sound_speed_min(ds_v112):
    cs_min = ds_v112.get_sound_speed(which_values="minimum")
    assert np.isclose(cs_min, 0.29617802)


def test_ds_sound_speed_max(ds_v112):
    cs_max = ds_v112.get_sound_speed(which_values="maximum")
    assert np.isclose(cs_max, 0.40806667)


def test_ds_alfven_speed(ds_v112):
    ca = ds_v112.get_alfven_speed()
    idxs = [2, 8, 15, -1]
    values = [0.94652052, 0.93669782, 0.91177547, 0.61700952]
    assert np.all(np.isclose(ca[idxs], values))


def test_ds_tube_speed(ds_v112):
    ct = ds_v112.get_tube_speed()
    idxs = [3, 17, -3, -7]
    values = [0.37351949, 0.35630853, 0.27120232, 0.27910677]
    assert np.all(np.isclose(ct[idxs], values))


def test_ds_tube_speed_cartesian(ds_v112_eta):
    ct = ds_v112_eta.get_tube_speed()
    assert ct is None


def test_ds_reynolds_no_eta(ds_v112):
    reynolds = ds_v112.get_reynolds_nb()
    assert reynolds is None


def test_ds_reynolds(ds_v112_eta):
    reynolds = ds_v112_eta.get_reynolds_nb()
    assert np.all(np.isclose(reynolds, 3535.53390593))


def test_ds_magnetic_reynolds_no_eta(ds_v112):
    magnetic_reynolds = ds_v112.get_magnetic_reynolds_nb()
    assert magnetic_reynolds is None


def test_ds_magnetic_reynolds(ds_v112_eta):
    magnetic_reynolds = ds_v112_eta.get_magnetic_reynolds_nb()
    assert np.all(np.isclose(magnetic_reynolds, 1e4))


def test_ds_k0_squared(ds_v112):
    k02 = ds_v112.get_k0_squared()
    assert np.isclose(k02, 1 + (-1.2) ** 2)


def test_ds_no_matrices(ds_v112):
    with pytest.raises(MatricesNotPresent):
        ds_v112.get_matrix_B()
    with pytest.raises(MatricesNotPresent):
        ds_v112.get_matrix_A()


def test_ds_matrix_B(ds_v100):
    rows, cols, vals = ds_v100.get_matrix_B()
    assert len(rows) == len(cols) == len(vals)
    assert np.all([isinstance(i, np.integer) for i in rows])
    assert np.all([isinstance(i, np.integer) for i in cols])
    assert np.all(np.isreal(vals))


def test_ds_matrix_A(ds_v100):
    rows, cols, vals = ds_v100.get_matrix_A()
    assert len(rows) == len(cols) == len(vals)
    assert np.all([isinstance(i, np.integer) for i in rows])
    assert np.all([isinstance(i, np.integer) for i in cols])
    assert np.all([isinstance(i, complex) for i in vals])


def test_ds_get_efs_invalid_guesses(ds_v112):
    with pytest.raises(ValueError):
        ds_v112.get_eigenfunctions(3 + 2j, 10)


def test_ds_get_efs_guess_single(ds_v112):
    guess = ds_v112_ev_guess
    efs = ds_v112.get_eigenfunctions(ev_guesses=guess)
    assert isinstance(efs, np.ndarray)
    (ef,) = efs
    assert np.isclose(guess, ef.get("eigenvalue", np.NaN))


def test_ds_get_efs_guess_list(ds_v112):
    guess = [ds_v112_ev_guess] * 2
    efs = ds_v112.get_eigenfunctions(ev_guesses=guess)
    assert isinstance(efs, np.ndarray)
    for i, ef in enumerate(efs):
        assert np.isclose(guess[i], ef.get("eigenvalue", np.NaN))


def test_ds_get_efs_idx(ds_v112):
    efs = ds_v112.get_eigenfunctions(ev_idxs=ds_v112_ev_idx)
    assert isinstance(efs, np.ndarray)
    (ef,) = efs
    assert np.isclose(ds_v112_ev_guess, ef.get("eigenvalue", np.NaN))


def test_ds_get_efs_idx_list(ds_v112):
    efs = ds_v112.get_eigenfunctions(ev_idxs=[ds_v112_ev_idx] * 2)
    assert isinstance(efs, np.ndarray)
    for ef in efs:
        assert np.isclose(ds_v112_ev_guess, ef.get("eigenvalue", np.NaN))


def test_ds_get_efs_idx_invalid(ds_v112):
    with pytest.raises(ValueError):
        ds_v112.get_eigenfunctions(ev_idxs=10.5)


def test_ds_get_efs_idx_list_invalid(ds_v112):
    with pytest.raises(ValueError):
        ds_v112.get_eigenfunctions(ev_idxs=[10, 1.2])


def test_ds_get_evs(ds_v112):
    idxs, evs = ds_v112.get_nearest_eigenvalues(ds_v112_ev_guess)
    assert isinstance(idxs, np.ndarray)
    assert idxs[0] == ds_v112_ev_idx
    assert isinstance(evs, np.ndarray)
    assert np.isclose(evs[0], ds_v112_ev_guess)


def test_ds_get_continua(ds_v112):
    continua = ds_v112.continua
    assert isinstance(continua, dict)
    for value in continua.values():
        assert isinstance(value, np.ndarray)
        assert len(value) == len(ds_v112.grid_gauss)


def test_ds_get_parameters(ds_v112):
    params = ds_v112.parameters
    assert isinstance(params, dict)
    for value in params.values():
        assert np.isscalar(value)
