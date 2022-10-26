from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

from pylbo.exceptions import (
    EigenfunctionsNotPresent,
    EigenvectorsNotPresent,
    MatricesNotPresent,
    ResidualsNotPresent,
)
from pylbo.utilities.datfile_utils import (
    read_derived_eigenfunction,
    read_ef_grid,
    read_eigenfunction,
    read_eigenvalues,
    read_eigenvectors,
    read_equilibrium_arrays,
    read_grid,
    read_grid_gauss,
    read_matrix_A,
    read_matrix_B,
    read_residuals,
)
from pylbo.utilities.datfiles.file_manager import LegolasFileManager
from pylbo.utilities.datfiles.header_legacy import LegolasLegacyHeader
from pylbo.utilities.logger import pylboLogger
from pylbo.utilities.toolbox import get_values, transform_to_numpy
from pylbo.visualisation.continua import calculate_continua


def ensure_dataset(data: any) -> None:
    """Ensures that the given data is a :class:`LegolasDataSet`."""
    if not isinstance(data, LegolasDataSet):
        raise TypeError(f"expected a LegolasDataSet, got {type(data)}")


def ensure_dataseries(data: any) -> None:
    """Ensures that the given data is a :class:`LegolasDataSeries`."""
    if not isinstance(data, LegolasDataSeries):
        raise TypeError(f"expected a LegolasDataSeries, got {type(data)}")


class LegolasDataContainer(ABC):
    """
    Baseclass for a Legolas data container.
    """

    @abstractmethod
    def continua(self):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def has_efs(self):
        pass

    @abstractmethod
    def ef_grid(self):
        pass

    @abstractmethod
    def ef_names(self):
        pass

    @abstractmethod
    def has_derived_efs(self):
        pass

    @abstractmethod
    def derived_ef_names(self):
        pass

    @abstractmethod
    def has_ef_subset(self):
        pass

    @abstractmethod
    def has_matrices(self):
        pass

    @abstractmethod
    def has_eigenvectors(self):
        pass

    @abstractmethod
    def has_residuals(self):
        pass

    @abstractmethod
    def get_sound_speed(self, which_values=None):
        pass

    @abstractmethod
    def get_alfven_speed(self, which_values=None):
        pass

    @abstractmethod
    def get_tube_speed(self, which_values=None):
        pass

    @abstractmethod
    def get_reynolds_nb(self, which_values=None):
        pass

    @abstractmethod
    def get_magnetic_reynolds_nb(self, which_values=None):
        pass

    @abstractmethod
    def get_k0_squared(self):
        pass


class LegolasDataSet(LegolasDataContainer):
    """
    Main container for a single Legolas dataset.

    Parameters
    ----------
    datfile : str, ~os.PathLike
        Path to the datfile.

    Attributes
    ----------
    header : dict
        The datfile header.
    grid : numpy.ndarray
        The base grid.
    grid_gauss : numpy.ndarray
        The Gaussian grid.
    equilibria : dict
        Dictionary containing the equilibrium arrays.
    eigenvalues : numpy.ndarray
        Array containing the complex eigenvalues.
    geometry : str
        The geometry.
    scale_factor : numpy.ndarray
        Array with the scale factor. One for Cartesian geometries, r for cylindrical.
    x_start : float
        Start of the grid.
    x_end : float
        End of the grid
    gridpoints : int
        The number of base gridpoints.
    gauss_gridpoints : int
        The number of Gaussian gridpoints.
    matrix_gridpoints : int
        The dimension of the matrix.
    ef_gridpoints : int
        The number of eigenfunction gridpoints.
    gamma : float
        The ratio of specific heats.
    eq_type : str
        The type of equilibrium selected.
    cgs : bool
        If `True`, all units are in cgs.
    units : dict
        Dictionary containing the unit normalisations.
    eq_names : numpy.ndarray
        Array containing the names of the equilibrium arrays.
    legolas_version : ~pylbo._version.VersionHandler
        The current Legolas version.
    """

    def __init__(self, datfile):
        self.datfile = Path(datfile)
        with open(self.datfile, "rb") as istream:
            filemanager = LegolasFileManager(istream)
            self.legolas_version = filemanager.legolas_version
            if self.legolas_version < "2.0":
                self.header = LegolasLegacyHeader(filemanager)
            self.grid = read_grid(istream, self.header)
            self.grid_gauss = read_grid_gauss(istream, self.header)
            self.equilibria = read_equilibrium_arrays(istream, self.header)
            self.eigenvalues = read_eigenvalues(istream, self.header)

        self.geometry = self.header["geometry"]
        if self.geometry == "Cartesian":
            self.scale_factor = np.ones_like(self.grid_gauss)
            self.d_scale_factor = np.zeros_like(self.grid_gauss)
            pylboLogger.debug("dataset: scale factor set to unity.")
        else:
            self.scale_factor = self.grid_gauss
            self.d_scale_factor = np.ones_like(self.grid_gauss)
            pylboLogger.debug("dataset: scale factor set to radial coordinate.")

        self.x_start = self.header["x_start"]
        self.x_end = self.header["x_end"]
        self.gridpoints = self.header["gridpoints"]
        self.gauss_gridpoints = self.header["gauss_gridpoints"]
        self.matrix_gridpoints = self.header["matrix_gridpoints"]
        self.ef_gridpoints = self.header["ef_gridpoints"]
        self.gamma = self.header["gamma"]
        self.eq_type = self.header["eq_type"]
        self._parameters = self.header["parameters"]
        self.units = self.header["units"]
        self.cgs = self.units["cgs"]
        self.eq_names = self.header["equilibrium_names"]

        self._continua = calculate_continua(self)

    def __iter__(self):
        yield self

    @property
    def k2_str(self) -> str:
        """Returns the :math:`k_2` string."""
        return "$k_y$" if self.geometry.lower() == "cartesian" else "$m$"

    @property
    def k3_str(self) -> str:
        """Returns the :math:`k_3` string."""
        return "$k_z$" if self.geometry.lower() == "cartesian" else "$k$"

    @property
    def u1_str(self) -> str:
        """Returns the :math:`u_1` string."""
        return "x" if self.geometry.lower() == "cartesian" else "r"

    @property
    def u2_str(self) -> str:
        """Returns the :math:`u_2` string."""
        return "y" if self.geometry.lower() == "cartesian" else r"$\theta$"

    @property
    def u3_str(self) -> str:
        """Returns the :math:`u_3` string."""
        return "z"

    @property
    def continua(self) -> dict:
        """Returns the continua in a dict with the continua names as keys."""
        return self._continua

    @property
    def parameters(self) -> dict:
        """Returns the parameters in a dict with the parameter names as keys"""
        return self._parameters

    @property
    def has_efs(self) -> bool:
        """Returns `True` if eigenfunctions are present."""
        return self.header["has_efs"]

    @property
    def ef_grid(self) -> np.ndarray:
        """Returns the eigenfunction grid, None if eigenfunctions are not present."""
        if not self.has_efs:
            return None
        if getattr(self, "_ef_grid", None) is None:
            with open(self.datfile, "rb") as istream:
                self._ef_grid = read_ef_grid(istream, self.header)
        return self._ef_grid

    @property
    def ef_names(self) -> np.ndarray:
        """Retrieves the eigenfunction names, None if eigenfunctions are not present."""
        return self.header.get("ef_names", None)

    @property
    def has_derived_efs(self) -> bool:
        """Returns `True` if derived eigenfunctions are present."""
        return self.header["has_derived_efs"]

    @property
    def derived_ef_names(self) -> np.ndarray:
        """Retrieves the derived eigenfunction names, None if not present."""
        return self.header.get("derived_ef_names", None)

    @property
    def has_ef_subset(self) -> bool:
        """Returns `True` if the dataset contains a subset of the eigenfunctions."""
        return self.header["ef_subset_used"]

    @property
    def has_matrices(self) -> bool:
        """Checks if matrices are present."""
        return self.header["has_matrices"]

    @property
    def has_eigenvectors(self) -> bool:
        """Checks if eigenvectors are present."""
        return self.header["has_eigenvectors"]

    @property
    def has_residuals(self) -> bool:
        """Checks if residuals are present."""
        return self.header["has_residuals"]

    def get_sound_speed(self, which_values=None) -> Union[float, np.ndarray]:
        """
        Calculates the sound speed based on the equilibrium arrays,
        given by :math:`c_s = \\sqrt{\\frac{\\gamma p_0}{\\rho_0}}`.

        Parameters
        ----------
        which_values : str
            Callback to :meth:`get_values`, either "average"/"minimum"/"maximum".

        Returns
        -------
        float or numpy.ndarray
            Array with the sound speed at every grid point, or a float corresponding
            to the value of `which_values` if provided.
        """
        pressure = self.equilibria["T0"] * self.equilibria["rho0"]
        cs = np.sqrt(self.gamma * pressure / self.equilibria["rho0"])
        return get_values(cs, which_values)

    def get_alfven_speed(self, which_values=None) -> Union[float, np.ndarray]:
        """
        Calculates the Alfvén speed based on the equilibrium arrays,
        given by :math:`c_A = \\sqrt{\\frac{B_0^2}{\\rho_0}}`.

        Parameters
        ----------
        which_values : str
            Callback to :meth:`get_values`, either "average"/"minimum"/"maximum".

        Returns
        -------
        float or numpy.ndarray
            Array with the Alfvén speed at every grid point,
            or a float corresponding to the value of `which_values` if provided.
        """
        cA = self.equilibria["B0"] / np.sqrt(self.equilibria["rho0"])
        return get_values(cA, which_values)

    def get_tube_speed(self, which_values=None) -> Union[float, np.ndarray]:
        """
        Calculates the tube speed for a cylinder, given by
        :math:`c_t = \\frac{c_s c_A}{\\sqrt{c_s^2 + c_A^2}}`.

        Parameters
        ----------
        which_values : str
            Callback to :meth:`get_values`, either "average"/"minimum"/"maximum".

        Returns
        -------
        float or numpy.ndarray
            Array with the tube speed at every grid point,
            or a float corresponding to the value of `which_values` if provided.
            Returns `None` if the geometry is not cylindrical.
        """
        if not self.geometry == "cylindrical":
            pylboLogger.warning(
                "geometry is not cylindrical, unable to calculate tube speed"
            )
            return None
        else:
            cA = self.get_alfven_speed()
            cs = self.get_sound_speed()
            ct = cs * cA / np.sqrt(cs**2 + cA**2)
            return get_values(ct, which_values)

    def get_reynolds_nb(self, which_values=None) -> Union[float, np.ndarray]:
        """
        Calculates the Reynolds number, defined as
        :math:`R_e = \\frac{ac_s}{\\eta}` where the slabsize is given by
        :math:`a = x_{end} - x_{start}`.

        Parameters
        ----------
        which_values : str
            Callback to :meth:`get_values`, either "average"/"minimum"/"maximum".

        Returns
        -------
        float or numpy.ndarray
            Array with the Reynolds number at every grid point,
            or a float corresponding to the value of `which_values` if provided.
            Returns `None` if the resistivity is zero somewhere on the domain.
        """
        cs = self.get_sound_speed()
        a = self.x_end - self.x_start
        eta = self.equilibria["eta"]
        if (eta == 0).any():
            pylboLogger.warning(
                "resistivity is zero somewhere on the domain, unable to "
                "calculate the Reynolds number"
            )
            return None
        else:
            Re = a * cs / eta
            return get_values(Re, which_values)

    def get_magnetic_reynolds_nb(self, which_values=None) -> Union[float, np.ndarray]:
        """
        Calculates the magnetic Reynolds number, defined as
        :math:`R_m = \\frac{ac_A}{\\eta}` where the slabsize is given by
        :math:`a = x_{end} - x_{start}`.

        Parameters
        ----------
        which_values : str
            Callback to :meth:`get_values`, either "average"/"minimum"/"maximum".

        Returns
        -------
        float or numpy.ndarray
            Array with the magnetic Reynolds number at every grid point,
            or a float corresponding to the value of `which_values` if provided.
            Returns `None` if the resistivity is zero somewhere on the domain.
        """
        cA = self.get_alfven_speed()
        a = self.x_end - self.x_start
        eta = self.equilibria["eta"]
        if (eta == 0).any():
            pylboLogger.warning(
                "resistivity is zero somewhere on the domain, unable to "
                "calculate the magnetic Reynolds number"
            )
            return None
        else:
            Rm = a * cA / eta
            return get_values(Rm, which_values)

    def get_k0_squared(self) -> float:
        """
        Calculates the squared wave number, defined as
        :math:`k_0^2 = k_2^2 + k_3^2`.
        """
        return self.parameters.get("k2") ** 2 + self.parameters.get("k3") ** 2

    def get_matrix_B(self) -> tuple(np.ndarray, np.ndarray, np.ndarray):
        """
        Retrieves the matrix B from the datfile.

        Returns
        -------
        Tuple(rows: numpy.ndarray, cols: numpy.ndarray, vals: numpy.ndarray)
            Tuple containing the rows, columns and values of the non-zero B-matrix
            elements. Rows and columns are integers, values are real.

        Raises
        ------
        MatricesNotPresent
            If the matrices were not saved to the datfile.
        """
        if not self.has_matrices:
            raise MatricesNotPresent(self.datfile)
        with open(self.datfile, "rb") as istream:
            rows, cols, vals = read_matrix_B(istream, self.header)
        return rows, cols, vals

    def get_matrix_A(self) -> tuple(np.ndarray, np.ndarray, np.ndarray):
        """
        Retrieves the matrix A from the datfile.

        Returns
        -------
        Tuple(rows: numpy.ndarray, cols: numpy.ndarray, vals: numpy.ndarray)
            Tuple containing the rows, columns and values of the non-zero A-matrix
            elements. Rows and columns are integers, values are complex.

        Raises
        ------
        MatricesNotPresent
            If the matrices were not saved to the datfile.
        """
        if not self.has_matrices:
            raise MatricesNotPresent(self.datfile)
        with open(self.datfile, "rb") as istream:
            rows, cols, vals = read_matrix_A(istream, self.header)
        return rows, cols, vals

    def get_eigenvectors(self) -> np.ndarray:
        """
        Retrieves the eigenvectors from the datfile.

        Returns
        -------
        numpy.ndarray
            Array containing the eigenvectors. One eigenvector
            in each column.

        Raises
        ------
        EigenvectorsNotPresent
            If the eigenvectors were not saved to the datfile.
        """
        if not self.has_eigenvectors:
            raise EigenvectorsNotPresent(self.datfile)
        with open(self.datfile, "rb") as istream:
            evs = read_eigenvectors(istream, self.header)
        return evs

    def get_residuals(self) -> np.ndarray:
        """
        Retrieves the residuals from the datfile.

        Returns
        -------
        numpy.ndarray
            Array containing the residuals.

        Raises
        ------
        ResidualsNotPresent
            If the residuals were not saved to the datfile.
        """
        if not self.has_residuals:
            raise ResidualsNotPresent(self.datfile)
        with open(self.datfile, "rb") as istream:
            res = read_residuals(istream, self.header)
        return res

    def get_eigenfunctions(self, ev_guesses=None, ev_idxs=None) -> np.ndarray:
        """
        Returns the eigenfunctions based on given eigenvalue guesses or their
        indices. An array will be returned where every item is a dictionary, containing
        both the eigenvalue and its eigenfunctions. Either eigenvalue guesses or
        indices can be supplied, but not both.

        Parameters
        ----------
        ev_guesses : (list of) int, float, complex
            Eigenvalue guesses.
        ev_idxs : (list of) int
            Indices corresponding to the eigenvalues that need to be retrieved.

        Returns
        -------
        numpy.ndarray
            Array containing the eigenfunctions and eigenvalues corresponding to the
            supplied indices. Every index in this array contains a dictionary with the
            eigenfunctions and corresponding eigenvalue.
            The keys of each dictionary are the eigenfunction names.
        """
        if not self.has_efs:
            raise EigenfunctionsNotPresent("eigenfunctions not written to datfile")
        if ev_guesses is not None and ev_idxs is not None:
            raise ValueError(
                "get_eigenfunctions: either provide guesses or indices but not both"
            )
        if ev_guesses is not None:
            idxs, _ = self.get_nearest_eigenvalues(ev_guesses)
        else:
            idxs = transform_to_numpy(ev_idxs)
            for idx in idxs:
                if not isinstance(idx, (int, np.int64)):
                    raise ValueError("get_eigenfunctions: ev_idxs should be integers")
        eigenfuncs = np.array([{}] * len(idxs), dtype=dict)
        with open(self.datfile, "rb") as istream:
            for dict_idx, ef_idx in enumerate(idxs):
                efs = read_eigenfunction(istream, self.header, ef_idx)
                if efs is not None:
                    efs.update({"eigenvalue": self.eigenvalues[ef_idx]})
                eigenfuncs[dict_idx] = efs
        return eigenfuncs

    def get_derived_eigenfunctions(self, ev_guesses=None, ev_idxs=None) -> np.ndarray:
        """
        Returns the derived eigenfunctions based on given eigenvalue guesses or their
        indices. An array will be returned where every item is a dictionary, containing
        both the eigenvalue and its quantities. Either eigenvalue guesses or
        indices can be supplied, but not both.

        Parameters
        ----------
        ev_guesses : (list of) int, float, complex
            Eigenvalue guesses.
        ev_idxs : (list of) int
            Indices corresponding to the eigenvalues that need to be retrieved.

        Returns
        -------
        numpy.ndarray
            Array containing the derived eigenfunctions and eigenvalues
            corresponding to the supplied indices. Every index in this array
            contains a dictionary with the derived eigenfunctions and
            corresponding eigenvalue. The keys of each dictionary are the
            corresponding eigenfunction names.
        """
        if not self.has_derived_efs:
            raise EigenfunctionsNotPresent(
                "derived eigenfunctions not written to datfile"
            )
        if ev_guesses is not None and ev_idxs is not None:
            raise ValueError("either provide guesses or indices but not both")
        if ev_guesses is not None:
            idxs, _ = self.get_nearest_eigenvalues(ev_guesses)
        else:
            idxs = transform_to_numpy(ev_idxs)
            for idx in idxs:
                if not isinstance(idx, (int, np.int64)):
                    raise ValueError("ev_idxs should be integers")
        derived_efs = np.array([{}] * len(idxs), dtype=dict)
        with open(self.datfile, "rb") as istream:
            for dict_idx, ef_idx in enumerate(idxs):
                defs = read_derived_eigenfunction(istream, self.header, ef_idx)
                if defs is not None:
                    defs.update({"eigenvalue": self.eigenvalues[ef_idx]})
                derived_efs[dict_idx] = defs
        return derived_efs

    def get_nearest_eigenvalues(self, ev_guesses) -> tuple(np.ndarray, np.ndarray):
        """
        Calculates the eigenvalues nearest to a given guess. This calculates
        the nearest eigenvalue based on the distance between two points.

        Parameters
        ----------
        ev_guesses : float, complex, list of float, list of complex
            The guesses for the eigenvalues. These can be a single float/complex value,
            or a list/Numpy array of floats/complex values.

        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray)
            The indices of the nearest eigenvalues in the :attr:`eigenvalues` array.
            The nearest eigenvalues to the provided guesses, corresponding with the
            indices `idxs`.
        """
        ev_guesses = transform_to_numpy(ev_guesses)
        idxs = np.empty(shape=len(ev_guesses), dtype=int)
        eigenvals = np.empty(shape=len(ev_guesses), dtype=complex)
        for i, ev_guess in enumerate(ev_guesses):
            # distance from guess to all eigenvalues
            distances = (self.eigenvalues.real - ev_guess.real) ** 2 + (
                self.eigenvalues.imag - ev_guess.imag
            ) ** 2
            # closest distance (squared)
            idx = np.nanargmin(distances)
            idxs[i] = idx
            eigenvals[i] = self.eigenvalues[idx]
        return idxs, eigenvals


class LegolasDataSeries(LegolasDataContainer):
    def __init__(self, datfiles):
        self.datasets = [LegolasDataSet(datfile) for datfile in datfiles]
        self.geometry = set([ds.geometry for ds in self.datasets])
        if len(self.geometry) == 1:
            self.geometry = self.geometry.pop()

    def __iter__(self):
        for ds in self.datasets:
            yield ds

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return LegolasDataSeries(
                [ds.datfile for ds in self.datasets[idx.start : idx.stop : idx.step]]
            )
        else:
            return self.datasets[idx]

    def __len__(self):
        return len(self.datasets)

    @property
    def continua(self) -> dict:
        """
        Returns the continua. Each key corresponds to a multiple Numpy arrays,
        one for each dataset.
        """
        keys = self[0].continua.keys()
        _continua = {key: [] for key in keys}
        for ds in self:
            for key in keys:
                _continua[key].append(ds.continua[key])
        return {key: np.array(values) for key, values in _continua.items()}

    @property
    def parameters(self) -> dict:
        """
        Returns the parameters. Each key corresponds to multiple Numpy arrays,
        one for each dataset.
        """
        keys = self[0].parameters.keys()
        _params = {key: [] for key in keys}
        for ds in self:
            for key in keys:
                _params[key].append(ds.parameters[key])
        return {key: np.array(values) for key, values in _params.items()}

    @property
    def has_efs(self) -> np.ndarray:
        """Returns `True` if eigenfunctions are present."""
        return np.array([ds.has_efs for ds in self.datasets], dtype=bool)

    @property
    def ef_names(self) -> np.ndarray:
        """Returns the eigenfunction names."""
        return np.array([ds.ef_names for ds in self.datasets], dtype=object)

    @property
    def ef_grid(self) -> np.ndarray:
        """Returns the eigenfunction grid."""
        return np.array([ds.ef_grid for ds in self.datasets], dtype=object)

    @property
    def has_derived_efs(self) -> np.ndarray:
        """Returns `True` at index `i` if eigenfunctions are present in dataset `i`."""
        return np.array([ds.has_derived_efs for ds in self.datasets])

    @property
    def derived_ef_names(self) -> np.ndarray:
        """Returns the derived eigenfunction names."""
        return np.array([ds.derived_ef_names for ds in self.datasets], dtype=object)

    @property
    def has_ef_subset(self) -> np.ndarray:
        """Returns `True` at index `i` if the `i`-th dataset contains a subset."""
        return np.array([ds.has_ef_subset for ds in self.datasets], dtype=object)

    @property
    def has_matrices(self) -> np.ndarray:
        """Returns `True` at index `i` if the `i`-th dataset contains matrices."""
        return np.array([ds.has_matrices for ds in self.datasets], dtype=object)

    @property
    def has_eigenvectors(self) -> np.ndarray:
        """Returns `True` at index `i` if the `i`-th dataset contains eigenvectorst."""
        return np.array([ds.has_eigenvectors for ds in self.datasets], dtype=object)

    @property
    def has_residuals(self) -> np.ndarray:
        """Returns `True` at index `i` if the `i`-th dataset contains residuals."""
        return np.array([ds.has_residuals for ds in self.datasets], dtype=object)

    def get_sound_speed(self, which_values=None) -> np.ndarray:
        """
        Calculates the sound speed for the various datasets.

        Parameters
        ----------
        which_values : str
            Callback to :meth:`get_values`, either "average"/"minimum"/"maximum".

        Returns
        -------
        numpy.ndarray
            A Numpy array of same length as the number of datasets, containing the
            sound speeds. Elements are either arrays themselves or floats, depending
            on the value of `which_values`.
        """
        return np.array([ds.get_sound_speed(which_values) for ds in self.datasets])

    def get_alfven_speed(self, which_values=None) -> np.ndarray:
        """
        Calculates the Alfvén speed for the various datasets.

        Parameters
        ----------
        which_values : str
            Callback to :meth:`get_values`, either "average"/"minimum"/"maximum".

        Returns
        -------
        numpy.ndarray
            A Numpy array of same length as the number of datasets, containing the
            Alfvén speeds. Elements are either arrays themselves or floats, depending
            on the value of `which_values`.
        """
        return np.array([ds.get_alfven_speed(which_values) for ds in self.datasets])

    def get_tube_speed(self, which_values=None) -> np.ndarray:
        """
        Calculates the tube speed for the various datasets.

        Parameters
        ----------
        which_values : str
            Callback to :meth:`get_values`, either "average"/"minimum"/"maximum".

        Returns
        -------
        numpy.ndarray
            A Numpy array of same length as the number of datasets, containing the
            tube speeds. Elements are either arrays themselves or floats, depending
            on the value of `which_values`. Elements are None if the geometry is
            not cylindrical.
        """
        return np.array([ds.get_tube_speed(which_values) for ds in self.datasets])

    def get_reynolds_nb(self, which_values=None) -> np.ndarray:
        """
        Calculates the Reynolds number for the various datasets.

        Parameters
        ----------
        which_values : str
            Callback to :meth:`get_values`, either "average"/"minimum"/"maximum".

        Returns
        -------
        numpy.ndarray
            A Numpy array of same length as the number of datasets, containing the
            Reynolds number. Elements are either arrays themselves or floats, depending
            on the value of `which_values`.
            Elements are None if the resistivity is zero.
        """
        return np.array([ds.get_reynolds_nb(which_values) for ds in self.datasets])

    def get_magnetic_reynolds_nb(self, which_values=None) -> np.ndarray:
        """
        Calculates the magnetic Reynolds number for the various datasets.

        Parameters
        ----------
        which_values : str
            Callback to :meth:`get_values`, either "average"/"minimum"/"maximum".

        Returns
        -------
        numpy.ndarray
            A Numpy array of same length as the number of datasets, containing the
            magnetic Reynolds number. Elements are either arrays themselves
            or floats, depending on the value of `which_values`.
            Elements are None if the resistivity is zero.
        """
        return np.array(
            [ds.get_magnetic_reynolds_nb(which_values) for ds in self.datasets]
        )

    def get_k0_squared(self) -> np.ndarray:
        """
        Calculates the squared wave number for the various datasets.

        Returns
        -------
        numpy.ndarray
            A Numpy array of same length as the number of datasets, containing the
            squared wavenumber for each.
        """
        return np.array([ds.get_k0_squared() for ds in self.datasets], dtype=float)
