from __future__ import annotations

import difflib
from typing import Union

import numpy as np
from pylbo.data_containers import LegolasDataSeries
from pylbo.exceptions import BackgroundNotPresent
from pylbo.utilities.logger import pylboLogger
from pylbo.visualisation.utils import ef_name_to_latex, validate_ef_name


class ModeVisualisationData:
    """
    Class that contains the data used for eigenmode visualisations.

    Parameters
    ----------
    ds : ~pylbo.data_containers.LegolasDataSeries
        The dataseries containing the eigenfunctions and modes to visualise,
        having the same equilibria.
    omega : list[list[complex]]
        The (approximate) eigenvalue(s) of the mode(s) to visualise.
    ef_name : str
        The name of the eigenfunction to visualise.
    use_real_part : bool
        Whether to use the real part of the eigenmode solution.
    complex_factor : list[list[complex]]
        A complex factor to multiply the eigenmode solution with.
    add_background : bool
        Whether to add the equilibrium background to the eigenmode solution.

    Attributes
    ----------
    ds : ~pylbo.data_containers.LegolasDataSeries
        The dataseries containing the eigenfunctions and modes to visualise.
    omega : list[list[complex]]
        The (approximate) eigenvalue(s) of the mode(s) to visualise.
    eigenfunction : list[list[np.ndarray]]
        The eigenfunction of the mode(s) to visualise.
    use_real_part : bool
        Whether to use the real part of the eigenmode solution.
    complex_factor : list[list[complex]]
        The complex factors to multiply the eigenmode solution with.
    add_background : bool
        Whether to add the equilibrium background to the eigenmode solution.
    """

    def __init__(
        self,
        ds: LegolasDataSeries,
        omega: list[np.ndarray[complex]],
        ef_name: str = None,
        use_real_part: bool = True,
        complex_factor: list[np.ndarray[complex]] = None,
        add_background: bool = False,
    ) -> None:
        self.ds = ds
        self.ds_bg = self.ds.datasets[0]
        self.use_real_part = use_real_part
        if add_background and not self.ds_bg.has_background:
            raise BackgroundNotPresent(self.ds_bg.datfile, "add background to solution")
        self.add_background = add_background
        self._print_bg_info = True

        self._ef_name = None if ef_name is None else validate_ef_name(ds, ef_name)
        self._ef_name_latex = None if ef_name is None else self.get_ef_name_latex()
        self._all_efs = [
            dataset.get_eigenfunctions(ev_guesses=omega[i])
            for i, dataset in enumerate(self.ds.datasets)
        ]
        self.omega = []
        self.eigenfunction = []
        for all_efs in self._all_efs:
            self.omega.append([efs.get("eigenvalue") for efs in all_efs])
            self.eigenfunction.append([efs.get(self._ef_name) for efs in all_efs])
        self.complex_factor = self._validate_complex_factor(complex_factor)

    @property
    def k2(self) -> float:
        """The k2 wave number of the eigenmode solution."""
        return self.ds.parameters["k2"]

    @property
    def k3(self) -> float:
        """The k3 wave number of the eigenmode solution."""
        return self.ds.parameters["k3"]

    @property
    def part_name(self) -> str:
        """
        Returns the name of the part of the eigenmode solution to use, i.e.
        'real' or 'imag'.
        """
        return "real" if self.use_real_part else "imag"

    def get_ef_name_latex(self) -> str:
        """Returns the latex representation of the eigenfunction name."""
        return ef_name_to_latex(
            self._ef_name, geometry=self.ds.geometry, real_part=self.use_real_part
        )

    def _validate_complex_factor(
        self, complex_factor: list[list[complex]]
    ) -> list[list[complex]]:
        """
        Validates the complex factors.

        Parameters
        ----------
        complex_factor : list[list[complex]]
            The complex factor to validate.

        Returns
        -------
        complex
            The complex factor if it is valid, otherwise 1.
        """
        if complex_factor is None:
            complex_factor = []
            for omegas in self.omega:
                complex_factor.append([1]*len(omegas))
        
        return complex_factor
        

    def get_mode_solution(
        self,
        ef: np.ndarray,
        omega: complex,
        complex_factor: complex,
        u2: Union[float, np.ndarray],
        u3: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
        k2: float,
        k3: float,
    ) -> np.ndarray:
        """
        Calculates the full eigenmode solution for given coordinates and time.
        If a complex factor was given, the eigenmode solution is multiplied with the
        complex factor. If :attr:`use_real_part` is True the real part of the eigenmode
        solution is returned, otherwise the complex part.

        Parameters
        ----------
        ef : np.ndarray
            The eigenfunction to use.
        omega : complex
            The eigenvalue to use.
        complex_factor : complex,
            The complex factor to multiply with.
        u2 : Union[float, np.ndarray]
            The y coordinate(s) of the eigenmode solution.
        u3 : Union[float, np.ndarray]
            The z coordinate(s) of the eigenmode solution.
        t : Union[float, np.ndarray]
            The time(s) of the eigenmode solution.
        k2 : float
            The x2 wavenumber of the mode.
        k3 : float
            The x3 wavenumber of the mode.

        Returns
        -------
        np.ndarray
            The real or imaginary part of the eigenmode solution for the given
            set of coordinate(s) and time(s).
        """
        solution = (
            complex_factor * ef * np.exp(1j * k2 * u2 + 1j * k3 * u3 - 1j * omega * t)
        )
        return getattr(solution, self.part_name)

    def get_background(self, shape: tuple[int, ...], name=None) -> np.ndarray:
        """
        Returns the background of the eigenmode solution.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the eigenmode solution.
        name : str
            The name of the background to use. If None, the background name
            will be inferred from the eigenfunction name.

        Returns
        -------
        np.ndarray
            The background of the eigenmode solution, sampled on the eigenfunction
            grid and broadcasted to the same shape as the eigenmode solution.
        """
        if name is None:
            name = self._get_background_name()
        bg = self.ds_bg.equilibria.get(name, np.zeros(self.ds_bg.gauss_gridpoints))
        bg_sampled = self._sample_background_on_ef_grid(bg)
        if self._print_bg_info:
            pylboLogger.info(f"background {name} broadcasted to shape {shape}")
        return np.broadcast_to(bg_sampled, shape=reversed(shape)).transpose()

    def _sample_background_on_ef_grid(self, bg: np.ndarray) -> np.ndarray:
        """
        Samples the background array on the eigenfunction grid.

        Parameters
        ----------
        bg : np.ndarray
            The background array with Gaussian grid spacing

        Returns
        -------
        np.ndarray
            The background array with eigenfunction grid spacing
        """
        if self._print_bg_info:
            pylboLogger.info(
                f"sampling background [{len(bg)}] on eigenfunction grid "
                f"[{len(self.ds_bg.ef_grid)}]"
            )
        return np.interp(self.ds_bg.ef_grid, self.ds_bg.grid_gauss, bg)

    def _get_background_name(self) -> str:
        """
        Returns the name of the background.

        Returns
        -------
        str
            The closest match between the eigenfunction name and the equilibrium
            name.

        Raises
        ------
        ValueError
            If the eigenfunction name is a magnetic vector potential component.
        """
        if self._ef_name in ("a1", "a2", "a3"):
            raise ValueError(
                "Unable to add a background to the magnetic vector potential."
            )
        name = None
        name_temp = difflib.get_close_matches(self._ef_name, self.ds_bg.eq_names, 1)
        if name_temp != []:
            (name,) = name_temp
        if self._print_bg_info and name is not None:
            pylboLogger.info(
                f"adding background for '{self._ef_name}', closest match is '{name}'"
            )
        else:
            print(f"Background is zero or not implemented for '{self._ef_name}'")
        return name
