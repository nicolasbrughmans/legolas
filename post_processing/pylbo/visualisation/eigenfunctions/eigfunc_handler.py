import numpy as np
from pylbo.data_containers import LegolasDataSet
from pylbo.exceptions import EigenfunctionsNotPresent
from pylbo.utilities.toolbox import transform_to_numpy
from pylbo.visualisation.eigenfunctions.eigfunc_interface import EigenfunctionInterface
from pylbo.visualisation.utils import ef_name_to_latex


class EigenfunctionHandler(EigenfunctionInterface):
    """
    Main handler for eigenfunctions.
    """

    def __init__(self, data, ef_ax, spec_ax):
        super().__init__(data, ef_ax, spec_ax)
        # extract unique names and preserve order
        fnames, idxs = np.unique(self.data.ef_names, return_index=True)
        self._function_names = fnames[np.argsort(idxs)]
        self.spec_axis.set_title(f"{self.spec_axis.get_title()} -- eigenfunctions")

    def _check_data_is_present(self):
        if not any(transform_to_numpy(self.data.has_efs)):
            raise EigenfunctionsNotPresent(
                "None of the given datfiles has eigenfunctions written to it."
            )

    def update_plot(self):
        self.axis.clear()
        if not self._selected_idxs:
            self._display_tooltip()
            return
        ef_name = self._function_names[self._selected_name_idx]
        for ds, idxs_dict in self._selected_idxs.items():
            idxs = np.array([int(idx) for idx in idxs_dict.keys()])
            ef_container = ds.get_eigenfunctions(ev_idxs=idxs)
            for ev_idx, efs in zip(idxs, ef_container):
                ef = efs.get(ef_name)
                if self._use_real_part:
                    ef = ef.real
                else:
                    ef = ef.imag
                # check for retransform
                if (
                    self._retransform
                    and ef_name in ("rho", "v1", "v3", "T", "a2")
                    and self.data.geometry == "cylindrical"
                ):
                    ef = ef * ds.ef_grid
                label = super()._get_label(ds, ev_idx, efs.get("eigenvalue"))
                # get color of selected point
                color = self._selected_idxs.get(ds).get(str(ev_idx)).get_color()
                self.axis.plot(ds.ef_grid, ef, color=color, label=label)
                if self._draw_resonances:
                    self._show_resonances(ds, ev_idx, color)
        self.axis.axhline(y=0, linestyle="dotted", color="grey")
        if isinstance(self.data, LegolasDataSet):
            self.axis.axvline(x=self.data.x_start, linestyle="dotted", color="grey")
        self.axis.set_title(self._get_title(ef_name))
        self.axis.legend(loc="best", fontsize=8)

    def _get_title(self, ef_name):
        """
        Creates the title of the eigenfunction plot.
        If the eigenfunction is retransformed an 'r' is prepended if appropriate,
        along with Re/Im depending on the real/imaginary part shown.

        Parameters
        ----------
        ef_name : str
            The name of the eigenfunction as present in the datfile.

        Returns
        -------
        name : str
            The 'new' name for the eigenfunction, used as title.
        """
        if (
            self.data.geometry == "cylindrical"
            and ef_name in ("rho", "v1", "v3", "T", "a2")
            and self._retransform
        ):
            ef_name = f"r{ef_name}"

        ef_name = ef_name_to_latex(
            ef_name, geometry=self.data.geometry, real_part=self._use_real_part
        )
        name = rf"{ef_name} eigenfunction"
        return name

    def _mark_points_without_data_written(self):
        self._condition_to_make_transparent = "has_efs"
        super()._mark_points_without_data_written()
