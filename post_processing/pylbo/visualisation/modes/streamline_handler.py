import numpy as np
from typing import Union
from matplotlib.axes import Axes as mpl_axes
from matplotlib.streamplot import StreamplotSet
from matplotlib.quiver import Quiver
from pylbo.data_containers import LegolasDataContainer, LegolasDataSet
from pylbo.visualisation.eigenfunctions.eigfunc_interface import EigenfunctionInterface
from pylbo.visualisation.utils import ef_name_to_latex
from pylbo.visualisation.modes.mode_data import ModeVisualisationData
from pylbo.visualisation.modes.mode_figure import ModeFigure

class StreamlineHandler:
    """
    Main handler for streamlines.
    """

    def __init__(
            self, 
            xgrid : np.ndarray, 
            coordgrid : np.ndarray, 
            field : str, 
            data : ModeVisualisationData, 
            axes : mpl_axes, 
            add_background : bool,
            **kwargs
            ):
        
        self.xgrid = xgrid
        self.coordgrid = coordgrid
        self.data = data

        self.field = field
        self.ax = axes
        self.polar = (self.ax.name == "polar")
        self.add_background = add_background

        self.coord_dict = {"theta" : "3", "z" : "2", "y" : "3"}
        self.streamlines = None
        self._kwargs = kwargs
        if "density" not in kwargs.keys(): self._kwargs["density"] = 2.0
        if "color" not in kwargs.keys(): self._kwargs["color"] = "w"
        if "broken_streamlines" not in kwargs.keys(): self._kwargs["broken_streamlines"] = True


    def draw_streamlines(self) -> StreamplotSet:
        self._clear_streamlines()
        if self.polar:
            xdata = self.coord_data
            ydata = self.u1_data
            xvec = self._solutions[1]
            yvec = self._solutions[0]
        else:
            xdata = self.u1_data.transpose()
            ydata = self.coord_data.transpose()
            xvec = self._solutions[0].transpose()
            yvec = self._solutions[1].transpose()
        print("Just before end")
        self.streamlines = self.ax.streamplot(xdata, ydata, xvec, yvec, zorder=2, **self._kwargs)
        self.streamlines.lines.set_alpha(0.7)
        self.streamlines.arrows.set_alpha(0.7)
        print("After end")

    def _clear_streamlines(self) -> None:
        try:
            self.streamlines.remove()
        except AttributeError:
            pass

    ## Further implement vectorplots, perhaps rename some things to make them more general
    ## Also make sure that in 2d plot, redrawing etc also takes care of the vectors and streamlines. 
    ## In particular, this has to be the case for movie making, but also look at drawing procedures (is there need for flexible redrawing?)

    # def draw_vectors(self) -> Quiver:
    #     if self.streamlines is not None: self.streamlines.clear()
    #     if self.polar:
    #         xdata = self.coord_data
    #         ydata = self.u1_data
    #         xvec = self._solutions[1]
    #         yvec = self._solutions[0]
    #     else:
    #         xdata = self.u1_data.transpose()
    #         ydata = self.coord_data.transpose()
    #         xvec = self._solutions[0].transpose()
    #         yvec = self._solutions[1].transpose()
    #     print("Just before end")
    #     self.streamlines = self.ax.quiver(xdata, ydata, xvec, yvec, color="w", zorder=2, broken_streamlines=True, density=2)
    #     self.streamlines.lines.set_alpha(0.7)
    #     self.streamlines.arrows.set_alpha(0.7)
    #     print("After end")

    def set_slicing_axis(self, slicing_axis, u2axis, u3axis) -> None:
        self.slicing_axis = slicing_axis
        self._u2axis = u2axis
        self._u3axis = u3axis

    def set_time(self, time) -> None:
        self.time_data = time

    def set_streamplot_arrays(self, u2, u3) -> None:
        axis = self.slicing_axis
        self.solution_shape = (len(self.xgrid), len(self.coordgrid))
        x_2d, coord_2d = np.meshgrid(self.xgrid, self.coordgrid, indexing="ij")

        self.coord_data = coord_2d
        self.u1_data = x_2d
        self.u2_data = coord_2d if axis == self._u3axis else u2
        self.u3_data = coord_2d if axis == self._u2axis else u3
        

    def set_solutions(self) -> np.ndarray:
        """
        Returns the eigenmode solution for a given time.

        Parameters
        ----------
        u2_data : Union[float, np.ndarray]
            The u2 data from the Plot2d.
        u3_data : Union[float, np.ndarray]
            The u3 data from the Plot2d.

        Returns
        -------
        np.ndarray
            The eigenmode solution.
        """
        # name = validate_ef_name(self.data.ds_bg, name)
        fields = self.get_field_names()
        solutions = [np.zeros_like(self.u1_data), np.zeros_like(self.u1_data)]

        for all_efs, k2, k3 in zip(self.data._all_efs, self.data.k2, self.data.k3):
            for ef_highres in all_efs:
                for i in range(2):
                    ef = ef_highres.get(fields[i])
                    ef = np.interp(self.xgrid, self.data.ds_bg.ef_grid, ef)
                    ef = np.broadcast_to(ef, shape=reversed(self.solution_shape)).transpose()
                    solutions[i] += self.data.get_mode_solution(
                        ef=ef,
                        omega=ef_highres.get("eigenvalue"),
                        u2=self.u2_data,
                        u3=self.u3_data,
                        t=self.time_data,
                        k2=k2,
                        k3=k3
                    )

        if self.add_background:
            bgs = self.get_bg_names()
            for i in range(2):
                bg_temp = self.data.get_background(name=bgs[i], shape=self.data.ds_bg.ef_grid.shape)
                bg = np.interp(self.xgrid, self.data.ds_bg.ef_grid, bg_temp)
                bg = np.broadcast_to(bg, shape=reversed(self.solution_shape)).transpose()
                solutions[i] += bg

        self._solutions = solutions
    

    def get_field_names(self) -> list:
        field1 = self.field + "1"
        field2 = self.field + self.coord_dict[self.slicing_axis]
        return [field1, field2]
    
    def get_bg_names(self) -> list:
        bg1 = self.field + "01"
        bg2 = self.field + "0" + self.coord_dict[self.slicing_axis]
        return [bg1, bg2]