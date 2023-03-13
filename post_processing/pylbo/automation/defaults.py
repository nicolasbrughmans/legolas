import numpy as np

namelist_items = {
    "gridlist": [
        ("geometry", str),
        ("x_start", (int, np.integer, float)),
        ("x_end", (int, np.integer, float)),
        ("gridpoints", (int, np.integer)),
        ("force_r0", bool),
        ("coaxial", bool),
    ],
    "equilibriumlist": [
        ("equilibrium_type", str),
        ("boundary_type", str),
        ("use_defaults", bool),
    ],
    "savelist": [
        ("write_matrices", bool),
        ("write_eigenvectors", bool),
        ("write_residuals", bool),
        ("write_eigenfunctions", bool),
        ("write_derived_eigenfunctions", bool),
        ("show_results", bool),
        ("basename_datfile", str),
        ("output_folder", str),
        ("logging_level", (int, np.integer)),
        ("write_eigenfunction_subset", bool),
        ("eigenfunction_subset_center", complex),
        ("eigenfunction_subset_radius", (int, np.integer, float)),
    ],
    "physicslist": [
        ("physics_type", str),
        ("mhd_gamma", float),
        ("incompressible", bool),
        ("dropoff_edge_dist", (int, np.integer, float)),
        ("dropoff_width", (int, np.integer, float)),
        ("flow", bool),
        ("radiative_cooling", bool),
        ("ncool", (int, np.integer)),
        ("cooling_curve", str),
        ("external_gravity", bool),
        ("parallel_conduction", bool),
        ("perpendicular_conduction", bool),
        ("fixed_tc_para_value", (int, np.integer, float)),
        ("fixed_tc_perp_value", (int, np.integer, float)),
        ("resistivity", bool),
        ("fixed_resistivity_value", (int, np.integer, float)),
        ("use_eta_dropoff", bool),
        ("viscosity", bool),
        ("viscosity_value", (int, np.integer, float)),
        ("viscous_heating", bool),
        ("hall_mhd", bool),
        ("hall_dropoff", bool),
        ("elec_inertia", bool),
        ("inertia_dropoff", bool),
        ("electron_fraction", (int, np.integer, float)),
    ],
    "unitslist": [
        ("unit_density", (int, np.integer, float)),
        ("unit_temperature", (int, np.integer, float)),
        ("unit_magneticfield", (int, np.integer, float)),
        ("unit_length", (int, np.integer, float)),
        ("mean_molecular_weight", (int, np.integer, float)),
    ],
    "solvelist": [
        ("solver", str),
        ("arpack_mode", str),
        ("number_of_eigenvalues", (int, np.integer)),
        ("which_eigenvalues", str),
        ("maxiter", (int, np.integer)),
        ("sigma", (int, np.integer, float, complex)),
        ("ncv", (int, np.integer)),
        ("tolerance", (int, np.integer, float)),
    ],
}
