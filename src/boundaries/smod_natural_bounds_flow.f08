submodule (mod_boundary_manager:smod_natural_boundaries) smod_natural_bounds_flow
  implicit none

contains

  module procedure add_natural_flow_terms
    use mod_equilibrium, only: v_field

    real(dp)  :: rho
    real(dp)  :: v01
    type(matrix_elements_t) :: elements

    if (.not. settings%physics%flow%is_enabled()) return

    rho = rho_field % rho0(grid_idx)
    v01 = v_field % v01(grid_idx)
    elements = new_matrix_elements(state_vector=settings%get_state_vector())

    ! ==================== Cubic * Cubic ====================
    call elements%add(-ic * rho * v01, "v1", "v1", spline1=h_cubic, spline2=h_cubic)
    ! ==================== Quadratic * Quadratic ====================
    call elements%add(-ic * rho * v01, "v3", "v3", spline1=h_quad, spline2=h_quad)
    call elements%add(-ic * rho * v01, "T", "T", spline1=h_quad, spline2=h_quad)

    call add_to_quadblock(quadblock, elements, weight, settings%dims)
    call elements%delete()
  end procedure add_natural_flow_terms

end submodule smod_natural_bounds_flow
