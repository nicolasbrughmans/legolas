submodule (mod_matrix_manager) smod_heatloss_matrix
  implicit none

contains

  module procedure add_heatloss_matrix_terms
    real(dp) :: rho
    real(dp) :: Lrho, LT, L0
    real(dp) :: gamma_1
    type(matrix_elements_t) :: elements

    if (settings%physics%is_incompressible) return

    gamma_1 = settings%physics%get_gamma_1()
    rho = background%density%rho0(x_gauss)
    Lrho = physics%heatloss%get_dLdrho(x_gauss)
    LT = physics%heatloss%get_dLdT(x_gauss)
    L0 = physics%heatloss%get_L0(x_gauss)

    elements = new_matrix_elements(state_vector=settings%get_state_vector())

    call elements%add(-ic * gamma_1 * (L0 + rho * Lrho), "T", "rho", h_quad, h_quad)
    call elements%add(-ic * gamma_1 * rho * LT, "T", "T", h_quad, h_quad)

    call add_to_quadblock(quadblock, elements, weight, settings%dims)
    call elements%delete()
  end procedure add_heatloss_matrix_terms

end submodule smod_heatloss_matrix
