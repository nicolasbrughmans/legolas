submodule (mod_matrix_manager) smod_regular_matrix
  implicit none

contains

  module procedure add_bmatrix_terms
    type(matrix_elements_t) :: elements
    real(dp)  :: rho, eps

    rho = rho_field % rho0(gauss_idx)
    eps = eps_grid(gauss_idx)
    elements = new_matrix_elements(state_vector=settings%get_state_vector())

    ! Quadratic * Quadratic
    call elements%add(1.0_dp, "rho", "rho", spline1=h_quad, spline2=h_quad)
    call elements%add(eps * rho, "v2", "v2", spline1=h_quad, spline2=h_quad)
    call elements%add(rho, "v3", "v3", spline1=h_quad, spline2=h_quad)
    call elements%add(rho, "T", "T", spline1=h_quad, spline2=h_quad)
    call elements%add(eps, "a1", "a1", spline1=h_quad, spline2=h_quad)

    ! Cubic * Cubic
    call elements%add(rho, "v1", "v1", spline1=h_cubic, spline2=h_cubic)
    call elements%add(1.0_dp, "a2", "a2", spline1=h_cubic, spline2=h_cubic)
    call elements%add(eps, "a3", "a3", spline1=h_cubic, spline2=h_cubic)

    call add_to_quadblock(quadblock, elements, current_weight, settings%dims)
    call elements%delete()
  end procedure add_bmatrix_terms


  module procedure add_regular_matrix_terms
    use mod_equilibrium, only: grav_field

    type(matrix_elements_t) :: elements
    real(dp)  :: eps, deps
    real(dp)  :: rho, drho
    real(dp)  :: T0, dT0
    real(dp)  :: B01, B02, dB02, drB02, B03, db03
    real(dp)  :: Fop_plus, Gop_plus, Gop_min, WVop
    real(dp) :: gamma_1

    gamma_1 = settings%physics%get_gamma_1()

    ! grid variables
    eps = eps_grid(gauss_idx)
    deps = d_eps_grid_dr(gauss_idx)
    ! density variables
    rho = rho_field % rho0(gauss_idx)
    drho = rho_field % d_rho0_dr(gauss_idx)
    ! temperature variables
    T0 = T_field % T0(gauss_idx)
    dT0 = T_field % d_T0_dr(gauss_idx)
    ! magnetic field variables
    B01 = B_field % B01
    B02 = B_field % B02(gauss_idx)
    dB02 = B_field % d_B02_dr(gauss_idx)
    drB02 = deps * B02 + eps * dB02
    B03 = B_field % B03(gauss_idx)
    dB03 = B_field % d_B03_dr(gauss_idx)
    ! operators
    Fop_plus = get_F_operator(gauss_idx, which="plus")
    Gop_plus = get_G_operator(gauss_idx, which="plus")
    Gop_min = get_G_operator(gauss_idx, which="minus")
    WVop = get_wv_operator(gauss_idx)

    elements = new_matrix_elements(state_vector=settings%get_state_vector())

    ! ==================== Quadratic * Cubic ====================
    call elements%add(-drho, "rho", "v1", spline1=h_quad, spline2=h_cubic)
    call elements%add( &
      k3 * (drB02 - ic * k2 * B01) / eps, "v2", "a2", spline1=h_quad, spline2=h_cubic &
    )
    call elements%add( &
      k2 * (ic * k2 * B01 - drB02) / eps, "v2", "a3", spline1=h_quad, spline2=h_cubic &
    )
    call elements%add( &
      k3 * (dB03 - ic * k3 * B01), "v3", "a2", spline1=h_quad, spline2=h_cubic &
    )
    call elements%add( &
      k2 * (ic * k3 * B01 - dB03), "v3", "a3", spline1=h_quad, spline2=h_cubic &
    )
    if (.not. settings%physics%is_incompressible) call elements%add( &
      -dT0 * rho, "T", "v1", spline1=h_quad, spline2=h_cubic &
    )

    ! ==================== Quadratic * dCubic ====================
    call elements%add(-rho, "rho", "v1", spline1=h_quad, spline2=dh_cubic)
    call elements%add(k2 * B03 / eps, "v2", "a2", spline1=h_quad, spline2=dh_cubic)
    call elements%add(eps * k3 * B03, "v2", "a3", spline1=h_quad, spline2=dh_cubic)
    call elements%add( &
      -(k2 * B02 + ic * deps * B01) / eps, &
      "v3", &
      "a2", &
      spline1=h_quad, &
      spline2=dh_cubic &
    )
    call elements%add(-eps * k3 * B02, "v3", "a3", spline1=h_quad, spline2=dh_cubic)
    call elements%add(-gamma_1 * T0 * rho, "T", "v1", spline1=h_quad, spline2=dh_cubic)

    ! ==================== Quadratic * Quadratic ====================
    call elements%add(rho * k2, "rho", "v2", spline1=h_quad, spline2=h_quad)
    call elements%add(rho * k3, "rho", "v3", spline1=h_quad, spline2=h_quad)
    call elements%add(k2 * T0 / eps, "v2", "rho", spline1=h_quad, spline2=h_quad)
    call elements%add(k2 * rho / eps, "v2", "T", spline1=h_quad, spline2=h_quad)
    call elements%add(-WVop * B03, "v2", "a1", spline1=h_quad, spline2=h_quad)
    call elements%add(k3 * T0, "v3", "rho", spline1=h_quad, spline2=h_quad)
    call elements%add(k3 * rho, "v3", "T", spline1=h_quad, spline2=h_quad)
    call elements%add( &
      ic * deps * k2 * B01 / eps + B02 * WVop, &
      "v3", &
      "a1", &
      spline1=h_quad, &
      spline2=h_quad &
    )
    call elements%add( &
      gamma_1 * k2 * rho * T0, "T", "v2", spline1=h_quad, spline2=h_quad &
    )
    call elements%add( &
      gamma_1 * k3 * rho * T0, "T", "v3", spline1=h_quad, spline2=h_quad &
    )
    call elements%add(-eps * B03, "a1", "v2", spline1=h_quad, spline2=h_quad)
    call elements%add(B02, "a1", "v3", spline1=h_quad, spline2=h_quad)

    ! ==================== Cubic * Quadratic ====================
    call elements%add(-deps * T0 / eps, "v1", "rho", spline1=h_cubic, spline2=h_quad)
    if (settings%physics%gravity%is_enabled()) call elements%add( &
      grav_field%grav(gauss_idx), "v1", "rho", spline1=h_cubic, spline2=h_quad &
    )
    call elements%add(-deps * rho / eps, "v1", "T", spline1=h_cubic, spline2=h_quad)
    call elements%add(deps * Gop_plus, "v1", "a1", spline1=h_cubic, spline2=h_quad)
    call elements%add(ic * B01, "a2", "v3", spline1=h_cubic, spline2=h_quad)
    call elements%add(-ic * eps * B01, "a3", "v2", spline1=h_cubic, spline2=h_quad)

    ! ==================== dCubic * Quadratic ====================
    call elements%add(-T0, "v1", "rho", spline1=dh_cubic, spline2=h_quad)
    call elements%add(-rho, "v1", "T", spline1=dh_cubic, spline2=h_quad)
    call elements%add(-eps * Gop_min, "v1", "a1", spline1=dh_cubic, spline2=h_quad)

    ! ==================== Cubic * Cubic ====================
    call elements%add(-k3 * Fop_plus, "v1", "a2", spline1=h_cubic, spline2=h_cubic)
    call elements%add(k2 * Fop_plus, "v1", "a3", spline1=h_cubic, spline2=h_cubic)
    call elements%add(-B03, "a2", "v1", spline1=h_cubic, spline2=h_cubic)
    call elements%add(B02, "a3", "v1", spline1=h_cubic, spline2=h_cubic)

    ! ==================== Cubic * dCubic ====================
    call elements%add(-deps * B03 / eps, "v1", "a2", spline1=h_cubic, spline2=dh_cubic)
    call elements%add(-deps * B02, "v1", "a3", spline1=h_cubic, spline2=dh_cubic)

    ! ==================== dCubic * dCubic ====================
    call elements%add(-B03, "v1", "a2", spline1=dh_cubic, spline2=dh_cubic)
    call elements%add(eps * B02, "v1", "a3", spline1=dh_cubic, spline2=dh_cubic)

    ! ==================== dQuadratic * Quadratic ====================
    call elements%add(-ic * eps * k3 * B01, "v2", "a1", spline1=dh_quad, spline2=h_quad)
    call elements%add(ic * k2 * B01, "v3", "a1", spline1=dh_quad, spline2=h_quad)

    ! ==================== dQuadratic * dCubic ====================
    call elements%add(ic * eps * B01, "v2", "a3", spline1=dh_quad, spline2=dh_cubic)
    call elements%add(-ic * B01, "v3", "a2", spline1=dh_quad, spline2=dh_cubic)

    call add_to_quadblock(quadblock, elements, current_weight, settings%dims)
    call elements%delete()
  end procedure add_regular_matrix_terms

end submodule smod_regular_matrix
