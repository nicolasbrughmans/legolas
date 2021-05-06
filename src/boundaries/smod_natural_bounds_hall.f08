submodule (mod_boundary_manager:smod_natural_boundaries) smod_natural_bounds_hall
  implicit none

contains

  module procedure add_natural_hall_terms
    use mod_global_variables, only: hall_mhd, elec_pressure, gauge
    use mod_equilibrium, only: hall_field

    real(dp)  :: eps
    real(dp)  :: rho, T0, B01, B02, B03
    real(dp)  :: eta_H

    if (.not. hall_mhd) then
      return
    end if

    eps = eps_grid(grid_idx)

    rho = rho_field % rho0(grid_idx)
    T0 = T_field % T0(grid_idx)
    B01 = B_field % B01
    B02 = B_field % B02(grid_idx)
    B03 = B_field % B03(grid_idx)

    eta_H = hall_field % hallfactor(grid_idx)

    ! ==================== Quadratic * Quadratic ====================
    call reset_factor_positions(new_size=1)
    ! H(6, 6)
    factors(1) = -eta_H * eps * (k3 * B02 - k2 * B03 / eps) / rho
    positions(1, :) = [6, 6]
    call subblock(quadblock, factors, positions, weight, h_quad, h_quad)
    if (elec_pressure) then
      call reset_factor_positions(new_size=2)
      ! H(6, 1)
      factors(1) = -eta_H * T0 / rho
      positions(1, :) = [6, 1]
      ! H(6, 5)
      factors(2) = -eta_H
      positions(2, :) = [6, 5]
      call subblock(quadblock, factors, positions, weight, h_quad, h_quad)
    end if

    ! ==================== Quadratic * dCubic ====================
    call reset_factor_positions(new_size=2)
    ! H(6, 7)
    factors(1) = -eta_H * B03 / rho
    positions(1, :) = [6, 7]
    ! H(6, 8)
    factors(2) = eta_H * eps * B02 / rho
    positions(2, :) = [6, 8]
    call subblock(quadblock, factors, positions, weight, h_quad, dh_cubic)

    ! ==================== Cubic * dCubic ====================
    if (gauge == 'Coulomb') then
      call reset_factor_positions(new_size=2)
      ! H(7, 8)
      factors(1) = eta_H * ic * B01 * eps / rho
      positions(1, :) = [7, 8]
      ! H(8, 7)
      factors(2) = -eta_H * ic * B01 / rho
      positions(2, :) = [8, 7]
      call subblock(quadblock, factors, positions, weight, h_cubic, dh_cubic)
    end if

  end procedure add_natural_hall_terms

  module procedure add_natural_hall_Bterms
    use mod_global_variables, only: elec_inertia, gauge
    use mod_equilibrium, only: hall_field

    real(dp)  :: eps, rho, eta_e

    if (.not. (elec_inertia .and. (gauge == 'Coulomb'))) then
      return
    end if

    eps = eps_grid(grid_idx)
    rho = rho_field % rho0(grid_idx)
    eta_e = hall_field % inertiafactor(grid_idx)

    ! ==================== Cubic * dCubic ====================
    call reset_factor_positions(new_size=2)
    ! H(7, 7)
    factors(1) = -eta_e / rho
    positions(1, :) = [6, 7]
    ! H(8, 8)
    factors(2) = -eta_e * eps / rho
    positions(2, :) = [6, 8]
    call subblock(quadblock, factors, positions, weight, h_cubic, dh_cubic)

  end procedure add_natural_hall_Bterms

end submodule
