submodule (mod_matrix_manager) smod_penalty_matrix

  implicit none

contains

  module procedure add_penalty_matrix_terms
    use mod_global_variables, only: gauge, penalty

    real(dp)  :: eps, deps
    real(dp)  :: B01, B02, B03

    ! grid variables
    eps = eps_grid(gauss_idx)
    deps = d_eps_grid_dr(gauss_idx)

    ! equilibrium variables
    B01 = B_field % B01
    B02 = B_field % B02(gauss_idx)
    B03 = B_field % B03(gauss_idx)

    ! ==================== Quadratic * Quadratic ====================
    call reset_factor_positions(new_size=1)
    ! P(6, 6)
    if (gauge == 'Coulomb') then
      factors(1) = -penalty * deps**2
    else if (gauge == 'B__parallel') then
      factors(1) = -penalty * B01**2
    else
      factors(1) = 0.0d0
    end if
    positions(1, :) = [6, 6]
    call subblock(quadblock, factors, positions, current_weight, h_quad, h_quad)

    ! ==================== Quadratic * dQuadratic ====================
    call reset_factor_positions(new_size=1)
    ! P(6, 6)
    if (gauge == 'Coulomb') then
      factors(1) = -penalty * deps * eps
    else
      factors(1) = 0.0d0
    end if
    positions(1, :) = [6, 6]
    call subblock(quadblock, factors, positions, current_weight, h_quad, dh_quad)

    ! ==================== dQuadratic * Quadratic ====================
    call reset_factor_positions(new_size=1)
    ! P(6, 6)
    if (gauge == 'Coulomb') then
      factors(1) = -penalty * deps * eps
    else
      factors(1) = 0.0d0
    end if
    positions(1, :) = [6, 6]
    call subblock(quadblock, factors, positions, current_weight, dh_quad, h_quad)

    ! ==================== dQuadratic * dQuadratic ====================
    call reset_factor_positions(new_size=1)
    ! P(6, 6)
    if (gauge == 'Coulomb') then
      factors(1) = -penalty * eps**2
    else
      factors(1) = 0.0d0
    end if
    positions(1, :) = [6, 6]
    call subblock(quadblock, factors, positions, current_weight, dh_quad, dh_quad)

    ! ==================== Quadratic * Cubic ====================
    call reset_factor_positions(new_size=2)
    ! P(6, 7)
    positions(1, :) = [6, 7]
    ! P(6, 8)
    positions(2, :) = [6, 8]
    if (gauge == 'Coulomb') then
      factors(1) = penalty * k2 * deps / eps
      factors(2) = penalty * k3 * deps * eps
    else if (gauge == 'B_parallel') then
      factors(1) = -penalty * ic * B01 * B02 / eps
      factors(2) = -penalty * ic * B01 * B03
    else
      factors(1) = 0.0d0
      factors(2) = 0.0d0
    end if
    call subblock(quadblock, factors, positions, current_weight, h_quad, h_cubic)

    ! ==================== dQuadratic * Cubic ====================
    call reset_factor_positions(new_size=2)
    ! P(6, 7)
    positions(1, :) = [6, 7]
    ! P(6, 8)
    positions(2, :) = [6, 8]
    if (gauge == 'Coulomb') then
      factors(1) = penalty * k2
      factors(2) = penalty * k3 * eps**2
    else
      factors(1) = 0.0d0
      factors(2) = 0.0d0
    end if
    call subblock(quadblock, factors, positions, current_weight, dh_quad, h_cubic)

    ! ==================== Cubic * Quadratic ====================
    call reset_factor_positions(new_size=2)
    ! P(7, 6)
    positions(1, :) = [7, 6]
    ! P(8, 6)
    positions(2, :) = [8, 6]
    if (gauge == 'Coulomb') then
      factors(1) = penalty * k2 * deps / eps
      factors(2) = penalty * k3 * deps * eps
    else if (gauge == 'B_parallel') then
      factors(1) = -penalty * ic * B01 * B02 / eps
      factors(2) = -penalty * ic * B01 * B03
    else
      factors(1) = 0.0d0
      factors(2) = 0.0d0
    end if
    call subblock(quadblock, factors, positions, current_weight, h_cubic, h_quad)

    ! ==================== Cubic * dQuadratic ====================
    call reset_factor_positions(new_size=2)
    ! P(7, 6)
    positions(1, :) = [7, 6]
    ! P(8, 6)
    positions(2, :) = [8, 6]
    if (gauge == 'Coulomb') then
      factors(1) = penalty * k2
      factors(2) = penalty * k3 * eps**2
    else
      factors(1) = 0.0d0
      factors(2) = 0.0d0
    end if
    call subblock(quadblock, factors, positions, current_weight, h_cubic, dh_quad)

    ! ==================== Cubic * Cubic ====================
    call reset_factor_positions(new_size=4)
    ! P(7, 7)
    positions(1, :) = [7, 7]
    ! P(7, 8)
    positions(2, :) = [7, 8]
    ! P(8, 7)
    positions(3, :) = [8, 7]
    ! P(8, 8)
    positions(4, :) = [8, 8]
    if (gauge == 'Coulomb') then
      factors(1) = -penalty * (k2 / eps)**2
      factors(2) = -penalty * k2 * k3
      factors(3) = -penalty * k2 * k3
      factors(4) = -penalty * (k3 * eps)**2
    else if (gauge == 'B_parallel') then
      factors(1) = penalty * (B02 / eps)**2
      factors(2) = penalty * B02 * B03 / eps
      factors(3) = penalty * B02 * B03 / eps
      factors(4) = penalty * B03**2
    else
      factors(1) = 0.0d0
      factors(2) = 0.0d0
      factors(3) = 0.0d0
      factors(4) = 0.0d0
    end if
    call subblock(quadblock, factors, positions, current_weight, h_cubic, h_cubic)

  end procedure add_penalty_matrix_terms
end submodule smod_penalty_matrix
