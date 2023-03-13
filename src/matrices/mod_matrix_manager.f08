module mod_matrix_manager
  use mod_global_variables, only: dp, ir, ic
  use mod_grid, only: grid, grid_gauss, eps_grid, d_eps_grid_dr
  use mod_build_quadblock, only: add_to_quadblock
  use mod_equilibrium, only: rho_field, T_field, B_field
  use mod_equilibrium_params, only: k2, k3
  use mod_logging, only: logger, str
  use mod_matrix_elements, only: matrix_elements_t, new_matrix_elements
  use mod_matrix_shortcuts, only: get_G_operator, get_F_operator, get_wv_operator
  use mod_settings, only: settings_t
  implicit none

  !> quadratic basis functions
  real(dp)  :: h_quad(4)
  !> derivative of quadratic basis functions
  real(dp)  :: dh_quad(4)
  !> cubic basis functions
  real(dp)  :: h_cubic(4)
  !> derivative of cubic basis functions
  real(dp)  :: dh_cubic(4)

  interface
    module subroutine add_bmatrix_terms(gauss_idx, current_weight, quadblock, settings)
      integer, intent(in)   :: gauss_idx
      real(dp), intent(in)  :: current_weight
      complex(dp), intent(inout)  :: quadblock(:, :)
      type(settings_t), intent(in) :: settings
    end subroutine add_bmatrix_terms

    module subroutine add_regular_matrix_terms( &
      gauss_idx, current_weight, quadblock, settings &
    )
      integer, intent(in)   :: gauss_idx
      real(dp), intent(in)  :: current_weight
      complex(dp), intent(inout)  :: quadblock(:, :)
      type(settings_t), intent(in) :: settings
    end subroutine add_regular_matrix_terms

    module subroutine add_flow_matrix_terms( &
      gauss_idx, current_weight, quadblock, settings &
    )
      integer, intent(in)   :: gauss_idx
      real(dp), intent(in)  :: current_weight
      complex(dp), intent(inout)  :: quadblock(:, :)
      type(settings_t), intent(in) :: settings
    end subroutine add_flow_matrix_terms

    module subroutine add_resistive_matrix_terms( &
      gauss_idx, current_weight, quadblock, settings &
    )
      integer, intent(in)   :: gauss_idx
      real(dp), intent(in)  :: current_weight
      complex(dp), intent(inout)  :: quadblock(:, :)
      type(settings_t), intent(in) :: settings
    end subroutine add_resistive_matrix_terms

    module subroutine add_cooling_matrix_terms( &
      gauss_idx, current_weight, quadblock, settings &
    )
      integer, intent(in)   :: gauss_idx
      real(dp), intent(in)  :: current_weight
      complex(dp), intent(inout)  :: quadblock(:, :)
      type(settings_t), intent(in) :: settings
    end subroutine add_cooling_matrix_terms

    module subroutine add_conduction_matrix_terms( &
      gauss_idx, current_weight, quadblock, settings &
    )
      integer, intent(in)   :: gauss_idx
      real(dp), intent(in)  :: current_weight
      complex(dp), intent(inout)  :: quadblock(:, :)
      type(settings_t), intent(in) :: settings
    end subroutine add_conduction_matrix_terms

    module subroutine add_viscosity_matrix_terms( &
      gauss_idx, current_weight, quadblock, settings &
    )
      integer, intent(in)   :: gauss_idx
      real(dp), intent(in)  :: current_weight
      complex(dp), intent(inout)  :: quadblock(:, :)
      type(settings_t), intent(in) :: settings
    end subroutine add_viscosity_matrix_terms

    module subroutine add_hall_matrix_terms( &
      gauss_idx, current_weight, quadblock, settings &
    )
      integer, intent(in)   :: gauss_idx
      real(dp), intent(in)  :: current_weight
      complex(dp), intent(inout)  :: quadblock(:, :)
      type(settings_t), intent(in) :: settings
    end subroutine add_hall_matrix_terms

    module subroutine add_hall_bmatrix_terms( &
      gauss_idx, current_weight, quadblock, settings &
    )
      integer, intent(in)   :: gauss_idx
      real(dp), intent(in)  :: current_weight
      complex(dp), intent(inout)  :: quadblock(:, :)
      type(settings_t), intent(in) :: settings
    end subroutine add_hall_bmatrix_terms
  end interface

  private

  public  :: build_matrices

contains

  subroutine build_matrices(matrix_B, matrix_A, settings)
    use mod_global_variables, only: n_gauss, gaussian_weights
    use mod_spline_functions, only: quadratic_factors, quadratic_factors_deriv, &
      cubic_factors, cubic_factors_deriv
    use mod_matrix_structure, only: matrix_t
    use mod_boundary_manager, only: apply_boundary_conditions

    !> the B-matrix
    type(matrix_t), intent(inout) :: matrix_B
    !> the A-matrix
    type(matrix_t), intent(inout) :: matrix_A
    !> the settings object
    type(settings_t), intent(in) :: settings

    !> quadblock for the A-matrix
    complex(dp), allocatable :: quadblock_A(:, :)
    !> quadblock for the B-matrix
    complex(dp), allocatable :: quadblock_B(:, :)
    !> left side of current interval
    real(dp)  :: x_left
    !> right side of current interval
    real(dp)  :: x_right
    !> current position in the Gaussian grid
    real(dp)  :: current_x_gauss
    !> current weight
    real(dp)  :: current_weight

    integer :: i, j, k, l, idx1, idx2
    integer :: quadblock_idx, gauss_idx, dim_quadblock

    dim_quadblock = settings%dims%get_dim_quadblock()
    allocate(quadblock_A(dim_quadblock, dim_quadblock))
    allocate(quadblock_B, mold=quadblock_A)
    ! used to shift the quadblock along the main diagonal
    quadblock_idx = 0

    do i = 1, settings%grid%get_gridpts() - 1
      ! reset quadblocks
      quadblock_A = (0.0d0, 0.0d0)
      quadblock_B = (0.0d0, 0.0d0)

      ! get interval boundaries
      x_left = grid(i)
      x_right = grid(i + 1)

      ! loop over Gaussian points to calculate integral
      do j = 1, n_gauss
        ! current grid index of Gaussian grid
        gauss_idx = (i - 1) * n_gauss + j
        current_x_gauss = grid_gauss(gauss_idx)
        current_weight = gaussian_weights(j)

        ! calculate spline functions for this point in the Gaussian grid
        call quadratic_factors(current_x_gauss, x_left, x_right, h_quad)
        call quadratic_factors_deriv(current_x_gauss, x_left, x_right, dh_quad)
        call cubic_factors(current_x_gauss, x_left, x_right, h_cubic)
        call cubic_factors_deriv(current_x_gauss, x_left, x_right, dh_cubic)

        ! get matrix elements
        call add_bmatrix_terms(gauss_idx, current_weight, quadblock_B, settings)
        call add_regular_matrix_terms(gauss_idx, current_weight, quadblock_A, settings)
        if (settings%physics%flow%is_enabled()) call add_flow_matrix_terms( &
          gauss_idx, current_weight, quadblock_A, settings &
        )
        if (settings%physics%resistivity%is_enabled()) then
          call add_resistive_matrix_terms( &
            gauss_idx, current_weight, quadblock_A, settings &
          )
        end if
        if (settings%physics%cooling%is_enabled()) call add_cooling_matrix_terms( &
          gauss_idx, current_weight, quadblock_A, settings &
        )
        if (settings%physics%conduction%is_enabled()) then
          call add_conduction_matrix_terms( &
            gauss_idx, current_weight, quadblock_A, settings &
          )
        end if
        if (settings%physics%viscosity%is_enabled()) call add_viscosity_matrix_terms( &
          gauss_idx, current_weight, quadblock_A, settings &
        )
        if (settings%physics%hall%is_enabled()) then
          call add_hall_matrix_terms(gauss_idx, current_weight, quadblock_A, settings)
          call add_hall_bmatrix_terms(gauss_idx, current_weight, quadblock_B, settings)
        end if
      end do

      ! dx from integral
      quadblock_B = quadblock_B * (x_right - x_left)
      quadblock_A = quadblock_A * (x_right - x_left)

      ! fill matrices
      !> @note  The quadblock is shifted along the main (tri)diagonal.
      !!        We add `dim_subblock` instead of `dim_quadblock` to the indices,
      !!        due to overlapping of the bottom-right part of the quadblock with the
      !!        top-left part of the next grid interval.
      do k = 1, dim_quadblock
        do l = 1, dim_quadblock
          idx1 = k + quadblock_idx
          idx2 = l + quadblock_idx
          call matrix_B%add_element( &
            row=idx1, column=idx2, element=real(quadblock_B(k, l)) &
          )
          call matrix_A%add_element(row=idx1, column=idx2, element=quadblock_A(k, l))
        end do
      end do
      quadblock_idx = quadblock_idx + settings%dims%get_dim_subblock()
    end do

    deallocate(quadblock_A, quadblock_B)
    call apply_boundary_conditions(matrix_A, matrix_B, settings)
  end subroutine build_matrices

end module mod_matrix_manager
