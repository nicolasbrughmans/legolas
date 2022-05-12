submodule (mod_solvers) smod_arpack_main
  use mod_global_variables, only: arpack_mode, dim_quadblock
  use mod_arpack_type, only: arpack_t, new_arpack_config
  use mod_linear_systems, only: solve_linear_system_complex_banded
  implicit none

  interface
    module subroutine solve_arpack_general(arpack_cfg, matrix_A, matrix_B, omega, vr)
      !> arpack configuration
      type(arpack_t), intent(in) :: arpack_cfg
      !> matrix A
      type(matrix_t), intent(in) :: matrix_A
      !> matrix B
      type(matrix_t), intent(in) :: matrix_B
      !> array with eigenvalues
      complex(dp), intent(out)  :: omega(:)
      !> array with right eigenvectors
      complex(dp), intent(out)  :: vr(:, :)
    end subroutine solve_arpack_general
  end interface

contains


  module procedure arnoldi
#if _ARPACK_FOUND
    use mod_global_variables, only: which_eigenvalues, number_of_eigenvalues, maxiter

    !> type containing parameters for arpack configuration
    type(arpack_t) :: arpack_cfg

    select case(arpack_mode)
    case("general")
      call log_message("Arnoldi iteration, general mode", level="debug")
      arpack_cfg = new_arpack_config( &
        evpdim=matrix_A%matrix_dim, &
        mode=2, &
        bmat="G", &
        which=which_eigenvalues, &
        nev=number_of_eigenvalues, &
        tolerance=1.0d-14, &
        maxiter=maxiter &
      )
      call solve_arpack_general(arpack_cfg, matrix_A, matrix_B, omega, vr)
    case default
      call log_message("unknown mode for ARPACK: " // arpack_mode, level="error")
      return
    end select

    call arpack_cfg%destroy()

#else
  call log_message( &
    "ARPACK was not found and/or CMake failed to link", level="warning" &
  )
  call log_message( &
    "unable to use 'arnoldi', try another solver!", level="error" &
  )
#endif
  end procedure arnoldi

end submodule smod_arpack_main