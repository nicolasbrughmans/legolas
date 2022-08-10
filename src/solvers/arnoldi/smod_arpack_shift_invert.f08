! =============================================================================
!> Module containing the implementation for the ARPACK shift-invert-type solver,
!! that is, given the general eigenvalue problem $$ AX = \omega BX, $$ choose a
!! shift \(\sigma\) and solve the problem $$ (A - \sigma B)X = \omega X, $$ thereby
!! finding \(k\) eigenvalues of the shifted problem that satisfy a given criterion.
submodule (mod_solvers:smod_arpack_main) smod_arpack_shift_invert
  use mod_banded_matrix, only: banded_matrix_t, new_banded_matrix
  use mod_transform_matrix, only: matrix_to_banded
  implicit none

contains

  !> Implementation of the ARPACK shift-invert solver
  module procedure solve_arpack_shift_invert
    !> contains the basis vectors
    complex(dp) :: basis_vectors(arpack_cfg%get_evpdim(), arpack_cfg%get_ncv())
    !> work array of length 3N
    complex(dp) :: workd(3 * arpack_cfg%get_evpdim())
    !> work array
    complex(dp) :: workl(arpack_cfg%get_lworkl())
    !> work array of length ncv
    real(dp) :: rwork(arpack_cfg%get_ncv())
    !> integer array with pointers to mark work array locations
    integer :: ipntr(14)
    !> logical array of dimension ncv, sets which Ritz vectors to compute
    logical :: select_vectors(arpack_cfg%get_ncv())
    !> work array of length 2*ncv
    complex(dp) :: workev(2 * arpack_cfg%get_ncv())

    integer :: diags
    logical :: converged
    type(banded_matrix_t) :: amat_min_sigmab_band
    type(banded_matrix_t) :: bmat_banded
    integer :: xstart, xend, ystart, yend
    complex(dp) :: bxvector(arpack_cfg%get_evpdim())

    call log_message("creating banded A - sigma*B", level="debug")
    diags = dims%get_dim_quadblock() - 1
    !> @note we don't do `matrix_to_banded(matrix_A - matrix_B * sigma)` as it appears
    !! that in rare cases this gives rise to numerical difficulties. Depending on the
    !! equilibrium, for some rather small `sigma` the differences between direct
    !! conversion and operating on `band%AB` are on the order of 1e-8 to 1e-9 for the
    !! imaginary part (as B is of type complex but technically real, all imaginary
    !! components are zero), which seems sufficient to throw off the solver. This can
    !! be mitigated by doing `matrix_to_banded(matrix_A*(1/sigma) - matrix_B)` instead,
    !! followed by multiplying `band%AB` with `sigma`, but then this gives issues
    !! for large sigmas. Operating on the `AB` matrices directly appears to be more
    !! stable, and we ensure that they are compatible. @endnote
    call matrix_to_banded(matrix_A, diags, diags, amat_min_sigmab_band)
    call matrix_to_banded(matrix_B, diags, diags, bmat_banded)

    ! check compatibility
    if (.not. amat_min_sigmab_band%is_compatible_with(bmat_banded)) then
      call log_message( &
        "Arnoldi shift-invert: banded matrices are not compatible!", level="error" &
      )
      call amat_min_sigmab_band%destroy()
      call bmat_banded%destroy()
      return
    end if

    ! form A - sigma*B, we ensured the banded matrices are compatible
    amat_min_sigmab_band%AB = amat_min_sigmab_band%AB - sigma * bmat_banded%AB
    call bmat_banded%destroy()  ! we no longer need this one

    call log_message("doing Arnoldi shift-invert", level="debug")
    converged = .false.
    do while(.not. converged)
      call znaupd( &
        arpack_cfg%ido, &
        arpack_cfg%get_bmat(), &
        arpack_cfg%get_evpdim(), &
        arpack_cfg%get_which(), &
        arpack_cfg%get_nev(), &
        arpack_cfg%get_tolerance(), &
        arpack_cfg%residual, &
        arpack_cfg%get_ncv(), &
        basis_vectors, &
        size(basis_vectors, dim=1), &
        arpack_cfg%iparam, &
        ipntr, &
        workd, &
        workl, &
        arpack_cfg%get_lworkl(), &
        rwork, &
        arpack_cfg%info &
      )

      ! x is given by workd(ipntr(1))
      xstart = ipntr(1)
      xend = xstart + arpack_cfg%get_evpdim() - 1
      ! y is given by workd(ipntr(2))
      ystart = ipntr(2)
      yend = ystart + arpack_cfg%get_evpdim() - 1

      select case(arpack_cfg%ido)
      case(-1, 1)
        ! ido = -1 on first call, forces starting vector in OP range
        ! get y <--- OP * x
        ! we need R = OP*x = inv[A - sigma*B]*B*x
        ! 1. calculate u = B*x
        ! 2. solve linear system [A - sigma*B] * R = u for R
        bxvector = matrix_B * workd(xstart:xend)
        workd(ystart:yend) = solve_linear_system_complex_banded( &
          bandmatrix=amat_min_sigmab_band, vector=bxvector &
        )
      case default
        ! when convergence is achieved or maxiter is reached
        exit
      end select
    end do

    ! check info parameter from znaupd, this errors if necessary
    call arpack_cfg%parse_znaupd_info(converged)
    ! if we have a normal exit, extract the eigenvalues through zneupd
    call zneupd( &
      .true., &  ! always calculate eigenvectors, negligible additional cost in ARPACK
      "A", &  ! calculate Ritz vectors
      select_vectors, &
      omega(1:arpack_cfg%get_nev()), &
      vr(:, 1:arpack_cfg%get_nev()), &
      size(vr, dim=1), &
      sigma, &
      workev, &
      arpack_cfg%get_bmat(), &
      arpack_cfg%get_evpdim(), &
      arpack_cfg%get_which(), &
      arpack_cfg%get_nev(), &
      arpack_cfg%get_tolerance(), &
      arpack_cfg%residual, &
      arpack_cfg%get_ncv(), &
      basis_vectors, &
      size(basis_vectors, dim=1), &
      arpack_cfg%iparam, &
      ipntr, &
      workd, &
      workl, &
      arpack_cfg%get_lworkl(), &
      rwork, &
      arpack_cfg%info &
    )

    call log_message( &
      "performing eigenvalue backtransformation to original problem (nu -> omega)", &
      level="debug" &
    )
    !> @note In applying shift-invert we made the transformation \(C = inv[B]*A\) and
    !! solved the standard eigenvalue problem \(CX = \nu X\) instead since B isn't
    !! always Hermitian (e.g. if we include Hall).
    !! According to the ARPACK documentation, section 3.2.2, this
    !! implies that we must manually transform the eigenvalues \(\nu_j\) from \(C\)
    !! to the eigenvalues \(\omega_j\) from the original system. This uses the relation
    !! $$ \omega_j = \sigma + \frac{1}{\nu_j} $$
    !! @endnote
    omega = sigma + (1.0d0 / omega)

    call arpack_cfg%parse_zneupd_info()
    call arpack_cfg%parse_finished_stats()
  end procedure solve_arpack_shift_invert

end submodule smod_arpack_shift_invert
