! =============================================================================
!> This submodule initialises and calculates the base eigenfunctions, i.e. the ones
!! corresponding to the basic state vector variables (rho, v1, v2, etc.)
submodule(mod_eigenfunctions) smod_base_efs
  implicit none

contains

  !> Initialises the base eigenfunction array, sets the corresponding names and state
  !! vector indices, allocates the (subset of) eigenfunctions.
  module procedure initialise_base_eigenfunctions
    integer :: i

    allocate(base_eigenfunctions(size(state_vector)))
    do i = 1, size(base_eigenfunctions)
      base_eigenfunctions(i) % state_vector_index = i
      base_eigenfunctions(i) % name = state_vector(i)
      allocate(base_eigenfunctions(i) % quantities(size(ef_grid), nb_eigenfuncs))
    end do
    efs_initialised = .true.
  end procedure initialise_base_eigenfunctions


  !> Calculates the eigenfunctions corresponding to the requested eigenvalues and
  !! sets them as attributes for the corresponding types.
  module procedure calculate_base_eigenfunctions
    integer :: i, j, eigenvalue_idx
    complex(dp) :: assembled_ef(size(ef_grid))

    do j = 1, size(base_eigenfunctions)
      do i = 1, size(ef_written_idxs)
        eigenvalue_idx = ef_written_idxs(i)
        assembled_ef = assemble_eigenfunction( &
          base_ef=base_eigenfunctions(j), &
          eigenvector=right_eigenvectors(:, eigenvalue_idx), &
          dim_subblock=dims%get_dim_subblock() &
        )
        call retransform_eigenfunction(base_eigenfunctions(j) % name, assembled_ef)
        base_eigenfunctions(j) % quantities(:, i) = assembled_ef
      end do
    end do
  end procedure calculate_base_eigenfunctions

end submodule smod_base_efs
