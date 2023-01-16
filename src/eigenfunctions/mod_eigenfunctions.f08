module mod_eigenfunctions
  use mod_global_variables, only: dp
  use mod_settings, only: settings_t
  use mod_base_efs, only: base_ef_t
  use mod_derived_efs, only: derived_ef_t
  use mod_derived_ef_names, only: create_and_set_derived_state_vector
  implicit none

  private

  type, public :: eigenfunctions_t
    type(settings_t), pointer, private :: settings
    type(base_ef_t), allocatable :: base_efs(:)
    type(derived_ef_t), allocatable :: derived_efs(:)
    logical, allocatable :: ef_written_flags(:)
    integer, allocatable :: ef_written_idxs(:)
    real(dp), allocatable :: ef_grid(:)

  contains

    procedure, public :: initialise
    procedure, public :: assemble
    procedure, public :: delete

    procedure, private :: select_eigenfunctions_to_save
  end type eigenfunctions_t

  public :: new_eigenfunctions

contains

  function new_eigenfunctions(settings) result(eigenfunctions)
    type(settings_t), target, intent(inout) :: settings
    type(eigenfunctions_t) :: eigenfunctions
    eigenfunctions%settings => settings
  end function new_eigenfunctions


  subroutine initialise(this, omega)
    class(eigenfunctions_t), intent(inout) :: this
    complex(dp), intent(in) :: omega(:)
    character(len=:), allocatable :: state_vector(:)
    character(len=:), allocatable :: derived_state_vector(:)
    integer :: i, nb_efs

    call this%select_eigenfunctions_to_save(omega)
    this%ef_grid = get_ef_grid(this%settings)
    state_vector = this%settings%get_state_vector()
    nb_efs = size(this%ef_written_idxs)

    allocate(this%base_efs(size(state_vector)))
    do i = 1, size(this%base_efs)
      call this%base_efs(i)%initialise( &
        name=state_vector(i), ef_grid_size=size(this%ef_grid), nb_efs=nb_efs &
      )
    end do
    deallocate(state_vector)

    if (.not. this%settings%io%write_derived_eigenfunctions) return

    derived_state_vector = create_and_set_derived_state_vector(this%settings)
    allocate(this%derived_efs(size(derived_state_vector)))
    do i = 1, size(this%derived_efs)
      call this%derived_efs(i)%initialise( &
        name=derived_state_vector(i), &
        ef_grid_size=size(this%ef_grid), &
        nb_efs=nb_efs &
      )
    end do
    deallocate(derived_state_vector)
  end subroutine initialise


  subroutine assemble(this, right_eigenvectors)
    class(eigenfunctions_t), intent(inout) :: this
    complex(dp), intent(in) :: right_eigenvectors(:, :)
    integer :: i
    do i = 1, size(this%base_efs)
      call this%base_efs(i)%assemble( &
        settings=this%settings, &
        idxs_to_assemble=this%ef_written_idxs, &
        right_eigenvectors=right_eigenvectors, &
        ef_grid=this%ef_grid &
      )
    end do
  end subroutine assemble


  pure subroutine delete(this)
    class(eigenfunctions_t), intent(inout) :: this
    integer :: i
    do i = 1, size(this%base_efs)
      call this%base_efs(i)%delete()
    end do
    if (allocated(this%base_efs)) deallocate(this%base_efs)
    if (allocated(this%ef_written_flags)) deallocate(this%ef_written_flags)
    if (allocated(this%ef_written_idxs)) deallocate(this%ef_written_idxs)
    if (allocated(this%ef_grid)) deallocate(this%ef_grid)
  end subroutine delete


  pure subroutine select_eigenfunctions_to_save(this, omega)
    class(eigenfunctions_t), intent(inout) :: this
    complex(dp), intent(in) :: omega(:)
    integer :: i

    allocate(this%ef_written_flags(size(omega)))
    if (this%settings%io%write_ef_subset) then
      this%ef_written_flags = eigenvalue_is_inside_subset_radius( &
        eigenvalue=omega, &
        radius=this%settings%io%ef_subset_radius, &
        center=this%settings%io%ef_subset_center &
      )
    else
      this%ef_written_flags = .true.
    end if
    ! extract indices of those eigenvalues that have their eigenfunctions written
    allocate(this%ef_written_idxs(count(this%ef_written_flags)))
    this%ef_written_idxs = pack([(i, i=1, size(omega))], this%ef_written_flags)
  end subroutine select_eigenfunctions_to_save


  elemental logical function eigenvalue_is_inside_subset_radius( &
    eigenvalue, radius, center &
  )
    complex(dp), intent(in) :: eigenvalue
    real(dp), intent(in) :: radius
    complex(dp), intent(in) :: center
    real(dp) :: distance_from_subset_center

    distance_from_subset_center = sqrt( &
      (aimag(eigenvalue) - aimag(center)) ** 2 &
      + (real(eigenvalue) - real(center)) ** 2 &
    )
    eigenvalue_is_inside_subset_radius = ( &
      distance_from_subset_center <= radius &
    )
  end function eigenvalue_is_inside_subset_radius


  pure function get_ef_grid(settings) result(ef_grid)
    use mod_grid, only: grid
    type(settings_t), intent(in) :: settings
    real(dp), allocatable :: ef_grid(:)
    integer :: grid_idx

    allocate(ef_grid(settings%grid%get_ef_gridpts()))
    ! first gridpoint, left edge
    ef_grid(1) = grid(1)
    ! other gridpoints
    do grid_idx = 1, settings%grid%get_gridpts() - 1
      ! position of center point in grid interval
      ef_grid(2 * grid_idx) = 0.5_dp * (grid(grid_idx) + grid(grid_idx + 1))
      ! position of end point in grid interval
      ef_grid(2 * grid_idx + 1) = grid(grid_idx + 1)
    end do
  end function get_ef_grid

end module mod_eigenfunctions
