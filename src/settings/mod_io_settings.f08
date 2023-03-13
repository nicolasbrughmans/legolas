module mod_io_settings
  use mod_global_variables, only: dp
  implicit none

  private

  type, public :: io_t
    logical, public :: write_matrices
    logical, public :: write_eigenvectors
    logical, public :: write_residuals
    logical, public :: write_eigenfunctions
    logical, public :: write_derived_eigenfunctions
    logical, public :: write_ef_subset
    real(dp), public :: ef_subset_radius
    complex(dp), public :: ef_subset_center
    logical, public :: show_results

    character(:), private, allocatable :: basename_datfile
    character(:), private, allocatable :: output_folder

  contains

    procedure, public :: set_basename_datfile
    procedure, public :: get_basename_datfile
    procedure, public :: set_output_folder
    procedure, public :: get_output_folder
    procedure, public :: should_compute_eigenvectors
    procedure, public :: set_all_io_to_false
    procedure, public :: delete
  end type io_t

  public :: new_io_settings


contains

  pure function new_io_settings() result(io)
    use mod_global_variables, only: NaN

    type(io_t) :: io

    io%write_matrices = .false.
    io%write_eigenvectors = .false.
    io%write_residuals = .false.
    io%write_eigenfunctions = .true.
    io%write_derived_eigenfunctions = .false.
    io%write_ef_subset = .false.
    io%ef_subset_radius = NaN
    io%ef_subset_center = cmplx(NaN, NaN, kind=dp)
    io%show_results = .true.
    call io%set_basename_datfile("datfile")
    call io%set_output_folder("output")
  end function new_io_settings


  pure subroutine set_basename_datfile(this, basename_datfile)
    class(io_t), intent(inout) :: this
    character(len=*), intent(in) :: basename_datfile
    this%basename_datfile = trim(adjustl(basename_datfile))
  end subroutine set_basename_datfile


  pure function get_basename_datfile(this) result(basename_datfile)
    class(io_t), intent(in) :: this
    character(len=:), allocatable :: basename_datfile
    basename_datfile = trim(adjustl(this%basename_datfile))
  end function get_basename_datfile


  pure subroutine set_output_folder(this, output_folder)
    class(io_t), intent(inout) :: this
    character(len=*), intent(in) :: output_folder
    this%output_folder = trim(adjustl(output_folder))
  end subroutine set_output_folder


  pure function get_output_folder(this) result(output_folder)
    class(io_t), intent(in) :: this
    character(len=:), allocatable :: output_folder
    output_folder = trim(adjustl(this%output_folder))
  end function get_output_folder


  pure logical function should_compute_eigenvectors(this)
    class(io_t), intent(in) :: this
    should_compute_eigenvectors = ( &
      this%write_eigenfunctions &
      .or. this%write_eigenvectors &
      .or. this%write_residuals &
    )
  end function should_compute_eigenvectors


  pure subroutine set_all_io_to_false(this)
    class(io_t), intent(inout) :: this
    this%write_matrices = .false.
    this%write_eigenvectors = .false.
    this%write_residuals = .false.
    this%write_eigenfunctions = .false.
    this%write_derived_eigenfunctions = .false.
  end subroutine set_all_io_to_false


  pure subroutine delete(this)
    class(io_t), intent(inout) :: this
    if (allocated(this%basename_datfile)) deallocate(this%basename_datfile)
    if (allocated(this%output_folder)) deallocate(this%output_folder)
  end subroutine delete

end module mod_io_settings
