! =============================================================================
!> Module to handle parfile reading.
!! Contains subroutines to retrieve the parfile based on the commandline arguments
!! and to read the parfile, setting the global variables.
module mod_input
  use mod_global_variables
  use mod_equilibrium_params
  use mod_logging, only: log_message
  use mod_settings, only: settings_t
  implicit none

  private

  !> IO unit for the parfile
  integer :: unit_par = 101

  public :: read_parfile
  public :: get_parfile

contains


  !> Reads in the supplied parfile and sets the equilibrium parameters and
  !! global variables to their specified values.
  !! @note    The order of the different namelists in the parfile does not matter, this
  !!          is automatically handled. @endnote
  !! @warning Throws an error if:
  !!
  !! - both a density and temperature unit are present in the parfile.
  !! - a <tt>unit_density</tt> is supplied, but no length or magnetic field unit.
  !! - a <tt>unit_temprature</tt> is supplied, but no length or magnetic field unit. @endwarning
  !! @warning If <tt>dry_run</tt> is <tt>True</tt>, this automatically sets eigenfunction
  !!          and matrix saving to <tt>False</tt>, independent of the settings in the parfile! @endwarning
  subroutine read_parfile(parfile, settings)
    use mod_check_values, only: is_equal, is_NaN

    !> the name of the parfile
    character(len=*), intent(in)  :: parfile
    !> the settings object
    type(settings_t), intent(inout) :: settings

    ! physicslist params
    character(str_len) :: physics_type
    real(dp)    :: mhd_gamma
    ! unitlist params
    real(dp)    :: unit_density, unit_temperature, unit_magneticfield, unit_length
    real(dp)    :: mean_molecular_weight
    ! gridlist params
    integer     :: gridpoints
    ! savelist params
    logical :: write_matrices, write_eigenvectors, write_residuals
    logical :: write_eigenfunctions, write_derived_eigenfunctions
    logical :: show_results
    character(len=str_len) :: basename_datfile, output_folder
    ! solvelist params
    character(len=str_len) :: solver, arpack_mode
    character(len=2) :: which_eigenvalues
    integer :: number_of_eigenvalues, maxiter, ncv
    real(dp) :: tolerance
    complex(dp) :: sigma

    namelist /physicslist/  &
        physics_type, mhd_gamma, flow, radiative_cooling, ncool, cooling_curve, &
        external_gravity, thermal_conduction, use_fixed_tc_para, &
        fixed_tc_para_value, use_fixed_tc_perp, fixed_tc_perp_value, &
        resistivity, use_fixed_resistivity, fixed_eta_value, &
        use_eta_dropoff, dropoff_edge_dist, dropoff_width, &
        viscosity, viscous_heating, viscosity_value, incompressible, &
        hall_mhd, hall_substitution, hall_dropoff, &
        elec_inertia, inertia_dropoff, electron_fraction
    namelist /unitslist/    &
        cgs_units, unit_density, unit_temperature, unit_magneticfield, unit_length, &
        mean_molecular_weight
    namelist /gridlist/ &
        geometry, x_start, x_end, gridpoints, force_r0, coaxial
    namelist /equilibriumlist/ &
        equilibrium_type, boundary_type, use_defaults, remove_spurious_eigenvalues, &
        nb_spurious_eigenvalues
    namelist /savelist/ &
        write_matrices, write_eigenvectors, write_residuals, write_eigenfunctions, show_results, &
        basename_datfile, output_folder, logging_level, dry_run, &
        write_derived_eigenfunctions, write_eigenfunction_subset, &
        eigenfunction_subset_center, eigenfunction_subset_radius
    namelist /paramlist/  &
        k2, k3, cte_rho0, cte_T0, cte_B01, cte_B02, cte_B03, cte_v02, cte_v03, &
        cte_p0, p1, p2, p3, p4, p5, p6, p7, p8, &
        alpha, beta, delta, theta, tau, lambda, nu, &
        r0, rc, rj, Bth0, Bz0, V, j0, g, eq_bool
    namelist /solvelist/  &
        solver, arpack_mode, number_of_eigenvalues, which_eigenvalues, maxiter, sigma, &
        ncv, tolerance

    call init_equilibrium_params()
    ! if no parfile supplied flag error
    if (parfile == "") then
      call log_message("no parfile supplied!", level="error")
      return
    end if

    ! physicslist defaults
    physics_type = ""
    mhd_gamma = 0.0d0
    ! gridlist defaults
    gridpoints = 0
    ! unitlist defaults
    unit_density = NaN
    unit_temperature = NaN
    unit_magneticfield = NaN
    unit_length = NaN
    mean_molecular_weight = NaN
    ! savelist defaults
    basename_datfile = "datfile"
    output_folder = "output"
    write_matrices = .false.
    write_eigenvectors = .false.
    write_residuals = .false.
    write_eigenfunctions = .true.
    write_derived_eigenfunctions = .false.
    show_results = .true.
    ! solvelist defaults
    solver = "QR-invert"
    arpack_mode = "standard"
    number_of_eigenvalues = 100
    which_eigenvalues = "LM"
    maxiter = 0
    ncv = 0
    tolerance = DP_limit

    open(unit_par, file=trim(parfile), status='old')
    !! Start reading namelists, rewind so they can appear out of order
          rewind(unit_par)
          read(unit_par, gridlist, end=1001)

    1001  rewind(unit_par)
          read(unit_par, physicslist, end=1002)

    1002  rewind(unit_par)
          read(unit_par, equilibriumlist, end=1003)

    1003  rewind(unit_par)
          read(unit_par, savelist, end=1004)

    1004  rewind(unit_par)
          read(unit_par, paramlist, end=1005)

    1005  rewind(unit_par)
          read(unit_par, unitslist, end=1006)

    1006  rewind(unit_par)
          read(unit_par, solvelist, end=1007)

    1007  close(unit_par)

    ! Set gridpoints and gamma, if supplied
    if (.not. gridpoints == 0) then
      call set_gridpts(gridpoints)
    end if
    if (.not. is_equal(mhd_gamma, 0.0d0)) then
      call set_gamma(mhd_gamma)
    end if

    if (physics_type == "") physics_type = "mhd"
    call settings%initialise(physics_type, gridpoints)

    ! set io settings
    settings%io%write_matrices = write_matrices
    settings%io%write_eigenvectors = write_eigenvectors
    settings%io%write_residuals = write_residuals
    settings%io%write_eigenfunctions = write_eigenfunctions
    settings%io%write_derived_eigenfunctions = write_derived_eigenfunctions
    settings%io%show_results = show_results
    if (dry_run) call settings%io%set_all_io_to_false()
    call settings%io%set_basename_datfile(basename_datfile)
    call settings%io%set_output_folder(output_folder)

    ! set solver settings
    call settings%solvers%set_solver(solver)
    call settings%solvers%set_arpack_mode(arpack_mode)
    settings%solvers%number_of_eigenvalues = number_of_eigenvalues
    settings%solvers%which_eigenvalues = which_eigenvalues
    settings%solvers%maxiter = maxiter
    settings%solvers%ncv = ncv
    settings%solvers%tolerance = tolerance

    call check_and_set_supplied_unit_normalisations( &
      unit_density, &
      unit_temperature, &
      unit_magneticfield, &
      unit_length, &
      mean_molecular_weight &
    )

    ! for an ef subset, check if global subset parameters are properly set
    if (write_eigenfunction_subset) then
      call check_global_eigenfunction_subset_parameters()
    end if
  end subroutine read_parfile


  !> Checks the unit normalisations that are supplied (if any), sets the
  !! unit normalisations if valid.
  subroutine check_and_set_supplied_unit_normalisations( &
    unit_density, &
    unit_temperature, &
    unit_magneticfield, &
    unit_length, &
    mean_molecular_weight &
  )
    use mod_units, only: set_normalisations
    use mod_check_values, only: is_NaN

    !> supplied unit density
    real(dp), intent(in)  :: unit_density
    !> supplied unit temperature
    real(dp), intent(in)  :: unit_temperature
    !> supplied unit magneticfield
    real(dp), intent(in)  :: unit_magneticfield
    !> supplied unit length
    real(dp), intent(in)  :: unit_length
    !> supplied mean molecular weight
    real(dp), intent(in)  :: mean_molecular_weight

    ! Provide normalisations, if supplied
    if (.not. is_NaN(unit_density) .and. .not. is_NaN(unit_temperature)) then
      call log_message( &
        "unit density and unit temperature cannot both be provided in the parfile!", &
        level="error" &
      )
      return
    end if

    ! set normalisations if density is supplied
    if (.not. is_NaN(unit_density)) then
      if (is_NaN(unit_magneticfield) .or. is_NaN(unit_length)) then
        call log_message( &
          "unit_density found, unit_magneticfield and unit_length also required.", &
          level="error" &
        )
        return
      end if
      if (is_NaN(mean_molecular_weight)) then
        call set_normalisations( &
          new_unit_density=unit_density, &
          new_unit_magneticfield=unit_magneticfield, &
          new_unit_length=unit_length &
        )
      else
        call set_normalisations( &
          new_unit_density=unit_density, &
          new_unit_magneticfield=unit_magneticfield, &
          new_unit_length=unit_length, &
          new_mean_molecular_weight=mean_molecular_weight &
        )
      end if
    end if

    ! set normalisations if temperature is supplied
    if (.not. is_NaN(unit_temperature)) then
      if (is_NaN(unit_magneticfield) .or. is_NaN(unit_length)) then
        call log_message( &
          "unit_temperature found, unit_magneticfield and unit_length also required.", &
          level="error" &
        )
        return
      end if
      if (is_NaN(mean_molecular_weight)) then
        call set_normalisations( &
          new_unit_temperature=unit_temperature, &
          new_unit_magneticfield=unit_magneticfield, &
          new_unit_length=unit_length &
        )
      else
        call set_normalisations( &
          new_unit_temperature=unit_temperature, &
          new_unit_magneticfield=unit_magneticfield, &
          new_unit_length=unit_length, &
          new_mean_molecular_weight=mean_molecular_weight &
        )
      end if
    end if
  end subroutine check_and_set_supplied_unit_normalisations


  !> Called when the eigenfunction subset selection is enabled, this checks if the
  !! global variables are properly set.
  subroutine check_global_eigenfunction_subset_parameters()
    use mod_global_variables, only: eigenfunction_subset_center, &
      eigenfunction_subset_radius
    use mod_check_values, only: is_NaN

    if (is_NaN(eigenfunction_subset_center)) then
      call log_message("eigenfunction_subset_center must be set!", level="error")
      return
    end if
    if (is_NaN(eigenfunction_subset_radius)) then
      call log_message("eigenfunction_subset_radius must be set!", level="error")
      return
    end if
  end subroutine check_global_eigenfunction_subset_parameters


  ! LCOV_EXCL_START <this routine is excluded from coverage>
  !> Parses the command line arguments and retrieves the parfile passed.
  !! @warning Throws an error if
  !!
  !! - command line arguments can not be parsed.
  !! - the parfile is not found. @endwarning
  !! @note If no parfile is passed, the code uses a default configuration. @endnote
  subroutine get_parfile(filename_par)
    !> the name of the parfile
    character(len=*), intent(out) :: filename_par

    integer :: num_args, i
    character(len=5*str_len), allocatable :: args(:)
    logical :: file_exists

    num_args = command_argument_count()
    allocate(args(num_args))
    do i = 1, num_args
      call get_command_argument(i, args(i))
    end do

    filename_par = ""

    do i = 1, num_args, 2
      select case(args(i))
      case('-i')
        filename_par = args(i+1)
      case default
        call log_message("unable to read in command line arguments", level='error')
      end select
    end do

    ! If no parfile supplied, flag an error
    if (filename_par == "") then
      call log_message("no parfile supplied, please provide the file path with the -i flag", level="error")
    end if

    !! Check if supplied file exists
    inquire(file=trim(filename_par), exist=file_exists)
    if (.not. file_exists) then
      call log_message(("parfile not found: " // trim(filename_par)), level='error')
    end if
  end subroutine get_parfile
  ! LCOV_EXCL_STOP

end module mod_input
