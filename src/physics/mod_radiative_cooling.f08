! =============================================================================
!> Module containing radiative cooling-related routines.
!! This module is responsible for initialising the radiative cooling
!! variables and a correct handling of the cooling curves.
!! If an interpolated cooling curve is selected this module calls the
!! interpolation module to create one.
module mod_radiative_cooling
  use mod_global_variables, only: dp
  use mod_logging, only: logger
  use mod_settings, only: settings_t
  use mod_background, only: background_t
  use mod_physics_utils, only: physics_i
  use mod_function_utils, only: from_function
  use mod_cooling_curves, only: interpolate_cooling_curves, get_lambdaT, get_dlambdadT
  implicit none

  private

  type, public :: cooling_t
    procedure(physics_i), pointer, nopass :: lambdaT
    procedure(physics_i), pointer, nopass :: dlambdadT
    procedure(physics_i), pointer, nopass :: L0
    procedure(physics_i), pointer, nopass :: dLdT
    procedure(physics_i), pointer, nopass :: dLdrho
    logical, private :: is_initialised

  contains
    procedure, public :: enable
    procedure, public :: delete
  end type cooling_t

  !> interpolated temperatures from radiative cooling table
  real(dp), allocatable :: interp_table_T(:)
  !> interpolated lambda(T) from radiative cooling table
  real(dp), allocatable :: interp_table_L(:)
  !> interpolated dlambda(T)/dT from radiative cooling table
  real(dp), allocatable :: interp_table_dLdT(:)
  !> boolean to use an interpolated curve or not (if <tt>False</tt>, piecewise is used)
  logical, save         :: interpolated_curve = .true.

  public  :: initialise_radiative_cooling
  public  :: set_radiative_cooling_values
  public  :: radiative_cooling_clean
  public :: new_cooling

contains

  function new_cooling() result(cooling)
    type(cooling_t) :: cooling
    cooling%is_initialised = .false.
    ! do nothing here since it depends on the cooling curves used, which is not
    ! necessarily known when calling the constructor
    cooling%lambdaT => zero
    cooling%dlambdadT => zero
    cooling%L0 => zero
    cooling%dLdT => zero
    cooling%dLdrho => zero
  end function new_cooling


  real(dp) function zero(x, settings, background)
    real(dp), intent(in) :: x
    type(settings_t), intent(in) :: settings
    type(background_t), intent(in) :: background
    zero = 0.0_dp
  end function zero


  subroutine enable(this, settings, background)
    class(cooling_t), intent(inout) :: this
    type(settings_t), intent(in) :: settings
    type(background_t), intent(in) :: background
    logical :: use_interpolated_curve

    use_interpolated_curve = (settings%physics%cooling%get_cooling_curve() /= "rosner")
    if (use_interpolated_curve .and. .not. this%is_initialised) then
      call interpolate_cooling_curves(settings)
      this%is_initialised = .true.
    end if
    this%lambdaT => get_lambdaT
    this%dlambdadT => get_dlambdadT
    this%dLdT => get_dLdT
    this%dLdrho => get_dLdrho
  end subroutine enable


  real(dp) function get_dLdT(x, settings, background)
    real(dp), intent(in) :: x
    type(settings_t), intent(in) :: settings
    type(background_t), intent(in) :: background
    real(dp) :: dlambdadT

    get_dLdT = background%density%rho0(x) * get_dlambdadT(x, settings, background)
  end function get_dLdT


  real(dp) function get_dLdrho(x, settings, background)
    real(dp), intent(in) :: x
    type(settings_t), intent(in) :: settings
    type(background_t), intent(in) :: background

    get_dLdrho = get_lambdaT(x, settings, background)
  end function get_dLdrho


  pure subroutine delete(this)
    class(cooling_t), intent(inout) :: this
    nullify(this%lambdaT)
    nullify(this%dlambdadT)
    nullify(this%L0)
    nullify(this%dLdT)
    nullify(this%dLdrho)
    this%is_initialised = .false.
  end subroutine delete


  !> Initialises the radiative cooling variables.
  !! This routine first selects and allocates the correct cooling
  !! tables, depending on the desired curve. These tables are used
  !! to interpolate the final cooling curve using \p ncool points.
  !! @warning Throws an error if the cooling curve is unknown.
  subroutine initialise_radiative_cooling(settings)
    use mod_cooling_curves

    type(settings_t), intent(in) :: settings
    real(dp), allocatable :: table_T(:), table_L(:)
    integer :: ntable, ncool

    ncool = settings%physics%cooling%get_interpolation_points()
    select case(settings%physics%cooling%get_cooling_curve())
    case("jc_corona")
       ntable = n_jc_corona
       allocate(table_T(ntable))
       allocate(table_L(ntable))
       table_T = t_jc_corona
       table_L = l_jc_corona

    case("dalgarno")
       ntable = n_dalgarno
       allocate(table_T(ntable))
       allocate(table_L(ntable))
       table_T = t_dalgarno
       table_L = l_dalgarno

    case("ml_solar")
       ntable = n_ml_solar
       allocate(table_T(ntable))
       allocate(table_L(ntable))
       table_T = t_ml_solar
       table_L = l_ml_solar

    case("spex")
       ntable = n_spex
       allocate(table_T(ntable))
       allocate(table_L(ntable))
       table_T = t_spex
       table_l = l_spex

    case("spex_dalgarno")
       ntable = n_spex + n_dalgarno2 - 6
       allocate(table_T(ntable))
       allocate(table_L(ntable))
       table_T(1:n_dalgarno2-1) = t_dalgarno2(1:n_dalgarno2-1)
       table_L(1:n_dalgarno2-1) = l_dalgarno2(1:n_dalgarno2-1)
       table_T(n_dalgarno2:ntable) = t_spex(6:n_spex)
       table_L(n_dalgarno2:ntable) = l_spex(6:n_SPEX) + log10(n_spex_enh(6:n_spex))

    case("rosner")
      interpolated_curve = .false.

    case default
      call logger%error( &
        "unknown cooling curve: " // settings%physics%cooling%get_cooling_curve() &
      )
      return
    end select

    if (interpolated_curve) then
      allocate(interp_table_T(ncool))
      allocate(interp_table_L(ncool))
      allocate(interp_table_dLdT(ncool))

      ! Initialise interpolated tables to zero
      interp_table_T    = 0.0d0
      interp_table_L    = 0.0d0
      interp_table_dLdT = 0.0d0

      call create_cooling_curve(settings, table_T, table_L)

      deallocate(table_T)
      deallocate(table_L)
    end if

  end subroutine initialise_radiative_cooling


  !> Sets the radiative cooling attributes of the corresponding types.
  !! This is called _after_ the equilibrium is initialised in the submodule.
  !! @note    No cooling is applied when T0 is below the lower limit of the
  !!          cooling curve. If T0 is above the upper limit of the cooling curve,
  !!          pure Bremmstrahlung is assumed. @endnote
  !! @warning Throws an error if the cooling curve is unknown.
  subroutine set_radiative_cooling_values(settings, background, rc_field)
    use mod_types, only: cooling_type
    use mod_cooling_curves, only: get_rosner_cooling
    use mod_interpolation, only: lookup_table_value
    use mod_grid, only: grid_gauss

    type(settings_t), intent(in) :: settings
    type(background_t), intent(in) :: background
    !> the type containing the radiative cooling attributes
    type(cooling_type), intent(inout)   :: rc_field

    real(dp) :: lambda_T(settings%grid%get_gauss_gridpts())
    real(dp) :: dlambda_dT(settings%grid%get_gauss_gridpts())
    real(dp) :: T0, min_T, max_T
    integer :: i, ncool

    lambda_T = 0.0d0
    dlambda_dT = 0.0d0
    ncool = settings%physics%cooling%get_interpolation_points()

    if (interpolated_curve) then
      min_T = minval(interp_table_T)
      max_T = maxval(interp_table_T)
      do i = 1, settings%grid%get_gauss_gridpts()
        ! current temperature in the grid
        T0 = background%temperature%T0(grid_gauss(i))
        if (T0 <= min_T) then
          ! no cooling if T0 below lower limit cooling curve
          lambda_T(i) = 0.0d0
          dlambda_dT(i) = 0.0d0
        else if (T0 >= max_T) then
          ! assume Bremmstrahlung sqrt(T/Tmax) if T above upper limit cooling curve
          lambda_T(i) = interp_table_L(ncool) * sqrt(T0 / max_T)
          dlambda_dT(i) = 0.5d0 * interp_table_L(ncool) / sqrt(T0 * max_T)
        else
          ! lookup lambda(T0) and dlambda(T0) in interpolated tables
          lambda_T(i) = lookup_table_value(T0, interp_table_T, interp_table_L)
          dlambda_dT(i) = lookup_table_value(T0, interp_table_T, interp_table_dLdT)
        end if
      end do
    else
      ! In this case an analytical cooling curve is used
      select case(settings%physics%cooling%get_cooling_curve())
      case("rosner")
        call get_rosner_cooling( &
          settings=settings, &
          T0=from_function(background%temperature%T0, grid_gauss), &
          lambda_T=lambda_T, &
          dlambda_dT=dlambda_dT &
        )
      end select
    end if

    rc_field%lambdaT = lambda_T
    ! dL/dT = rho0 * d_lambda_dT where lambda_T equals the cooling curve
    rc_field%dL_dT = from_function(background%density%rho0, grid_gauss) * dlambda_dT
    ! dL/drho = lambda(T)
    rc_field%dL_drho = lambda_T

    if (settings%physics%heating%is_enabled()) then
      rc_field%dL_dT = rc_field%dL_dT - rc_field%dH_dT
      rc_field%dL_drho = rc_field%dL_drho - rc_field%dH_drho
    end if
  end subroutine set_radiative_cooling_values


  !> Creates an interpolated cooling curve based on the chosen table.
  !! Calls a second-order polynomial interpolation routine and takes
  !! care of normalisations.
  !! @note    The interpolated cooling curves are normalised on exit.
  subroutine create_cooling_curve(settings, table_T, table_L)
    use mod_interpolation, only: interpolate_table, get_numerical_derivative

    type(settings_t), intent(in) :: settings
    !> temperature values in the cooling table
    real(dp), intent(in)  :: table_T(:)
    !> luminosity values in the cooling table
    real(dp), intent(in)  :: table_L(:)
    real(dp) :: unit_temperature, unit_lambdaT, unit_dlambdaT_dT

    unit_temperature = settings%units%get_unit_temperature()
    unit_lambdaT = settings%units%get_unit_lambdaT()
    unit_dlambdaT_dT = unit_lambdaT / unit_temperature

    ! cooling tables contain dimensionful values on a logarithmic scale.
    ! To avoid resampling the table on an unequally spaced temperature array
    ! (by doing 10**Tvals) we FIRST interpolate the logarithmic table values on an
    ! equally spaced T-array on log-scale.
    ! This will yield log10(L(t)) and log10(T) interpolated (dimensionful) values
    call interpolate_table( &
      settings%physics%cooling%get_interpolation_points(), &
      table_T, &
      table_L, &
      interp_table_T, &
      interp_table_L &
    )
    ! rescale to "actual" L(T) and normalise
    interp_table_L = 10.0d0**interp_table_L / unit_lambdaT
    ! now we normalise T, but taking care that his is actually log(T). So normalising
    ! here means doing log10(T) - log10(Tunit), corresponding to T/Tunit non-log scale
    interp_table_T = interp_table_T - log10(unit_temperature)

    ! calculate dL(T) / dlogT (hence chain rule: dL(T)/dlogT = (dL(T)/dT) * T
    call get_numerical_derivative(interp_table_T, interp_table_L, interp_table_dLdT)
    ! rescale back to actual (already normalised) values
    interp_table_T = 10.0d0**interp_table_T
    ! and hence dL(T)/dT = dL(T) / dlogT * (1 / T)
    interp_table_dLdT = interp_table_dLdT / interp_table_T

    ! LCOV_EXCL_START
    ! save these curves to a file if we're in debug mode
    if (logger%get_logging_level() >= 3) then
      open( &
        unit=1002, &
        file="debug_coolingcurves", &
        access="stream", &
        status="unknown", &
        action="write" &
      )
      write(1002) size(table_T)
      write(1002) 10.0d0**table_T
      write(1002) 10.0d0**table_L
      write(1002) size(interp_table_T)
      write(1002) interp_table_T * unit_temperature
      write(1002) interp_table_L * unit_lambdaT
      write(1002) interp_table_dLdT * unit_dlambdaT_dT
      close(1002)
      call logger%debug("cooling curves saved to file 'debug_coolingcurves'")
    end if
    ! LCOV_EXCL_STOP
  end subroutine create_cooling_curve


  !> Cleanup routine, deallocates all variables allocated at module-scope.
  subroutine radiative_cooling_clean()
    if (interpolated_curve) then
      if (allocated(interp_table_T)) deallocate(interp_table_T)
      if (allocated(interp_table_L)) deallocate(interp_table_L)
      if (allocated(interp_table_dLdT)) deallocate(interp_table_dLdT)
    end if
  end subroutine radiative_cooling_clean

end module mod_radiative_cooling
