module mod_settings
  use mod_logging, only: log_message
  use mod_global_variables, only: str_len_arr
  use mod_settings_dims, only: dims_t, new_block_dims
  implicit none

  private

  !> main type containing the program settings
  type, public :: settings_t
    !> array containing the current state vector
    character(len=:), private, allocatable :: state_vector(:)
    !> current physics_type
    character(len=:), private, allocatable :: physics_type
    !> total number of equations
    integer :: nb_eqs
    !> dimensions object
    type(dims_t) :: dims

    contains

    procedure, private :: set_state_vector
    procedure :: get_state_vector
    procedure :: set_physics_type
    procedure :: get_physics_type
    procedure :: delete
  end type settings_t

  public :: new_settings

contains

  !> Constructor for a new settings object.
  function new_settings() result(settings)
    !> the new settings object
    type(settings_t) :: settings

    settings%physics_type = ""
    settings%nb_eqs = 0
  end function new_settings


  !> Sets the physics type, followed by the state vector associated with the
  !! given physics type and the corresponding block dimensions.
  subroutine set_physics_type(this, physics_type)
    !> the settings object to set the physics type for
    class(settings_t), intent(inout) :: this
    !> the physics type to set
    character(len=*), intent(in) :: physics_type

    this%physics_type = physics_type
    call this%set_state_vector(physics_type)
    this%dims = new_block_dims(nb_eqs=this%nb_eqs)
  end subroutine set_physics_type


  !> Returns the physics type of the settings object.
  pure function get_physics_type(this) result(physics_type)
    !> the settings object to get the physics type for
    class(settings_t), intent(in) :: this
    !> the physics type to get
    character(len=len(this%physics_type)) :: physics_type

    physics_type = this%physics_type
  end function get_physics_type


  !> Set the state vector based on the physics type.
  subroutine set_state_vector(this, physics_type)
    !> the settings object to set the state vector for
    class(settings_t), intent(inout) :: this
    !> the physics type to set the state vector for
    character(len=*), intent(in) :: physics_type

    select case(physics_type)
    case ("mhd")
      this%state_vector = [ &
        character(len=str_len_arr) :: "rho", "v1", "v2", "v3", "T", "a1", "a2", "a3" &
      ]
    case ("hydro")
      this%state_vector = [character(len=str_len_arr) :: "rho", "v1", "v2", "v3", "T"]
    case default
      call log_message("Unknown physics_type: " // physics_type, level="error")
    end select
    this%nb_eqs = size(this%state_vector)
  end subroutine set_state_vector


  !> Returns the state vector of the settings object.
  pure function get_state_vector(this) result(state_vector)
    !> the settings object to get the state vector for
    class(settings_t), intent(in) :: this
    !> the state vector to get
    character(len=:), allocatable :: state_vector(:)

    state_vector = this%state_vector
  end function get_state_vector


  !> Destructor for a settings object.
  pure subroutine delete(this)
    !> the settings object to delete
    class(settings_t), intent(inout) :: this

    deallocate(this%state_vector)
    deallocate(this%physics_type)
  end subroutine delete

end module mod_settings
