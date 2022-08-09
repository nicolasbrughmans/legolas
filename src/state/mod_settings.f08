module mod_settings
  use mod_logging, only: log_message
  use mod_global_variables, only: str_len_arr
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
    !> dimension of one finite element integral block
    integer, private :: dim_integralblock
    !> dimension of one subblock
    integer, private :: dim_subblock
    !> dimension of one quadblock
    integer, private :: dim_quadblock

    contains

    procedure, private :: set_block_dims
    procedure :: get_dim_subblock
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
    call this%set_block_dims()
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


  !> Sets the block dimensions.
  subroutine set_block_dims(this)
    !> the settings object to set the block dimensions for
    class(settings_t), intent(inout) :: this

    this%dim_integralblock = 2
    this%dim_subblock = this%nb_eqs * this%dim_integralblock
    this%dim_quadblock = this%dim_integralblock * this%dim_subblock
  end subroutine set_block_dims


  !> Returns the dimension of one subblock.
  pure integer function get_dim_subblock(this)
    !> the settings object to get the subblock dimension for
    class(settings_t), intent(in) :: this

    get_dim_subblock = this%dim_subblock
  end function get_dim_subblock


  !> Destructor for a settings object.
  pure subroutine delete(this)
    !> the settings object to delete
    class(settings_t), intent(inout) :: this

    deallocate(this%state_vector)
    deallocate(this%physics_type)
  end subroutine delete

end module mod_settings
