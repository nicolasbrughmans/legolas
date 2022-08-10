module mod_settings_dims
  implicit none

  private

  type, public :: dims_t
    !> dimension of one finite element integral block
    integer, private :: dim_integralblock = 0
    !> dimension of one subblock
    integer, private :: dim_subblock = 0
    !> dimension of one quadblock
    integer, private :: dim_quadblock = 0
    !> dimension of the matrices
    integer, private :: dim_matrix = 0

    contains

    procedure :: get_dim_subblock
    procedure :: get_dim_quadblock
    procedure :: set_dim_matrix
    procedure :: get_dim_matrix
  end type dims_t

  public :: new_block_dims

contains

  !> Constructor for a new block dimensions object. Sets the
  !! integralblock, subblock and quadblock dimensions consistent with the
  !! given number of equations.
  pure function new_block_dims(nb_eqs) result(dims)
    !> the number of equations
    integer, intent(in) :: nb_eqs
    !> the new block dimensions object
    type(dims_t) :: dims

    dims%dim_integralblock = 2
    dims%dim_subblock = nb_eqs * dims%dim_integralblock
    dims%dim_quadblock = dims%dim_integralblock * dims%dim_subblock
  end function new_block_dims


  !> Returns the dimension of one subblock.
  pure integer function get_dim_subblock(this)
    !> the settings object to get the subblock dimension for
    class(dims_t), intent(in) :: this

    get_dim_subblock = this%dim_subblock
  end function get_dim_subblock


  !> Returns the dimension of one quadblock.
  pure integer function get_dim_quadblock(this)
    !> the settings object to get the quadblock dimension for
    class(dims_t), intent(in) :: this

    get_dim_quadblock = this%dim_quadblock
  end function get_dim_quadblock


  !> Sets the matrix dimension depending on the number of points in the base grid.
  pure subroutine set_dim_matrix(this, gridpts)
    !> the settings object to set the matrix dimension for
    class(dims_t), intent(inout) :: this
    !> the number of points in the base grid
    integer, intent(in) :: gridpts

    this%dim_matrix = gridpts * this%dim_subblock
  end subroutine set_dim_matrix


  !> Returns the dimension of the matrices.
  pure integer function get_dim_matrix(this)
    !> the settings object to get the matrix dimension for
    class(dims_t), intent(in) :: this

    get_dim_matrix = this%dim_matrix
  end function get_dim_matrix

end module mod_settings_dims
