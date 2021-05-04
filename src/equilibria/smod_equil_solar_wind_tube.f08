! =============================================================================
!> This submodule defines a solar wind flux tube embedded in a uniform magnetic
!! environment. The geometry can be overridden in the parfile, and is
!! cylindrical by default for \( r \in [0, 10] \).
!!
!! This equilibrium is taken from
!! _Zhelyazkov, I. et al (2020). Hall-magnetohydrodynamic waves in flowing ideal
!! incompressible solar-wind plasmas: reconsidered.
!! Astrophys Space Sci 365:29._ [DOI](https://doi.org/10.1007/s10509-020-3741-7).
!! @note For best results, it is recommended to enable mesh accumulation. @endnote
!! @note Default values are given by
!!
!! - <tt>k2</tt> = 0
!! - <tt>k3</tt> = 1
!! - <tt>cte_rho0</tt> = 1 : density value in the tube.
!! - <tt>cte_T0</tt> = 1 : temperature value in the tube.
!! - <tt>cte_B03</tt> = 1 : magnetic field value in the tube.
!! - <tt>cte_v03</tt> = 1 : relative velocity in the tube w.r.t. external matter.
!! - <tt>r0</tt> = 1 : radius of the tube.
!! - <tt>alpha</tt> = 0.679 : ratio of external over internal density.
!! - <tt>beta</tt> = 1.177 : ratio of external over internal magnetic field strength.
!!
!! and can all be changed in the parfile. @endnote
! SUBMODULE: smod_equil_solar_wind_tube
submodule(mod_equilibrium) smod_equil_solar_wind_tube
  implicit none

contains

  module subroutine solar_wind_tube_eq()
    use mod_global_variables, only: flow, hall_mhd, elec_pressure, mesh_accumulation
    use mod_equilibrium_params, only: cte_rho0, cte_T0, cte_B03, cte_v03, r0, &
                                      alpha, beta

    real(dp)  :: r, rho_e, T_e, B_e
    integer   :: i

    call allow_geometry_override(default_geometry='cylindrical', default_x_start=0.0d0, default_x_end=10.0d0)

    if (use_defaults) then
      mesh_accumulation = .true.

      flow = .true.
      hall_mhd = .true.
      elec_pressure = .true.

      cte_rho0 = 1.0d0
      cte_T0 = 1.0d0
      cte_B03 = 1.0d0
      cte_v03 = 1.0d0
      r0 = 1.0d0

      alpha = 0.679
      beta = 1.177

      k2 = 0.0d0
      k3 = 1.0d0
    end if

    call initialise_grid()

    if (r0 > x_end) then
      call log_message("equilibrium: inner cylinder radius r0 > x_end", level='error')
    else if (r0 < x_start) then
      call log_message("equilibrium: inner cylinder radius r0 < x_start", level='error')
    end if

    rho_e = alpha * cte_rho0
    B_e = beta * cte_B03
    T_e = (cte_rho0 * cte_T0 + 0.5d0 * (cte_B03**2 - B_e**2)) / rho_e

    do i = 1, gauss_gridpts
      r = grid_gauss(i)

      if (r > r0) then
        rho_field % rho0(i) = rho_e
        B_field % B03(i) = B_e
        T_field % T0(i) = T_e
        v_field % v03(i) = 0.0d0
      else
        rho_field % rho0(i) = cte_rho0
        B_field % B03(i) = cte_B03
        T_field % T0(i) = cte_T0
        v_field % v03(i) = cte_v03
      end if
      B_field % B0(i) = B_field % B03(i)
    end do

  end subroutine solar_wind_tube_eq

end submodule smod_equil_solar_wind_tube
