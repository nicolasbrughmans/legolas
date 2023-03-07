import functools
import time

import matplotlib.lines as mpl_lines
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from pylbo._version import _mpl_version
from pylbo.utilities.logger import pylboLogger
import matplotlib.pyplot as plt


def timethis(func):
    @functools.wraps(func)
    def _time_method(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            pylboLogger.debug(
                f"{func.__name__} took {time.perf_counter() - t0} seconds to execute"
            )

    return _time_method


def get_axis_geometry(ax):
    """
    Retrieves the geometry of a given matplotlib axis.

    Parameters
    ----------
    ax : ~matplotlib.axes.Axes
        The axis to retrieve the geometry from.

    Returns
    -------
    tuple
        The geometry of the given matplotlib axis.
    """
    if _mpl_version >= "3.4":
        axis_geometry = ax.get_subplotspec().get_geometry()[0:3]
    else:
        # this is 1-based indexing by default, use 0-based here for consistency
        # with subplotspec in matplotlib 3.4+
        axis_geometry = transform_to_numpy(ax.get_geometry())
        axis_geometry[-1] -= 1
        axis_geometry = tuple(axis_geometry)
    return axis_geometry


def get_values(array, which_values):
    """
    Determines which values to retrieve from an array.

    Parameters
    ----------
    array : numpy.ndarray
        The array with values.
    which_values : str
        Can be one of the following:

            - "average": returns the average of the array
            - "minimum": returns the minimum of the array
            - "maximum": returns the maximum of the array

        If not supplied or equal to None, simply returns the array.

    Returns
    -------
    array : numpy.ndarray
        Numpy array with values depending on the argument provided.
    """
    if which_values is None:
        return array
    elif which_values == "average":
        return np.average(array)
    elif which_values == "minimum":
        return np.min(array)
    elif which_values == "maximum":
        return np.max(array)
    else:
        raise ValueError(f"unknown argument which_values: {which_values}")


def add_pickradius_to_item(item, pickradius):
    """
    Makes a matplotlib artist pickable and adds a pickradius.
    We have to handle this separately, because for line2D items the method
    :meth:`~matplotlib.axes.Axes.set_picker` is deprecated from version 3.3 onwards.

    Parameters
    ----------
    item : ~matplotlib.artist.Artist
        The artist which will be made pickable
    pickradius : int, float
        Sets the pickradius, which determines if something is "on" the picked point.
    """
    # set_picker is deprecated for line2D from matplotlib 3.3 onwards
    if isinstance(item, mpl_lines.Line2D) and _mpl_version >= "3.3":
        item.set_picker(True)
        item.pickradius = pickradius
    else:
        item.set_picker(pickradius)


def custom_enumerate(iterable, start=0, step=1):
    """
    Does a custom enumeration with a given stepsize.

    Parameters
    ----------
    iterable : ~typing.Iterable
        The iterable to iterate over.
    start : int
        The starting value for enumerate.
    step : int
        The stepsize between enumerate values.

    Yields
    ------
    start : int
        The current index in `iterable`, incremented with `step`.
    itr : ~typing.Iterable
        The corresponding entry of `iterable`.
    """
    for itr in iterable:
        yield start, itr
        start += step


def transform_to_list(obj: any) -> list:
    """
    Transforms a given input argument `obj` to a list. If `obj`
    is a Numpy-array or tuple, a cast to `list()` is invoked.

    Parameters
    ----------
    obj : any
        The object to transform.

    Returns
    -------
    list
        The object converted to a list.
    """
    if obj is None:
        return [obj]
    elif isinstance(obj, (tuple, np.ndarray)):
        return list(obj)
    elif isinstance(obj, list):
        return obj
    return [obj]


def transform_to_numpy(obj: any) -> np.ndarray:
    """
    Transforms a given input argument `obj` to a numpy array.

    Parameters
    ----------
    obj : any
        The object to transform.

    Returns
    -------
    numpy.ndarray
        The object transformed to a numpy array.

    """
    if obj is None:
        return np.asarray([obj])
    elif isinstance(obj, (tuple, list)):
        return np.asarray(obj)
    elif isinstance(obj, np.ndarray):
        return obj
    return np.asarray([obj])


def solve_cubic_exact(a, b, c, d):
    """
    Solves a given cubic polynomial of the form
    :math:`ax^3 + bx^2 + cx + d = 0` using the analytical cubic root formula
    instead of the general `numpy.roots` routine.
    From `StackOverflow <https://math.stackexchange.com/questions
    15865why-not-write-the-solutions-of-a-cubic-this-way/18873#18873/>`_.

    Parameters
    ----------
    a : int, float, complex
        Cubic coefficient.
    b : int, float, complex
        Quadratic coefficient.
    c : int, float, complex
        Linear coefficient.
    d : int, float, complex
        Constant term

    Returns
    -------
    roots : np.ndarray(ndim=3, dtype=complex)
        The three roots of the cubic polynomial as a Numpy array.
    """

    if a == 0:
        raise ValueError("cubic coefficient may not be zero")
    p = b / a
    q = c / a
    r = d / a
    Aterm = (
        -2 * p**3
        + 9 * p * q
        - 27 * r
        + 3
        * np.sqrt(3)
        * np.sqrt(
            -(p**2) * q**2
            + 4 * q**3
            + 4 * p**3 * r
            - 18 * p * q * r
            + 27 * r**2
        )
    ) ** (1 / 3) / (3 * 2 ** (1 / 3))
    Bterm = (-(p**2) + 3 * q) / (9 * Aterm)
    cterm_min = (-1 - np.sqrt(3) * 1j) / 2
    cterm_pos = (-1 + np.sqrt(3) * 1j) / 2
    x1 = -p / 3 + Aterm - Bterm
    x2 = -p / 3 + cterm_min * Aterm - cterm_pos * Bterm
    x3 = -p / 3 + cterm_pos * Aterm - cterm_min * Bterm
    return np.array([x1, x2, x3], dtype=complex)


def count_zeroes(eigfuncs):
    """
    Counts the number of zeroes of an array of complex eigenfunctions by looking at
    sign changes of the real and imaginary part of the eigenfunctions. Doesn't include the grid endpoints 
    in the count, since the boundary conditions are automatically satisfied. This only becomes accurate for 
    eigenfunctions with enough oscillations and is resolution dependent. Therefore, we take the
    minimum of the number of zeroes of the real and imaginary part.

    Parameters
    ----------
    eigfuncs : numpy.ndarray
        Array of eigenfunction arrays of complex numbers.

    Returns
    -------
    nzeroes : np.ndarray(dtype=int)
        Counter array containing the number of zeroes of the real or imaginary part of each input eigenfunction array.
    """

    nzeroes = np.array([], dtype=int)

    for eigfunc in eigfuncs:
        counter_real = 0
        counter_imag = 0
        sign_real_eigfunc = np.sign(np.real(eigfunc))
        sign_imag_eigfunc = np.sign(np.imag(eigfunc))

        for i in range(1,len(sign_real_eigfunc)-1):
            if sign_real_eigfunc[i-1] * sign_real_eigfunc[i] == -1: 
                counter_real += 1
            if sign_real_eigfunc[i-1] * sign_real_eigfunc[i] == 0: 
                if sign_real_eigfunc[i-2] * sign_real_eigfunc[i-1] == 0: 
                    counter_real += 1

            if sign_imag_eigfunc[i-1] * sign_imag_eigfunc[i] == -1: 
                counter_imag += 1
            if sign_imag_eigfunc[i-1] * sign_imag_eigfunc[i] == 0: 
                if sign_imag_eigfunc[i-2] * sign_imag_eigfunc[i-1] == 0: 
                    counter_imag += 1
        
        counter = min(counter_real, counter_imag)
        nzeroes = np.append(nzeroes, counter)

    return nzeroes


def invert_continuum_array(cont, r_gauss, sigma):
    """
    Finds the location of resonance for eigenmode solutions having a real part that might overlap with a continuum range.

    Parameters
    ----------
    cont : numpy.ndarray
        Array containing the range of a specific continuum. Automatically has the same length as r_gauss, 
        since it has the same shape as the equilibrium fields used to calculate the continua. Can be complex,
        but only the resonance with the real part is calculated.
    r_gauss : numpy.ndarray
        Array containing the grid on which equilibrium fields are defined.
    sigma : complex
        An eigenvalue solution of the generalized eigenvalue problem.

    Returns
    -------
    r_inv : None, float
        The location where there is resonance between the eigenmode and the continuum. Returns None if there
        is no resonance with the specified continuum.
    """

    diff = np.sign(np.real(cont) - np.real(sigma))

    if len(np.unique(diff)) < 2:
        # There is no sign change, value is not contained in array.
        return None
    else:
        for i in range(1,len(diff)-1):
            if diff[i]*diff[i-1] < 0:
                # Linear interpolation between the points where the sign change occurs.
                r_inv = (np.real(sigma) - np.real(cont[i-1]))/(np.real(cont[i])-np.real(cont[i-1])) * (r_gauss[i]-r_gauss[i-1]) + r_gauss[i-1]
                return r_inv
            elif diff[i]*diff[i-1] == 0:
                # The exact same value is in the continuum array, return it.
                return r_gauss[i]


def calculate_wcom(ds, index, return_ev=False):
    """
    Add necessary information!!
    Returns 0 if no eigenfunctions are present for given index.
    """

    eigfuncs = ds.get_eigenfunctions(ev_idxs=[index], mute=True)
    if eigfuncs[0] is not None:
        omega_ef = eigfuncs[0].get("eigenvalue")
    else:
        omega_ef = 0.0

    if not ds.derived_efs_written or np.abs(omega_ef) < 1e-15:
        if return_ev:
            return None, omega_ef
        else:
            return None

    derived_eigfuncs = ds.get_derived_eigenfunctions(ev_idxs=[index], mute=True)

    vr_ef = eigfuncs[0].get("v1")
    vtheta_ef = eigfuncs[0].get("v2")
    vz_ef = eigfuncs[0].get("v3")
    T_ef = eigfuncs[0].get("T")
    rho_ef = eigfuncs[0].get("rho")
    Q1_ef = derived_eigfuncs[0].get("B1") # this does actually not play a role in the pressure perturbation
    Q2_ef = derived_eigfuncs[0].get("B2")
    Q3_ef = derived_eigfuncs[0].get("B3")

    r_ef = ds.ef_grid
    r = ds.grid_gauss

    B02 = interp1d(r, ds.equilibria["B02"], fill_value="extrapolate")(r_ef)
    B03 = interp1d(r, ds.equilibria["B03"], fill_value="extrapolate")(r_ef)
    B0 = np.sqrt(B02**2 + B03**2)
    B_total = np.sqrt((B02+Q2_ef)**2+(B03+Q3_ef)**2)
    rho0 = interp1d(r, ds.equilibria["rho0"], fill_value="extrapolate")(r_ef)
    T0 = interp1d(r, ds.equilibria["T0"], fill_value="extrapolate")(r_ef)
    p0 = rho0*T0
    gradp0 = np.gradient(ds.equilibria["rho0"]*ds.equilibria["T0"], r, edge_order=2)
    P_tot = p0 + 0.5*B0**2
    gradP_tot = gradp0 + 0.5*np.gradient(ds.equilibria["B0"]**2, r, edge_order=2)
    gradp0 = interp1d(r, gradp0, fill_value="extrapolate")(r_ef)
    gradP_tot = interp1d(r, gradP_tot, fill_value="extrapolate")(r_ef)

    gradv02 = np.gradient(ds.equilibria["v02"], r, edge_order=2)
    gradv02 = interp1d(r, gradv02, fill_value="extrapolate")(r_ef)
    gradv03 = np.gradient(ds.equilibria["v03"], r, edge_order=2)
    gradv03 = interp1d(r, gradv03, fill_value="extrapolate")(r_ef)

    v02 = interp1d(r, ds.equilibria["v02"], fill_value="extrapolate")(r_ef)
    v03 = interp1d(r, ds.equilibria["v03"], fill_value="extrapolate")(r_ef)
    Phiprime0 = interp1d(r, ds.equilibria['grav'], fill_value="extrapolate")(r_ef)

    ### Necessary parameters
    m = ds.parameters["k2"]
    k = ds.parameters["k3"]
    gamma = ds.gamma
    Omega = v02/r_ef
    Omega0 = m*Omega + k*v03
    omegatilde = omega_ef - Omega0

    F = m*B02/r_ef + k*B03
    k_par = F/B0
    omegaAsq = F**2 / rho0
    G = m*B03/r_ef - k*B02
    k_perp = G/B0
    hsq = m**2/r_ef**2 + k**2

    incompressible = False
    if gamma > 1e6: incompressible = True


    ### The crucial eigenfunctions: 
    chi = r_ef* (-vr_ef / (1j*omegatilde))
    ksi_theta = vr_ef * (gradv02 - Omega)/omegatilde**2 - vtheta_ef / (omegatilde*1j)
    ksi_z = vr_ef * (-gradv03)/omegatilde**2 - vz_ef / (omegatilde*1j)

    # ### The  corrected crucial eigenfunctions:
    # chi = r_ef* (-vr_ef / (1j*omegatilde))
    # ksi_theta = vr_ef * (-gradv02)/omegatilde**2 - vtheta_ef / (omegatilde*1j)
    # ksi_z = vr_ef * (gradv03)/omegatilde**2 - vz_ef / (omegatilde*1j)

    r = r_ef

    ### Calculating the field-based coordinates
    # compressible version of the factors:
    if not incompressible:

        omegaSsq = gamma*rho0*T0 / (gamma*rho0*T0 + B02**2+B03**2) * omegaAsq
        
        Anjo = rho0*(omegatilde**2-omegaAsq)
        Snjo = rho0*(gamma*rho0*T0 + B02**2+B03**2)*(omegatilde**2-omegaSsq)
        N = Anjo*Snjo/r
        D = (rho0**2 * omegatilde**4) - hsq*Snjo
    
        P = B02/r * F + rho0*Omega*omegatilde    
        Q = B02/r * (F*P + B02/r*Anjo)
        Lambda = rho0*(v02**2/r**2 - Phiprime0/r) # Deviation from Keplerian rotation due to Lorentz forces, = gradP_tot/r + B02**2/r**2
        C = 2/r**2 * (m*Snjo*P - r**2*rho0*omegatilde**2*(Q - 0.5*Anjo*Lambda)) # some of these terms are zero, but for later use a full implementation might be nice.

        ND = N/D
        CD = C/D

    # incompressible limit of the factors:
    if incompressible:
        
        ND = - rho0*(omegatilde**2-omegaAsq) / (r*hsq)

        CD = - 2*m*(B02*F + rho0*r*Omega*omegatilde) / (r**3*hsq) # this is zero for MRI m=0!!

    chiprime = np.gradient(chi, r, edge_order=2)

    ### Calculate the ksi vector: 13.93/4
    ksi = chi/r
    if not incompressible:
        eta = ((G*Snjo)*chiprime - 2*((k*gamma*p0*F*P) \
            - r*((P*B03/r)-(G*B02**2/r**2)+(0.5*G*Lambda))*rho0*omegatilde**2)*chi) / (r*B0*D)
        zeta = ((F*gamma*p0*Anjo)*chiprime + 2*((k*gamma*p0*G*P) \
            + r*((P*B02/r)-(F*B02**2/r**2)+(0.5*F*Lambda))*(rho0*omegatilde**2-hsq*B0**2))*chi) / (r*B0*D)
    if incompressible:
        eta = (-k_perp*chiprime + 2*k*k_par*(B02*F + rho0*v02*omegatilde)*chi/(r*rho0*(omegatilde**2-omegaAsq))) / (r*hsq)
        zeta = (-k_par*chiprime - 2*k*k_perp*(B02*F + rho0*v02*omegatilde)*chi/(r*rho0*(omegatilde**2-omegaAsq))) / (r*hsq)

    ### Alternative definition of eta/zeta:
    eta_easy = 1j * ((B03)*ksi_theta - (B02)*ksi_z) / B0
    zeta_easy = 1j * ((B02)*ksi_theta + (B03)*ksi_z) / B0

    eta_easy = eta_easy #/ np.real(eta_easy[-10]) * np.real(eta[-10])
    zeta_easy = zeta_easy #/ np.real(zeta_easy[-75]) * np.real(zeta[-75])

    # eta = eta_easy
    # zeta = zeta_easy

    # # Scaling and rotating result based on these values
    # r_one = np.abs(eta[2])
    # r_easy = np.abs(eta_easy[2])
    # theta_one = np.arctan(np.imag(eta[2])/np.real(eta[2]))
    # print(theta_one)
    # if np.sign(eta[2].real) < 0.0: theta_one = theta_one + np.pi
    # theta_easy = np.arctan(np.imag(eta_easy[2])/np.real(eta_easy[2]))
    # if np.sign(eta_easy[2].real) < 0.0: theta_easy = theta_easy + np.pi
    # r_scale = r_one/r_easy
    # theta_scale = theta_one - theta_easy
    # rotation = r_scale * np.exp(theta_scale*1j)

    # eta_easy = eta_easy*rotation
    # zeta_easy = zeta_easy*rotation

    # plt.figure()
    # plt.plot(r, np.real(eta), color='C0')
    # plt.plot(r, np.real(eta_easy), color='C0', linestyle='dotted')
    # plt.plot(r, np.real(zeta), color='C1')
    # plt.plot(r, np.real(zeta_easy), color='C1', linestyle='dotted')
    # plt.title('Zeta (real part)')
    # # plt.ylim([-8e-7,8e-7])
    # plt.figure()
    # plt.plot(r, np.imag(eta), color='C0')
    # plt.plot(r, np.imag(eta_easy), color='C0', linestyle='dotted')
    # plt.plot(r, np.imag(zeta), color='C1')
    # plt.plot(r, np.imag(zeta_easy), color='C1', linestyle='dotted')
    # plt.title('Zeta (imag part)')
    # # plt.ylim([-8e-6,8e-6])
    # plt.show()

    vecKsi = np.array([ksi,-eta*1j,-zeta*1j])

    ### Calculate div(vec ksi)
    divKsi = chiprime/r + eta*k_perp + zeta*k_par
    divKsiStar = np.conj(chiprime)/r - np.conj(eta)*k_perp - np.conj(zeta)*k_par

    ### Calculate vector Q (but this can also be obtained from Legolas derived eigfunc output... Is rotation the same though?)
    Qr = 1j * F * ksi
    Qtheta = -(np.gradient(B02*ksi, r, edge_order=2) - k*B0*eta)
    Qz = -(np.gradient(r*B03*ksi, r, edge_order=2) + m*B0*eta) / r
    Qperp = (B03*Qtheta - B02*Qz)/B0
    Qpar = (B02*Qtheta + B03*Qz)/B0
    vecQ = np.array([Qr,Qperp,Qpar])

    ### Calculate scalar v/cdot/nabla (because vector v has no r-component)
    v_dot_nabla = (B03*v02 - B02*v03)*k_perp*1j/B0 + (B02*v02 + B03*v03)*k_par*1j/B0

    ### Calculate terms from 13.74/78
    term1 = np.zeros_like(divKsi)
    if not incompressible:
        term1 = term1 + (gamma*p0 * divKsi * divKsiStar)
        term1 = term1 + np.conj(vecKsi[0])*gamma*(gradp0*divKsi + p0*np.gradient(divKsi, r, edge_order=2)) - (np.conj(vecKsi[1])*k_perp+np.conj(vecKsi[2])*k_par)*(gamma*p0*divKsi)

    term2 = F*1j *(np.conj(vecKsi[0])*vecQ[0] + np.conj(vecKsi[1])*vecQ[1] + np.conj(vecKsi[2])*vecQ[2])
    term2 = term2 - np.conj(vecKsi[0])*np.gradient(B02*Qtheta + B03*Qz, r, edge_order=2) + (np.conj(vecKsi[1])*k_perp+np.conj(vecKsi[2])*k_par)*(B02*Qtheta + B03*Qz)
    if not incompressible:
        term2 = term2 - divKsiStar*(B02*Qtheta + B03*Qz) 

    p_along_ksi = vecKsi[0]*gradp0 + (vecKsi[1]*k_perp+vecKsi[2]*k_par)*(p0)
    term3 = np.conj(vecKsi[0])*np.gradient(p_along_ksi, r, edge_order=2) - (np.conj(vecKsi[1])*k_perp + np.conj(vecKsi[2])*k_par)*(p_along_ksi)
    if not incompressible:
        term3 = term3 + p_along_ksi*divKsiStar

    term4 = - v_dot_nabla**2 * rho0 * (vecKsi[0]*np.conj(vecKsi[0]) + vecKsi[1]*np.conj(vecKsi[1]) + vecKsi[2]*np.conj(vecKsi[2]))
    term4 = term4 - v_dot_nabla * rho0 * (vecKsi[0]*np.conj(vecKsi[0]) + vecKsi[1]*np.conj(vecKsi[1]) + vecKsi[2]*np.conj(vecKsi[2])) * ((B03*v02 - B02*v03)*k_perp*1j/B0 + (B02*v02 + B03*v03)*k_par*1j/B0)
    # actually the same as the previous thing in this equilibrium!

    # for term,title in zip([term1,term2,term3,term4,np.gradient(B02*Qtheta + B03*Qz, edge_order=2)], ["term1","term2","term3","term4","np.gradient(B02*Qtheta + B03*Qz, r, edge_order=2)"]):
    #     plt.figure()
    #     plt.plot(r, np.real(term))
    #     plt.title(title)
    # plt.show()

    # print(np.max(np.abs(np.real(term1))), np.max(np.abs(np.imag(term1))))
    # print(np.max(np.abs(np.real(term2))), np.max(np.abs(np.imag(term2))))
    # print(np.max(np.abs(np.real(term3))), np.max(np.abs(np.imag(term3))))
    # print(np.max(np.abs(np.real(term4))), np.max(np.abs(np.imag(term4))))

    ### Calculating the volume integral
    integrand = term1 + term2 + term3 + term4

    w_com = simps(integrand, r)

    if np.abs(w_com) < 1e3:
        if return_ev:
            return (w_com, omega_ef)
        else:
            return w_com
    else:
        if return_ev:
            return (None, omega_ef)
        else:
            return None