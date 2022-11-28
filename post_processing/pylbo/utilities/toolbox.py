import functools
import time

import matplotlib.lines as mpl_lines
import numpy as np
from scipy.interpolate import interp1d
from pylbo._version import _mpl_version
from pylbo.utilities.logger import pylboLogger


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
    T_ef = eigfuncs[0].get("T")
    rho_ef = eigfuncs[0].get("rho")
    Q1_ef = derived_eigfuncs[0].get("B1") # this does actually not play a role in the pressure perturbation
    Q2_ef = derived_eigfuncs[0].get("B2")
    Q3_ef = derived_eigfuncs[0].get("B3")

    B02 = ds.equilibria["B02"]
    B03 = ds.equilibria["B03"]
    B0 = np.sqrt(B02**2 + B03**2)
    rho0 = ds.equilibria["rho0"]
    T0 = ds.equilibria["T0"]
    p0 = rho0*T0

    r_ef = ds.ef_grid
    r = ds.grid_gauss

    vr_ef_highres = interp1d(r_ef, vr_ef)(r)
    v02 = ds.equilibria["v02"]
    v03 = ds.equilibria["v03"]
    v02_lowres = interp1d(r, v02, fill_value="extrapolate")(r_ef)
    v03_lowres = interp1d(r, v03, fill_value="extrapolate")(r_ef)

    chi = r_ef*vr_ef/(v02_lowres*ds.parameters["k2"]*1j + v03_lowres*ds.parameters["k3"]*1j - omega_ef*1j) 

    Q2_highres = interp1d(r_ef, Q2_ef)(r)
    Q3_highres = interp1d(r_ef, Q3_ef)(r)
    T_highres = interp1d(r_ef, T_ef)(r)
    rho_highres = interp1d(r_ef, rho_ef)(r)
    chi_highres = interp1d(r_ef, chi)(r)

    ### Necessary parameters
    m = ds.parameters["k2"]
    k = ds.parameters["k3"]
    gamma = ds.gamma
    Omega = ds.equilibria["v02"]/r
    omegatilde = omega_ef - m*Omega - k*v03

    F = m*B02/r + k*B03
    k_par = F/B0
    omegaAsq = F**2 / rho0
    G = m*B03/r - k*B02
    k_perp = G/B0
    hsq = m**2/r**2 + k**2

    incompressible = (gamma > 1e6)

    # compressible version of the factors:
    if not incompressible:

        omegaSsq = gamma*rho0*T0 / (gamma*rho0*T0 + B02**2+B03**2) * omegaAsq
        
        Anjo = rho0*(omegatilde**2-omegaAsq)
        Snjo = rho0*(gamma*rho0*T0 + B02**2+B03**2)*(omegatilde**2-omegaSsq)
        N = Anjo*Snjo/r
        D = (rho0**2 * omegatilde**4) - hsq*Snjo
    
        P = B02/r * F + rho0*Omega*omegatilde    
        Q = B02/r * (F*P + B02/r*Anjo)
        Lambda = 0 # Keplerian rotation
        C = 2/r**2 * (m*Snjo*P - r**2*rho0*omegatilde**2*(Q - 0.5*Anjo*Lambda)) # some of these terms are zero, but for later use a full implementation might be nice.

        ND = N/D
        CD = C/D

    # incompressible limit of the factors:
    if incompressible:
        
        ND = - rho0*(omegatilde**2-omegaAsq) / (r*hsq)

        CD = - 2*m*(B02*F + rho0*r*Omega*omegatilde) / (r**3*hsq) # this is zero for MRI m=0!!

    chiprime = np.gradient(chi)
    chiprime_highres = interp1d(r_ef, chiprime)(r)

    ksiprime = np.gradient(chi*r_ef)
    ksiprime_highres = interp1d(r_ef, ksiprime)(r)

    Pi = -ND * chiprime_highres - CD * chi_highres

    p_ef = T_highres*rho0 + rho_highres*T0      # p1 = rho0 T1 + rho1 T0

    ### Calculate the ksi vector: 13.93/4
    ksi = chi_highres/r
    if not incompressible:
        eta = ((G*Snjo)*chiprime_highres - 2*((k*gamma*p0*F*P) \
            - r*((P*B03/r)-(G*B02**2/r**2)+(0.5*G*Lambda))*rho0*omegatilde**2)*chi_highres) / (r*B0*D)
        zeta = ((F*gamma*p0*Anjo)*chiprime_highres + 2*((k*gamma*p0*G*P) \
            + r*((P*B02/r)-(F*B02**2/r**2)+(0.5*F*Lambda))*(rho0*omegatilde**2-hsq*B0**2))*chi_highres) / (r*B0*D)
    if incompressible:
        eta = (-k_perp*chiprime_highres + 2*k*k_par*(B02*F + rho0*v02*omegatilde)*chi_highres/(r*rho0*(omegatilde**2-omegaAsq))) / (r*hsq)
        zeta = (-k_par*chiprime_highres - 2*k*k_perp*(B02*F + rho0*v02*omegatilde)*chi_highres/(r*rho0*(omegatilde**2-omegaAsq))) / (r*hsq)

    vecKsi = np.array([ksi,-eta*1j,-zeta*1j])

    ### Calculate div(vec ksi)
    divKsi = chiprime_highres/r + eta*k_perp + zeta*k_par
    divKsiStar = np.conj(chiprime_highres)/r - np.conj(eta)*k_perp - np.conj(zeta)*k_par

    ### Calculate vector Q (but this can also be obtained from Legolas derived eigfunc output... Is rotation the same though?)
    Qr = 1j * F * ksi
    Qtheta = -(np.gradient(B02*ksi) - k*B0*eta)
    Qz = -(np.gradient(r*B03*ksi) + m*B0*eta) / r
    Qperp = (B03*Qtheta - B02*Qz)/B0
    Qpar = (B02*Qtheta + B03*Qz)/B0
    vecQ = np.array([Qr,Qperp,Qpar])

    ### Calculate scalar v/cdot/nabla (because vector v has no r-component)
    v_dot_nabla = (B03*v02 - B02*v03)*k_perp*1j/B0 + (B02*v02 + B03*v03)*k_par*1j/B0

    ### Calculate terms from 13.74/78
    term1 = 0.0
    if not incompressible:
        term1 = term1 + (gamma*p0 * divKsi * divKsiStar)
        term1 = term1 + np.conj(vecKsi[0])*np.gradient(gamma*p0*divKsi) - (np.conj(vecKsi[1])*k_perp+np.conj(vecKsi[2])*k_par)*(gamma*p0*divKsi)

    term2 = F*1j *(np.conj(vecKsi[0])*vecQ[0] + np.conj(vecKsi[1])*vecQ[1] + np.conj(vecKsi[2])*vecQ[2])
    term2 = term2 - np.conj(vecKsi[0])*np.gradient(B02*Qtheta + B03*Qz) + (np.conj(vecKsi[1])*k_perp+np.conj(vecKsi[2])*k_par)*(B02*Qtheta + B03*Qz)
    if not incompressible:
        term2 = term2 - divKsiStar*(B02*Qtheta + B03*Qz) 

    p_along_ksi = vecKsi[0]*np.gradient(p0) + (vecKsi[1]*k_perp+vecKsi[2]*k_par)*(p0)
    term3 = np.conj(vecKsi[0])*np.gradient(p_along_ksi) - (np.conj(vecKsi[1])*k_perp + np.conj(vecKsi[2])*k_par)*(p_along_ksi)
    if not incompressible:
        term3 = term3 + p_along_ksi*divKsiStar

    term4 = - v_dot_nabla**2 * rho0 * (vecKsi[0]*np.conj(vecKsi[0]) + vecKsi[1]*np.conj(vecKsi[1]) + vecKsi[2]*np.conj(vecKsi[2]))
    term4 = term4 - v_dot_nabla * rho0 * (vecKsi[0]*np.conj(vecKsi[0]) + vecKsi[1]*np.conj(vecKsi[1]) + vecKsi[2]*np.conj(vecKsi[2])) * ((B03*v02 - B02*v03)*k_perp*1j/B0 + (B02*v02 + B03*v03)*k_par*1j/B0)
    # actually the same as the previous thing in this equilibrium!

    # print(np.max(np.abs(np.real(term1))), np.max(np.abs(np.imag(term1))))
    # print(np.max(np.abs(np.real(term2))), np.max(np.abs(np.imag(term2))))
    # print(np.max(np.abs(np.real(term3))), np.max(np.abs(np.imag(term3))))
    # print(np.max(np.abs(np.real(term4))), np.max(np.abs(np.imag(term4))))

    ### Calculating the volume integral
    integrand = term1 + term2 + term3 + term4

    w_com = np.trapz(r, integrand)

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