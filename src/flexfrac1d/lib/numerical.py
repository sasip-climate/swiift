from numbers import Real
import warnings

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate

from ._ph_utils import _unit_wavefield


def _growth_kernel(x: np.ndarray, mean: np.ndarray, std):
    kern = np.ones((mean.size, x.size))
    mask = np.nonzero(x > mean)
    kern[mask] = np.exp(-((x - mean) ** 2) / (2 * std**2))[mask]
    return kern


def free_surface(
    x,
    wave_params: tuple[np.ndarray],
    growth_params: tuple[np.ndarray, Real] | None,
) -> np.ndarray:
    c_amplitudes, c_wavenumbers = wave_params
    wave_shape = _unit_wavefield(x, c_wavenumbers)
    if growth_params is not None:
        kern = _growth_kernel(np.asarray(x), *growth_params)
        wave_shape *= kern
    eta = np.imag(c_amplitudes @ wave_shape)
    return eta


def _ode_system(
    x,
    w,
    *,
    floe_params: tuple[float],
    wave_params: tuple[np.ndarray],
    growth_params: tuple[np.ndarray, Real] | None,
) -> np.ndarray:
    red_num, _ = floe_params
    eta = free_surface(x, wave_params, growth_params)
    # Factor 4 comes from sqrt(2)**4
    wprime = np.vstack((w[1], w[2], w[3], 4 * red_num**4 * (eta - w[0])))
    return wprime


def _boundary_conditions(wa, wb):
    return np.array((wa[2], wb[2], wa[3], wb[3]))


def _solve_bvp(
    floe_params, wave_params, growth_params, **kwargs
) -> integrate._bvp.BVPResult:
    red_num, length = floe_params
    wavenumber = np.real(wave_params[1])
    n_mesh = max(5, int(length * max(red_num, wavenumber.max())))
    x0 = np.linspace(0, length, n_mesh)
    w0 = np.zeros((4, x0.size))

    opt = integrate.solve_bvp(
        lambda x, w: _ode_system(
            x,
            w,
            floe_params=floe_params,
            wave_params=wave_params,
            growth_params=growth_params,
        ),
        _boundary_conditions,
        x0,
        w0,
        **kwargs,
    )
    return opt


def _get_result(
    floe_params, wave_params, growth_params, num_params
) -> integrate._bvp.BVPResult:
    if num_params is None:
        num_params = dict()
    opt = _solve_bvp(floe_params, wave_params, growth_params, **num_params)
    if not opt.success:
        warnings.warn("Numerical solution did not converge", stacklevel=2)
    return opt


def _use_an_sol(
    an_sol: bool | None, length: float, growth_params: tuple | None
) -> None:
    if an_sol is None:
        if growth_params is None:
            an_sol = True
        else:
            # If the wave growth kernel mean is to the right of the floe
            # for every wave component, the wave is fully developed
            # and the analytical solution can be used
            an_sol = np.all(growth_params[0] > length)
    return an_sol


def _extract_from_poly(sol: interpolate.PPoly, n: int) -> interpolate.PPoly:
    return interpolate.PPoly(sol.c[:, :, n], sol.x, extrapolate=False)


def _extract_dis_poly(sol: interpolate.PPoly) -> interpolate.PPoly:
    return _extract_from_poly(sol, 0)


def _extract_cur_poly(sol: interpolate.PPoly) -> interpolate.PPoly:
    return _extract_from_poly(sol, 2)


def displacement(x, floe_params, wave_params, growth_params, num_params):
    opt = _get_result(floe_params, wave_params, growth_params, num_params)
    return _extract_dis_poly(opt.sol)(x)


def curvature(x, floe_params, wave_params, growth_params, num_params):
    opt = _get_result(floe_params, wave_params, growth_params, num_params)
    return _extract_cur_poly(opt.sol)(x)


def energy(floe_params, wave_params, growth_params, num_params) -> tuple[float]:
    """Numerically evaluate the energy

    The energy is up to a prefactor"""
    opt = _get_result(floe_params, wave_params, growth_params, num_params)
    curvature_poly = _extract_cur_poly(opt.sol)

    def unit_energy(x: float) -> float:
        return curvature_poly(x) ** 2

    return integrate.quad(unit_energy, *opt.x[[0, -1]])
