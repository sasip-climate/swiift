from __future__ import annotations

from collections.abc import Callable
import typing
import warnings

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate

from ._ph_utils import _unit_wavefield

IV = typing.TypeVar(
    "IV", float, np.ndarray[tuple[int], np.dtype[np.float64]]
)  # Integration variable.


def _growth_kernel(x: np.ndarray, mean: np.ndarray, std):
    kern = np.ones((mean.size, x.size))
    mask = np.nonzero(x > mean)
    kern[mask] = np.exp(-((x - mean) ** 2) / (2 * std**2))[mask]
    return kern


def free_surface(
    x,
    wave_params: tuple[np.ndarray, np.ndarray],
    growth_params: tuple[np.ndarray, float] | None,
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
    floe_params: tuple[float, float],
    wave_params: tuple[np.ndarray, np.ndarray],
    growth_params: tuple[np.ndarray, float] | None,
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
    analytical_solution: bool | None,
    length: float,
    growth_params: tuple | None,
    linear_curvature: bool | None = None,
) -> bool:
    """Determine whether to use an analytical solution.

    The displacement, curvature, and elastic energy have analytical expressions
    under certain conditions. These are used if `analytical_solution` is
    explicitely set to `True`. Otherwise, the other parameters are examined to
    determine if the analytical solutions can (and, therefore, should) be used.
    If `growth_params` is not provided, or if all its location values are
    greater than `length`, and if `linear_curvature` is not provided or set to
    `True`, analytical solutions will be used. If `linear_curvature` is set to
    `False`, numerical solutions will be used.

    Parameters
    ----------
    analytical_solution : bool, optional
       Set to `True` to force using analytical solutions.
    length : float
        Length of the floe.
    growth_params : tuple, optional
        Parameters of a wave growth kernel.
    linear_curvature : bool, optional
        Set to `False` to force using numerical approximations to the
        non-linear curvature. It has no effect for other variables
        (displacement, elastic energy) but *does* force using numerical
        solutions instead of analytical solutions, if set to `False`.

    Returns
    -------
    bool

    """
    if analytical_solution is not None:
        return analytical_solution
    if growth_params is None:
        if linear_curvature is None:
            return True
        # No analytical solution for non-linear curvature
        return linear_curvature
    else:
        if linear_curvature is None or linear_curvature:
            # If the wave growth kernel mean is to the right of the floe
            # for every wave component, the wave is fully developed
            # and the analytical solution can be used
            return np.all(growth_params[0] > length)
        return False


def _extract_from_poly(sol: interpolate.PPoly, n: int) -> interpolate.PPoly:
    return interpolate.PPoly(sol.c[:, :, n], sol.x, extrapolate=False)


def _extract_dis_poly(sol: interpolate.PPoly) -> interpolate.PPoly:
    return _extract_from_poly(sol, 0)


def _non_lin_curv(sol: interpolate.PPoly) -> Callable[[IV], np.ndarray]:
    def non_lin_curv(x: IV) -> np.ndarray:
        return (
            _extract_from_poly(sol, 2)(x)
            / (1 + _extract_from_poly(sol, 1)(x) ** 2) ** 1.5
        )

    return non_lin_curv


@typing.overload
def _extract_cur_poly(
    sol: interpolate.PPoly,
    is_linear: typing.Literal[True] = ...,
) -> interpolate.PPoly: ...


@typing.overload
def _extract_cur_poly(
    sol: interpolate.PPoly,
    is_linear: typing.Literal[False] = ...,
) -> Callable[[IV], np.ndarray]: ...


@typing.overload
def _extract_cur_poly(
    sol: interpolate.PPoly, is_linear: bool = ...
) -> interpolate.PPoly | Callable[[IV], np.ndarray]: ...


def _extract_cur_poly(sol: interpolate.PPoly, is_linear: bool = True):
    if is_linear:
        return _extract_from_poly(sol, 2)
    else:
        return _non_lin_curv(sol)


def displacement(x, floe_params, wave_params, growth_params, num_params):
    opt = _get_result(floe_params, wave_params, growth_params, num_params)
    return _extract_dis_poly(opt.sol)(x)


def curvature(
    x, floe_params, wave_params, growth_params, num_params, is_linear: bool = True
):
    opt = _get_result(floe_params, wave_params, growth_params, num_params)
    return _extract_cur_poly(opt.sol, is_linear)(x)


def _prepare_integrand(
    floe_params: tuple[float, float],
    wave_params: tuple[np.ndarray, np.ndarray],
    growth_params,
    num_params,
    linear_curvature: bool,
) -> tuple[Callable[[IV], IV], tuple[float, float]]:
    opt = _get_result(floe_params, wave_params, growth_params, num_params)
    curvature_poly = _extract_cur_poly(opt.sol, linear_curvature)

    def unit_energy(x):
        return curvature_poly(x) ** 2

    return unit_energy, (opt.x[0], opt.x[-1])


def _quad_integration(
    integrand: Callable[[float], float],
    bounds: tuple[float, float],
    **kwargs,
) -> float:
    result = integrate.quad(integrand, *bounds, **kwargs)
    return result[0]


def _tanhsinh_integration(
    integrand: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[float, float],
    **kwargs,
) -> float:
    try:
        result = integrate.tanhsinh(integrand, *bounds, **kwargs)
    except AttributeError:
        warnings.warn(
            "tanhsinh integration was made public in scipy 1.15.0. "
            "Proceeding anyway, but you might want to upgrade if possible, "
            "or use another method."
        )
        result = integrate._tanhsinh.tanhsinh(integrand, *bounds, **kwargs)

    return result.integral


def unit_energy(
    floe_params: tuple[float, float],
    wave_params: tuple[np.ndarray, np.ndarray],
    growth_params,
    num_params,
    integration_method: str | None = None,
    linear_curvature: bool = True,
    **kwargs,
) -> float:
    """Numerically evaluate the energy.

    The energy is up to a prefactor.

    """

    # TODO: actually pass the options to the integrator. For quad, there might
    # be convergence issues if many oscillations wrt the length of the floe. A
    # usual heuristic seems to be fixing `limit` to L / lambda * N with N
    # between 10 to 20. Choosing a big number doesn't hurt computing time, as
    # the integration stops when reaching the desired tolerance anyway.
    if integration_method is None:
        if hasattr(integrate, "tanhsinh"):
            integration_method = "tanhsinh"
        else:
            integration_method = "quad"
    integrand, bounds = _prepare_integrand(
        floe_params,
        wave_params,
        growth_params,
        num_params,
        linear_curvature,
    )

    if integration_method == "quad":
        return _quad_integration(integrand, bounds, **kwargs)
    elif integration_method == "tanhsinh":
        return _tanhsinh_integration(integrand, bounds, **kwargs)
    else:
        raise ValueError("Integration method should be `quad` or `tanhsinh`")
