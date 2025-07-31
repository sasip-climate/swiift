from __future__ import annotations

from collections.abc import Callable
import typing
import warnings

import numpy as np
from scipy._lib._util import _RichResult
import scipy.integrate as integrate
import scipy.interpolate as interpolate

from swiift.lib.constants import PI_2

from ._ph_utils import _unit_wavefield

IV = typing.TypeVar(
    "IV", float, np.ndarray[tuple[int], np.dtype[np.float64]]
)  # Integration variable.

CUBIC_BINOMIAL_COEFS = np.array([0, 0, 0, 1, 1, 2]), np.array([1, 2, 3, 2, 3, 3])


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
    linear_curvature: bool | None,
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
            # and the analytical solution can be used.
            # Alternatively, if there is a single component of the kernel whose
            # mean is to the left of the right edge, the numerical solution
            # must be used.
            return not np.any(growth_params[0] < length)
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


@typing.overload
def _prepare_integrand0(
    floe_params: tuple[float, float],
    wave_params: tuple[np.ndarray, np.ndarray],
    growth_params,
    num_params,
    linear_curvature: typing.Literal[True],
) -> tuple[interpolate.PPoly, tuple[float, float]]: ...


@typing.overload
def _prepare_integrand0(
    floe_params: tuple[float, float],
    wave_params: tuple[np.ndarray, np.ndarray],
    growth_params,
    num_params,
    linear_curvature: typing.Literal[False],
) -> tuple[typing.Callable[[IV], np.ndarray], tuple[float, float]]: ...


@typing.overload
def _prepare_integrand0(
    floe_params: tuple[float, float],
    wave_params: tuple[np.ndarray, np.ndarray],
    growth_params,
    num_params,
    linear_curvature: bool,
) -> tuple[
    interpolate.PPoly | typing.Callable[[IV], np.ndarray], tuple[float, float]
]: ...


def _prepare_integrand0(
    floe_params: tuple[float, float],
    wave_params: tuple[np.ndarray, np.ndarray],
    growth_params,
    num_params,
    linear_curvature: bool,
):
    opt = _get_result(floe_params, wave_params, growth_params, num_params)
    curvature_poly = _extract_cur_poly(opt.sol, linear_curvature)
    bounds = opt.x[0], opt.x[-1]
    return curvature_poly, bounds


def _square_cubic_poly(ppoly: interpolate.PPoly) -> interpolate.PPoly:
    # PPoly object have coefficients ordered opposite wrt to powers. That is,
    # for a cubic, c[0] is the coefficient of the cubic term, c[3] of the
    # constant term.
    new_cs = np.zeros((ppoly.c.shape[0] * 2 - 1, ppoly.c.shape[1]))
    cs = ppoly.c
    idx1, idx2 = CUBIC_BINOMIAL_COEFS
    new_cs[::2, :] = cs**2
    extra_terms = 2 * cs[idx1] * cs[idx2]

    # Need to do them one by one to avoid silent errors, as idx1 + idx2 has
    # values with multiplicity > 1.
    for idx, term in zip(idx1 + idx2, extra_terms):
        new_cs[idx] += term

    return interpolate.PPoly(new_cs, ppoly.x)


def _pseudo_analytical_integration(
    floe_params: tuple[float, float],
    wave_params: tuple[np.ndarray, np.ndarray],
    growth_params: tuple | None,
    num_params: dict,
) -> float:
    # TODO: for now, only works with linear curvature. Could be adapted to
    # nonlinear curvature, it would just require the partial fraction
    # decomposition of the ratio of two 6th order polynomials.
    curvature_poly, bounds = _prepare_integrand0(
        floe_params, wave_params, growth_params, num_params, True
    )
    # TODO: integral could be computed manually, without building the PPoly
    # object first.
    squared_curvature = _square_cubic_poly(curvature_poly)
    bounds = 0, floe_params[1]
    return squared_curvature.integrate(*bounds).item()


def _prepare_integrand(
    floe_params: tuple[float, float],
    wave_params: tuple[np.ndarray, np.ndarray],
    growth_params: tuple | None,
    num_params: dict,
    linear_curvature: bool,
) -> tuple[Callable[[IV], IV], tuple[float, float]]:
    curvature_poly, bounds = _prepare_integrand0(
        floe_params,
        wave_params,
        growth_params,
        num_params,
        linear_curvature,
    )

    def unit_energy(x):
        return curvature_poly(x) ** 2

    return unit_energy, bounds


def _estimate_quad_limit(
    floe_length: float, wave_params: tuple[np.ndarray, np.ndarray]
) -> int:
    # wave_params := complex amplitudes, complex wavenumbers
    # When using `quad`, there might be convergence issues if the integrand
    # observes many oscillations wrt the range of integration, that is the
    # length of the floe. A usual heuristic seems to be fixing `limit` to L /
    # lambda * N with N between 10 to 20. Choosing a big number doesn't hurt
    # computing time, as the integration stops when reaching the desired
    # tolerance anyway.
    factor = 20 / (PI_2)  # high N, scaled by 2pi to get a wavelength
    # We arbitrarily choose the wavenumber associated with the largest spectral
    # component.
    most_significant_wavenumber = np.real(
        wave_params[1][np.abs(wave_params[0]).argmax()]
    )
    # 50 is the default
    return max(
        50, np.ceil(factor * floe_length * most_significant_wavenumber).astype(int)
    )


@typing.overload
def _quad_integration(
    integrand: Callable[[float], float],
    bounds: tuple[float, float],
    limit: int,
    debug: typing.Literal[True],
    **kwargs,
) -> tuple[float, float]: ...


@typing.overload
def _quad_integration(
    integrand: Callable[[float], float],
    bounds: tuple[float, float],
    limit: int,
    debug: typing.Literal[False],
    **kwargs,
) -> float: ...


@typing.overload
def _quad_integration(
    integrand: Callable[[float], float],
    bounds: tuple[float, float],
    limit: int,
    debug: bool,
    **kwargs,
) -> float | tuple[float, float]: ...


def _quad_integration(
    integrand: Callable[[float], float],
    bounds: tuple[float, float],
    limit: int,
    debug: bool,
    **kwargs,
) -> float | tuple[float, float]:
    result = integrate.quad(integrand, *bounds, limit=limit, **kwargs)
    if debug:
        return result
    return result[0]


@typing.overload
def _tanhsinh_integration(
    integrand: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[float, float],
    debug: typing.Literal[True],
    **kwargs,
) -> _RichResult[float]: ...


@typing.overload
def _tanhsinh_integration(
    integrand: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[float, float],
    debug: typing.Literal[False],
    **kwargs,
) -> float: ...


@typing.overload
def _tanhsinh_integration(
    integrand: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[float, float],
    debug: bool,
    **kwargs,
) -> float | _RichResult[float]: ...


def _tanhsinh_integration(
    integrand: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[float, float],
    debug: bool,
    **kwargs,
) -> float | _RichResult[float]:
    default_quad_tol = 1.49e-8
    for key in ("atol", "rtol"):
        if key not in kwargs:
            kwargs[key] = default_quad_tol
    try:
        result = integrate.tanhsinh(integrand, *bounds, **kwargs)
    except AttributeError:
        warnings.warn(
            "tanhsinh integration was made public in scipy 1.15.0. "
            "Proceeding anyway, but you might want to upgrade if possible, "
            "or use another integration method."
        )
        result = integrate._tanhsinh.tanhsinh(integrand, *bounds, **kwargs)
    if debug:
        return result

    return result.integral


# TODO: improve docstring
def unit_energy(
    floe_params: tuple[float, float],
    wave_params: tuple[np.ndarray, np.ndarray],
    growth_params,
    num_params,
    integration_method: str | None = None,
    linear_curvature: bool = True,
    debug: bool = False,
    **kwargs,
) -> float:
    """Numerically evaluate the energy.

    The energy is up to a prefactor.

    """
    if not linear_curvature and integration_method == "pseudo_an":
        warnings.warn(
            f"The method {integration_method} can only be used with linear curvature. "
            "Using tanhsinh instead."
        )
        integration_method = "tanhsinh"

    if integration_method is None:
        if linear_curvature:
            integration_method = "pseudo_an"
        else:
            if hasattr(integrate, "tanhsinh"):
                integration_method = "tanhsinh"
            else:
                integration_method = "quad"

    if integration_method == "pseudo_an":
        return _pseudo_analytical_integration(
            floe_params, wave_params, growth_params, num_params
        )

    integrand, bounds = _prepare_integrand(
        floe_params,
        wave_params,
        growth_params,
        num_params,
        linear_curvature,
    )

    if integration_method == "quad":
        limit = kwargs.pop("limit", None)
        if limit is None:
            limit = _estimate_quad_limit(floe_params[1], wave_params)
        return _quad_integration(integrand, bounds, limit=limit, debug=debug, **kwargs)
    elif integration_method == "tanhsinh":
        return _tanhsinh_integration(integrand, bounds, debug=debug, **kwargs)
    else:
        raise ValueError(
            "Integration method should be `pseudo_an`, `quad`, or `tanhsinh`."
        )
