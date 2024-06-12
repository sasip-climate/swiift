from numbers import Real
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import warnings
from .displacement import _unit_wavefield


def _growth_kernel(x: np.ndarray, mean: np.ndarray, std):
    kern = np.ones((mean.size, x.size))
    mask = np.nonzero(x > mean)
    breakpoint()
    kern[mask] = np.exp(-((x - mean) ** 2) / (2 * std**2))[mask]
    return kern


def _ode_system(
    x,
    w,
    *,
    floe_params: tuple[float],
    wave_params: tuple[np.ndarray],
    growth_params: tuple[np.ndarray, [np.ndarray, Real]] | None = None
):
    red_num, length = floe_params
    amplitudes, c_wavenumbers, phases = wave_params

    wave_shape = _unit_wavefield(x, c_wavenumbers)
    if growth_params is not None:
        kern = _growth_kernel(x, *growth_params)
        wave_shape *= kern
    eta = np.imag((amplitudes * np.exp(1j * phases)) @ wave_shape)

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
        **kwargs
    )
    if not opt.success:
        warnings.warn("Numerical solution did not converge", stacklevel=2)
    return opt


def _energy(floe_params, wave_params, growth_params, num_params):
    """Numerically evaluate the energy

    The energy is up to a prefactor"""
    # Extract the part of the solution that corresponds to the second derivative,
    # so four terms are not computed where one suffices
    if num_params is None:
        num_params = {}
    opt = _solve_bvp(floe_params, wave_params, growth_params, **num_params)
    curvature_poly = interpolate.PPoly(opt.sol.c[:, :, 2], opt.sol.x, extrapolate=False)

    def unit_energy(x: float) -> float:
        return curvature_poly(x) ** 2

    return integrate.quad(unit_energy, *opt.x[[0, -1]])
