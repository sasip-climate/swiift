import numpy as np
from .constants import PI_D4, SQR2
from . import numerical


def _wavefield(x, c_amps, c_wavenumbers):
    return np.imag(c_amps @ _unit_wavefield(x, c_wavenumbers))


def _unit_wavefield(x, c_wavenumbers):
    return np.exp((1j * c_wavenumbers[:, None]) * x)


def _dis_par_amps(red_num: float, wave_params: tuple[np.ndarray]):
    """Complex amplitudes of individual particular solutions"""
    c_amplitudes, c_wavenumbers = wave_params
    return c_amplitudes / (1 + 0.25 * (c_wavenumbers / red_num) ** 4)


def _dis_hom_mat(red_num: float, length: float):
    """Linear application to determine, from the BCs, the coefficients of
    the four independent solutions to the homo ODE"""
    adim = red_num * length
    adim2 = 2 * adim
    denom = (
        -2
        * red_num**2
        * (np.expm1(-adim2) ** 2 + 2 * np.exp(-adim2) * (np.cos(adim2) - 1))
    )
    mat = np.array(
        [
            [
                SQR2 * np.exp(-adim) * np.sin(adim2 + PI_D4) - np.exp(-3 * adim),
                -SQR2 * np.cos(adim + PI_D4)
                - 3 * np.exp(-adim2) * np.sin(adim)
                + np.exp(-adim2) * np.cos(adim),
                np.exp(-adim) * np.sin(adim2) - np.exp(-adim) + np.exp(-3 * adim),
                np.cos(adim)
                - 2 * np.exp(-adim2) * np.sin(adim)
                - np.exp(-adim2) * np.cos(adim),
            ],
            [
                -SQR2 * np.exp(-adim) * np.cos(adim2 + PI_D4)
                + 2 * np.exp(-adim)
                - np.exp(-3 * adim),
                -SQR2 * np.sin(adim + PI_D4)
                + SQR2 * np.exp(-adim2) * np.cos(adim + PI_D4),
                -np.exp(-adim) * np.cos(adim2) + np.exp(-adim),
                np.sin(adim) - np.exp(-adim2) * np.sin(adim),
            ],
            [
                -1 + SQR2 * np.exp(-adim2) * np.cos(adim2 + PI_D4),
                (3 * np.sin(adim) + np.cos(adim)) * np.exp(-adim)
                - SQR2 * np.exp(-3 * adim) * np.sin(adim + PI_D4),
                (np.sin(adim2) + 1) * np.exp(-adim2) - 1,
                (-2 * np.sin(adim) + np.cos(adim)) * np.exp(-adim)
                - np.exp(-3 * adim) * np.cos(adim),
            ],
            [
                (SQR2 * np.sin(adim2 + PI_D4) - 2) * np.exp(-adim2) + 1,
                -SQR2 * np.exp(-adim) * np.sin(adim + PI_D4)
                + SQR2 * np.exp(-3 * adim) * np.cos(adim + PI_D4),
                (-np.cos(adim2) + 1) * np.exp(-adim2),
                np.exp(-adim) * np.sin(adim) - np.exp(-3 * adim) * np.sin(adim),
            ],
        ]
    )
    mat[:, 2:] /= red_num

    return mat / denom


def _dis_hom_rhs(floe_params: tuple[float], wave_params: tuple[np.ndarray]):
    """Vector onto which apply the matrix, to extract the coefficients"""
    red_num, length = floe_params
    _, c_wavenumbers = wave_params
    exp_arg = 1j * c_wavenumbers * length

    r1 = c_wavenumbers**2 * _dis_par_amps(red_num, wave_params)
    r2 = np.imag(r1 @ np.exp(exp_arg))
    r3 = 1j * c_wavenumbers * r1
    r4 = np.imag(r3 @ np.exp(exp_arg))
    r1 = np.imag(r1).sum()
    r3 = np.imag(r3).sum()

    return np.array((r1, r2, r3, r4))


def _dis_hom_coefs(
    floe_params: tuple[float], wave_params: tuple[np.ndarray]
) -> np.ndarray:
    """Coefficients of the four orthogonal homogeneous solutions"""
    return _dis_hom_mat(*floe_params) @ _dis_hom_rhs(floe_params, wave_params)


def _dis_hom(x: np.ndarray, floe_params: tuple[float], wave_params: tuple[np.ndarray]):
    """Homogeneous solution to the displacement ODE"""
    red_num, length = floe_params
    arr = red_num * x
    cosx, sinx = np.cos(arr), np.sin(arr)
    expx = np.exp(-red_num * (length - x))
    exmx = np.exp(-arr)
    return np.vstack(
        ([expx * cosx, expx * sinx, exmx * cosx, exmx * sinx])
    ).T @ _dis_hom_coefs(floe_params, wave_params)


def _dis_par(x: np.ndarray, red_num: float, wave_params: tuple[np.ndarray]):
    """Sum of the particular solutions to the displacement ODE"""
    return _wavefield(x, _dis_par_amps(red_num, wave_params), wave_params[1])


def displacement(
    x,
    floe_params: tuple[float],
    wave_params: tuple[np.ndarray],
    growth_params: tuple | None = None,
    an_sol: bool | None = None,
    num_params: dict | None = None,
):
    if numerical._use_an_sol(an_sol, floe_params[1], growth_params):
        return _dis_hom(x, floe_params, wave_params) + _dis_par(
            x, floe_params[0], wave_params
        )
    return numerical.displacement(
        x, floe_params, wave_params, growth_params, num_params
    )
