import numpy as np

# from .constants import PI_D4, SQR2
from .displacement import _dis_hom_coefs, _dis_par_amps
from . import numerical

# red_num, length = floe_params
# amplitudes, c_wavenumbers, phases = wave_params


# def _cur_wavefield(self, x, spectrum, complex_amps):
#     """Second derivative of the interface"""
#     return -np.imag(
#         (complex_amps * self.ice._c_wavenumbers**2)
#         @ np.exp(1j * self.ice._c_wavenumbers[:, None] * x)
#     )
def _cur_wavefield(x, red_num: float, wave_params):
    """Second derivative of the interface"""
    _, c_wavenumbers, _ = wave_params

    return -np.imag(
        (_dis_par_amps(red_num, wave_params) * c_wavenumbers**2)
        @ np.exp(1j * c_wavenumbers[:, None] * x)
    )


def _cur_hom(x, floe_params, wave_params):
    """Second derivative of the homogeneous part of the displacement"""
    red_num, length = floe_params
    arr = red_num * x
    cosx, sinx = np.cos(arr), np.sin(arr)
    expx = np.exp(-red_num * (length - x))
    exmx = np.exp(-arr)
    return (
        2
        * red_num**2
        * np.vstack(([-expx * sinx, expx * cosx, exmx * sinx, -exmx * cosx])).T
        @ _dis_hom_coefs(floe_params, wave_params)
    )


# def _cur_par(self, x, spectrum):
#     """Second derivative of the particular part of the displacement"""
#     return self._cur_wavefield(x, spectrum, self._dis_par_amps(spectrum))
def _cur_par(x, red_num, wave_params):
    """Second derivative of the particular part of the displacement"""
    return _cur_wavefield(x, red_num, wave_params)


def curvature(
    x,
    floe_params: tuple[float],
    wave_params: tuple[np.ndarray],
    growth_params: tuple | None = None,
    an_sol: bool | None = None,
    num_params: dict | None = None,
):
    """Curvature of the floe, i.e. second derivative of the vertical displacement"""
    if numerical._use_an_sol(an_sol, floe_params[1], growth_params):
        return _cur_hom(x, floe_params, wave_params) + _cur_par(
            x, floe_params[0], wave_params
        )
    return numerical.curvature(x, floe_params, wave_params, growth_params, num_params)
