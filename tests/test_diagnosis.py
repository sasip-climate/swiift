import typing

import numpy as np
import pytest

from swiift.model import frac_handlers
from swiift.model.model import WavesUnderFloe
from tests.helpers import growth_params_bool, make_growth_params, setup_wuf, wave_params

strain_handlers = (
    frac_handlers.BinaryStrainFracture,
    frac_handlers.MultipleStrainFracture,
)
resolutions = 0.21, 0.03, 1, 1.14, 5
coefs_nd = 1, 2, 4, 6


@typing.overload
def boiler_plate(
    growth_params_bool: typing.Literal[True],
    resolution: float,
    wave_params: tuple[np.ndarray, np.ndarray],
    fracture_handler: frac_handlers.BinaryFracture,
) -> tuple[frac_handlers._FractureDiag, WavesUnderFloe, tuple]: ...


@typing.overload
def boiler_plate(
    growth_params_bool: None,
    resolution: float,
    wave_params: tuple[np.ndarray, np.ndarray],
    fracture_handler: frac_handlers.BinaryFracture,
) -> tuple[frac_handlers._FractureDiag, WavesUnderFloe, None]: ...


@typing.overload
def boiler_plate(
    growth_params_bool: typing.Literal[True],
    resolution: float,
    wave_params: tuple[np.ndarray, np.ndarray],
    fracture_handler: frac_handlers._StrainFracture,
) -> tuple[frac_handlers._StrainDiag, WavesUnderFloe, tuple]: ...


@typing.overload
def boiler_plate(
    growth_params_bool: None,
    resolution: float,
    wave_params: tuple[np.ndarray, np.ndarray],
    fracture_handler: frac_handlers._StrainFracture,
) -> tuple[frac_handlers._StrainDiag, WavesUnderFloe, None]: ...


def boiler_plate(growth_params_bool, resolution, wave_params, fracture_handler):
    growth_params = make_growth_params(growth_params_bool, wave_params)
    wuf = setup_wuf(wave_params)
    diag = fracture_handler.diagnose(wuf, resolution, growth_params)
    return diag, wuf, growth_params


@pytest.mark.slow
@pytest.mark.parametrize("coef_nd", coefs_nd)
@pytest.mark.parametrize("wave_params", wave_params)
@pytest.mark.parametrize("growth_params_bool", growth_params_bool)
@pytest.mark.parametrize("resolution", resolutions)
def test_energy(coef_nd, wave_params, growth_params_bool, resolution):
    diag, wuf, growth_params = boiler_plate(
        growth_params_bool,
        resolution,
        wave_params,
        frac_handlers.BinaryFracture(coef_nd),
    )
    # for energy diagnostics, the bounds of the plate are excluded, so x[0] := Delta x
    assert diag.x[0] <= resolution
    assert diag.energy.shape == (diag.x.size, 2)
    assert hasattr(diag, "initial_energy") and diag.initial_energy == wuf.energy(
        growth_params
    )
    assert (
        hasattr(diag, "frac_energy_rate")
        and diag.frac_energy_rate == wuf.wui.ice.frac_energy_rate
    )


@pytest.mark.parametrize("handler", strain_handlers)
@pytest.mark.parametrize("coef_nd", coefs_nd)
@pytest.mark.parametrize("wave_params", wave_params)
@pytest.mark.parametrize("growth_params_bool", growth_params_bool)
@pytest.mark.parametrize("resolution", resolutions)
def test_strain(handler, coef_nd, wave_params, growth_params_bool, resolution):
    diag, wuf, _ = boiler_plate(
        growth_params_bool,
        resolution,
        wave_params,
        handler(coef_nd),
    )
    # for strain diagnostics, the bounds of the plate are included, so x[1] := Delta x
    assert diag.x[1] <= resolution
    assert diag.strain.shape == diag.x.shape
    assert hasattr(diag, "peaks")
    assert (
        hasattr(diag, "strain_extrema")
        and diag.peaks.shape == diag.strain_extrema.shape
    )
