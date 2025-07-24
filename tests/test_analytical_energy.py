import pathlib
import typing

import numpy as np
import pytest

import swiift.lib.physics as ph

# TODO: fiture these tests instead of running loops in individual functions

# Test configurations visually examined against solution from scipy.solve_bvp
PATH_EGY = pathlib.Path("tests/target/energy/")
PATH_PLY = pathlib.Path("tests/target/poly_analytical/")

TARGET_DIR = pathlib.Path("tests/target/physics_monochromatic")
X_AXES = np.load(TARGET_DIR.joinpath("x.npy"))
FLOE_PARAMS = np.load(TARGET_DIR.joinpath("floe_params.npy"))
C_AMPLITUDES = np.load(TARGET_DIR.joinpath("c_amplitudes.npy"))
C_WAVENUMBERS = np.load(TARGET_DIR.joinpath("c_wavenumbers.npy"))
DISPLACEMENTS_MONO = np.load(TARGET_DIR.joinpath("displacement_mono.npy"))
CURVATURES_MONO = np.load(TARGET_DIR.joinpath("curvature_mono.npy"))
ENERGIES_MONO = np.load(TARGET_DIR.joinpath("energy_mono.npy"))

T = typing.TypeVar("T", ph.DisplacementHandler, ph.CurvatureHandler, ph.EnergyHandler)


def _init_handler(i: int, handler_type: type[T], has_growth_params: bool = False) -> T:
    floe_params = FLOE_PARAMS[i]
    wave_params = (C_AMPLITUDES[i], C_WAVENUMBERS[i])
    if not has_growth_params:
        growth_params = None
    handler = handler_type(floe_params, wave_params, growth_params)
    return handler


def _test_mono(
    i: int,
    handler_type: type[ph.DisplacementHandler] | type[ph.CurvatureHandler],
    an_sol: bool,
    target: np.ndarray,
):
    x = X_AXES[i]
    handler = _init_handler(i, handler_type, False)
    computed = handler.compute(x, an_sol)
    if an_sol:
        assert np.allclose(computed, target[0, i])
    else:
        assert np.allclose(computed, target[1, i])


@pytest.mark.parametrize("test_case_idx", range(len(X_AXES)))
@pytest.mark.parametrize("an_sol", (True, False))
def test_displacement_mono(test_case_idx: int, an_sol: bool):
    _test_mono(test_case_idx, ph.DisplacementHandler, an_sol, DISPLACEMENTS_MONO)


@pytest.mark.parametrize("test_case_idx", range(len(X_AXES)))
@pytest.mark.parametrize("an_sol", (True, False))
def test_curvature_mono(test_case_idx: int, an_sol: bool):
    _test_mono(test_case_idx, ph.CurvatureHandler, an_sol, CURVATURES_MONO)


@pytest.mark.parametrize("test_case_idx", range(len(X_AXES)))
@pytest.mark.parametrize("params", ((True, None), (False, "tanhsinh"), (False, "quad")))
def test_energy_mono(test_case_idx: int, params: tuple[bool, str | None]):
    i = test_case_idx
    handler = _init_handler(test_case_idx, ph.EnergyHandler, False)

    an_sol, integration_method = params
    computed = handler.compute(an_sol=an_sol, integration_method=integration_method)
    if an_sol:
        assert np.allclose(computed, ENERGIES_MONO[0, i])
    else:
        if integration_method == "tanhsinh":
            assert np.allclose(computed, ENERGIES_MONO[1, i])
        elif integration_method == "quad":
            assert np.allclose(computed, ENERGIES_MONO[2, i])
        else:
            raise ValueError


def format_to_pack(
    red_num, length, wave_params_real
) -> tuple[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    # format raw floats to fields of Handlers
    floe_params = red_num, length
    wave_params = tuple(
        map(
            np.atleast_1d,
            (
                wave_params_real[0] * np.exp(1j * wave_params_real[3]),
                wave_params_real[1] + 1j * wave_params_real[2],
            ),
        )
    )
    return floe_params, wave_params


def test_dce_poly():
    sentinel = 0
    for handle in PATH_PLY.glob("*"):
        sentinel += 1
        loaded = np.loadtxt(handle.joinpath("values.ssv"))
        x, dis, cur = loaded
        egy = float(np.loadtxt(handle.joinpath("energy")))
        floe_params = np.loadtxt(handle.joinpath("floe_params.ssv"))
        wave_params_real = np.loadtxt(handle.joinpath("wave_params.ssv"))
        assert len(wave_params_real.shape) == 2
        floe_params, wave_params = format_to_pack(*floe_params, wave_params_real)
        displacement = ph.DisplacementHandler(floe_params, wave_params).compute
        curvature = ph.CurvatureHandler(floe_params, wave_params).compute
        energy = ph.EnergyHandler(floe_params, wave_params).compute

        _test_poly(dis, displacement, x, floe_params, wave_params)
        _test_poly(cur, curvature, x, floe_params, wave_params)
        _test_poly(egy, energy, floe_params, wave_params)
    assert sentinel > 1


def _test_poly(ref_val, function, *args):
    assert np.allclose(ref_val - function(*args), 0)


def test_energy():
    loaded = np.loadtxt(PATH_EGY.joinpath("energy_mono.ssv"))
    for vars in loaded.T:
        red_num, length, *wave_params_real = vars[:-1]
        floe_params, wave_params = format_to_pack(red_num, length, wave_params_real)
        handler = ph.EnergyHandler(floe_params, wave_params)
        assert np.isclose(vars[-1] - handler.compute(), 0)
