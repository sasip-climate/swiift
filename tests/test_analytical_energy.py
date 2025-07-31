import pathlib
import typing

import numpy as np
import pytest

import swiift.lib.physics as ph

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
    benchmark,
):
    x = X_AXES[i]
    handler = _init_handler(i, handler_type, False)
    computed = benchmark(handler.compute, x, an_sol)
    if an_sol:
        assert np.allclose(computed, target[0, i])
    else:
        assert np.allclose(computed, target[1, i])


@pytest.mark.parametrize("test_case_idx", range(len(X_AXES)))
@pytest.mark.parametrize("an_sol", (True, False))
@pytest.mark.benchmark(group="Displacement mono: ")
def test_displacement_mono(benchmark, test_case_idx: int, an_sol: bool):
    benchmark.group += f"case: {test_case_idx:02d}"
    _test_mono(
        test_case_idx, ph.DisplacementHandler, an_sol, DISPLACEMENTS_MONO, benchmark
    )


@pytest.mark.parametrize("test_case_idx", range(len(X_AXES)))
@pytest.mark.parametrize("an_sol", (True, False))
@pytest.mark.benchmark(group="Curvature mono: ")
def test_curvature_mono(benchmark, test_case_idx: int, an_sol: bool):
    benchmark.group += f"case: {test_case_idx:02d}"
    _test_mono(test_case_idx, ph.CurvatureHandler, an_sol, CURVATURES_MONO, benchmark)


@pytest.mark.parametrize("test_case_idx", range(len(X_AXES)))
@pytest.mark.parametrize(
    "integration_method",
    (
        None,
        "pseudo_an",
        "tanhsinh",
        "quad",
    ),
)
@pytest.mark.benchmark(group="Energy mono: ")
def test_energy_mono(benchmark, test_case_idx: int, integration_method: str | None):
    benchmark.group += f"case: {test_case_idx:02d}"
    i = test_case_idx
    handler: ph.EnergyHandler = _init_handler(
        test_case_idx,
        ph.EnergyHandler,
        False,
    )

    if integration_method is None:
        an_sol = True
    else:
        an_sol = False
    computed = benchmark(
        handler.compute, an_sol=an_sol, integration_method=integration_method
    )
    if an_sol:
        assert np.allclose(computed, ENERGIES_MONO[0, i])
    else:
        if integration_method == "pseudo_an":
            assert np.allclose(computed, ENERGIES_MONO[1, i])
        elif integration_method == "tanhsinh":
            assert np.allclose(computed, ENERGIES_MONO[2, i])
        elif integration_method == "quad":
            assert np.allclose(computed, ENERGIES_MONO[3, i])
        else:
            print(integration_method)
            raise ValueError(f"Invalid integration method: {integration_method}.")


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
