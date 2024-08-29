from collections.abc import Callable
import pathlib

import numpy as np
import pytest

import flexfrac1d.lib.physics as ph

# TODO: fiture these tests instead of running loops in individual functions

# Test configurations visually examined against solution from scipy.solve_bvp
PATH_DIS = pathlib.Path("tests/target/displacement")
PATH_CUR = pathlib.Path("tests/target/curvature")
PATH_EGY = pathlib.Path("tests/target/energy/")
PATH_PLY = pathlib.Path("tests/target/poly_analytical/")


def format_to_pack(
    red_num, length, wave_params_real
) -> tuple[tuple[float], tuple[np.ndarray]]:
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


def read_header(handle: pathlib.Path):
    with open(handle, "r") as file:
        header = file.readline()
    # remove trailing '# ' and split
    red_num, length, *wave_params_real = map(float, header[2:-1].split(","))
    return red_num, length, wave_params_real


def _test_analytical(root_dir: pathlib.Path, func: Callable):
    sentinel = 0  # make sure no error in path and at least one test was run
    for handle in root_dir.glob("*ssv"):
        sentinel += 1
        loaded = np.loadtxt(handle)
        # loaded[0]: along-floe space variable x
        # loaded[1]: reference values for func(x)
        floe_params, wave_params = format_to_pack(*read_header(handle))
        handler = func(floe_params, wave_params)

        # test func(x) against existing displacement
        assert np.allclose(loaded[1] - handler.compute(loaded[0]), 0)
    assert sentinel > 0


def test_displacement():
    _test_analytical(PATH_DIS, ph.DisplacementHandler)


@pytest.mark.parametrize
def test_displacement_wuf():
    # TODO:
    # * instantiate wui from parameters
    # * instantiate wuf from parameters and wui
    # * call `displacement`
    # * profit
    pass


def test_curvature():
    _test_analytical(PATH_CUR, ph.CurvatureHandler)


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


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_energy_no_attenuation():
    floe_params = 0.34, 126.12
    wave_params = (
        np.array([0.14808142 + 0.34891663j, 0.08581965 + 0.54191726j]),
        np.array([0.02674772 + 0.0j, 0.09422177 + 0.0j]),
    )
    ph.EnergyHandler(floe_params, wave_params).compute()
