#!/usr/bin/env python3

from collections.abc import Callable
import pathlib
import numpy as np
from flexfrac1d.lib.displacement import displacement
from flexfrac1d.lib.curvature import curvature
from flexfrac1d.lib.energy import energy

# Test configurations visually examined against solution from scipy.solve_bvp
PATH_DIS = pathlib.Path("tests/target/displacement")
PATH_CUR = pathlib.Path("tests/target/curvature")
PATH_EGY = pathlib.Path("tests/target/energy/")
PATH_PLY = pathlib.Path("tests/target/poly_analytical/")


def format_to_pack(
    red_num, length, wave_params_real
) -> tuple[tuple[float], tuple[np.ndarray]]:
    # format raw floats to output of FloeCoupled._pack
    floe_params = red_num, length
    wave_params = tuple(
        map(
            np.atleast_1d,
            (
                wave_params_real[0],
                wave_params_real[1] + 1j * wave_params_real[2],
                wave_params_real[3],
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

        # test func(x) against existing displacement
        assert np.all(loaded[1] == func(loaded[0], floe_params, wave_params))
    assert sentinel > 0


def test_displacement():
    _test_analytical(PATH_DIS, displacement)


def test_curvature():
    _test_analytical(PATH_CUR, curvature)


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

        _test_poly(dis, displacement, x, floe_params, wave_params)
        _test_poly(cur, curvature, x, floe_params, wave_params)
        _test_poly(egy, energy, floe_params, wave_params)
    assert sentinel > 1


def _test_poly(ref_val, function, *args):
    assert np.all(ref_val == function(*args))


def test_energy():
    loaded = np.loadtxt(PATH_EGY.joinpath("energy_mono.ssv"))
    for vars in loaded.T:
        red_num, length, *wave_params_real = vars[:-1]
        floe_params, wave_params = format_to_pack(red_num, length, wave_params_real)
        assert vars[-1] == energy(floe_params, wave_params)
