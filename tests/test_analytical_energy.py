#!/usr/bin/env python3

from collections.abc import Callable
import pathlib
import numpy as np
from flexfrac1d.lib.displacement import displacement
from flexfrac1d.lib.curvature import curvature

# Test configurations visually examined against solution from scipy.solve_bvp
PATH_DIS = pathlib.Path("tests/target/displacement")
PATH_CUR = pathlib.Path("tests/target/curvature")


def format_to_pack(handle: pathlib.Path) -> tuple[tuple[float], tuple[np.ndarray]]:
    """Format individual input parameters"""
    with open(handle, "r") as file:
        header = file.readline()

    # remove trailing '# ' and split
    _rn, _lg, *_wv = map(float, header[2:-1].split(","))
    # format raw floats to output of FloeCoupled._pack
    _fp = _rn, _lg
    _wp = tuple(map(np.atleast_1d, (_wv[0], _wv[1] + 1j * _wv[2], _wv[3])))
    return _fp, _wp


def _test_analytical(root_dir: pathlib.Path, func: Callable):
    sentinel = 0  # make sure no error in path and at least one test was run
    for handle in root_dir.glob("*ssv"):
        sentinel += 1
        loaded = np.loadtxt(handle)
        # loaded[0]: along-floe space variable x
        # loaded[1]: reference values for func(x)
        floe_params, wave_params = format_to_pack(handle)

        # test func(x) against existing displacement
        assert np.all(loaded[1] == func(loaded[0], floe_params, wave_params))
    assert sentinel > 0


def test_displacement():
    _test_analytical(PATH_DIS, displacement)


def test_curvature():
    _test_analytical(PATH_CUR, curvature)
