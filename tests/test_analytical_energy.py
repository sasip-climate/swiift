#!/usr/bin/env python3

import pathlib
import numpy as np
from flexfrac1d.lib.displacement import displacement

# Test configurations visually examined against solution from scipy.solve_bvp
PATH_DIS = pathlib.Path("tests/target/displacement")


def test_displacement():
    sentinel = 0  # make sure no error in path and at least one test was run
    for handle in PATH_DIS.glob("*ssv"):
        sentinel += 1
        loaded = np.loadtxt(handle)
        with open(handle, "r") as file:
            header = file.readline()

        # remove trailing '# ' and split
        _rn, _lg, *_wv = map(float, header[2:-1].split(","))
        # format raw floats to output of FloeCoupled._pack
        _fp = _rn, _lg
        _wp = tuple(map(np.atleast_1d, (_wv[0], _wv[1] + 1j * _wv[2], _wv[3])))

        # test displacement(x) against existing displacement
        assert np.all(loaded[1] == displacement(loaded[0], _fp, _wp))
    assert sentinel > 0
