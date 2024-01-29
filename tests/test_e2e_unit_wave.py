#!/usr/bin/env python3

"""Basic value tests on the attributes and methods of a Wave instance"""

import itertools
import numpy as np
import polars as pl
import polars.testing as pltesting
import pytest

from flexfrac1d.wave import Wave
from flexfrac1d.ice import Floe


def sub_dict(dct, keys):
    return {k: dct[k] for k in keys}


def gen_method_arguments():
    """Return a dict of arguments needed for the methods"""
    method_arguments = {"times": (1, 1.1, 2003.),
                        "xs": list(map(np.atleast_1d,
                                       (1, 1.1,
                                        np.array((0, 1, 2)),
                                        np.array((0., 1.))))),
                        "a0s": (.5, 1),
                        "lengths": (50, 65.1),
                        "x0s": (-10, 0, 10, 10.1),
                        "thicknesses": (.9, 2)}

    floes1 = [(Floe(_h, _x0, _length),)
              for _h, _x0, _length
              in itertools.product(method_arguments["thicknesses"],
                                   method_arguments["x0s"],
                                   method_arguments["lengths"])]
    floes2 = [_floe + (Floe(1.1,
                            (max(method_arguments["x0s"])
                             + max(method_arguments["lengths"]) + 1),
                            100.),)
              for _floe in floes1]
    method_arguments["floes"] = floes1 + floes2

    return method_arguments


SRC_TARGET = "gen_end_to_end/wave"


method_arguments = gen_method_arguments()
# Associate the arguments with the existing methods
methods = {'amp': sub_dict(method_arguments, ("times",)),
           'amp_att': sub_dict(method_arguments, ("xs", "a0s", "floes")),
           'calc_phase': sub_dict(method_arguments, ("xs", "times")),
           'mslf': sub_dict(method_arguments, ("x0s", "lengths", "times")),
           'waves': sub_dict(method_arguments, ("xs", "times"))}

constructor_arguments = {"amplitudes": (1, 1.1),
                         "wavelengths": (40, 45.2),
                         "betas": (0, 1, None),
                         "phis": (0, 1, 1.1, 2*np.pi+.1, None)}
constructor_combinations = list(itertools
                                .product(*constructor_arguments.values()))

old_to_new_map = {
    "type": "type",
    "n0": "amplitude",
    "E0": "energy",
    "wl": "wavelength",
    "k": "wavenumber",
    "omega": "ang_frequency",
    "T": "period",
    "beta": "beta",
    "phi": "phase",
}
new_to_old_map = {v: k for k, v in old_to_new_map.items()}


def test_attributes():
    attributes = ['type', 'n0', 'E0', 'wl', 'k', 'omega', 'T', 'beta', 'phi']

    n_waves = len(constructor_combinations)
    df_dict = dict(zip(constructor_arguments.keys(),
                       zip(*constructor_combinations)))
    df_dict['type'] = n_waves*[None]
    df_dict |= {att: np.full(n_waves, np.nan)
                for att in attributes if att != "type"}

    for i, (amp, wl, beta, phi) in enumerate(constructor_combinations):
        kwargs = {}
        if beta is not None:
            kwargs["beta"] = beta
        if phi is not None:
            kwargs["phase"] = phi
        wave = Wave(amp, wl, **kwargs)
        for att in attributes:
            df_dict[att][i] = getattr(wave, old_to_new_map[att])
    df = pl.from_dict(df_dict)

    df_src = pl.read_parquet(f"{SRC_TARGET}/attributes_reference.parquet")
    return pltesting.assert_frame_equal(df_src, df)


@pytest.mark.parametrize("method, args", methods.items())
def test_methods(method, args):
    n_waves = len(constructor_combinations)

    argument_combinations = list(itertools.product(*args.values()))
    n_met_args = len(argument_combinations)
    df_keys = sum(map(lambda _d: tuple(_d.keys()),
                      (constructor_arguments, args)), tuple()) + (method,)
    df_dict = {k: [None]*n_waves*len(argument_combinations)
               for k in df_keys}

    for i, cstr_args in enumerate(constructor_combinations):
        amp, wl, beta, phi = cstr_args
        kwargs = {}
        if beta is not None:
            kwargs["beta"] = beta
        if phi is not None:
            kwargs["phase"] = phi
        wave = Wave(amp, wl, **kwargs)

        for j, _args in enumerate(argument_combinations):
            idx = i*n_met_args + j
            for k, v in zip(constructor_arguments, cstr_args):
                df_dict[k][idx] = v
            for k, v in zip(args, _args):
                if k == "xs":
                    # cast to list necessary for Polars not to scream
                    df_dict[k][idx] = list(np.atleast_1d(v))
                elif k == "floes":
                    df_dict[k][idx] = [(_f.h, _f.x0, _f.L) for _f in v]
                else:
                    df_dict[k][idx] = v

            df_dict[method][idx] = getattr(wave, method)(*_args)
    df = pl.from_dict(df_dict)
    df_src = pl.read_parquet(f"{SRC_TARGET}/"
                             f"met_{method}_reference.parquet")
    return pltesting.assert_frame_equal(df_src, df)
