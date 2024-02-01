#!/usr/bin/env python3

# To be used with flexfrac1D v0.1.0

import itertools
import numpy as np
import polars as pl
import scipy.sparse as sparse
from flexfrac1d.wave import Wave
from flexfrac1d.ice import Floe


DIR_TARGET = "gen_end_to_end/floe"
BIG_NUM = 2**64
DUM_WAVE = Wave(1, 100)


def get_floe(comb, wave=None):
    thickness, left_edge, length, disp_rel, dx = comb
    kwargs = dict()
    if disp_rel is not None:
        kwargs["DispType"] = disp_rel
    if dx is not None:
        kwargs["dx"] = dx

    floe = Floe(thickness, left_edge, length, **kwargs)
    if wave is not None:
        floe.setWPars(wave)
    return floe


def sub_dict(dct, keys):
    return {k: dct[k] for k in keys}


def gen_test_attributes(attributes, constructor_keys, combinations):
    n_combinations = len(combinations)
    df_dict = dict(zip(constructor_keys, zip(*combinations)))
    problematic_attributes = ("DispType", "xF", "A", "Asp")
    df_dict["DispType"] = n_combinations * [None]
    df_dict |= {
        att: np.full(n_combinations, np.nan)
        for att in attributes
        if att not in problematic_attributes
    }

    for i, comb in enumerate(combinations):
        _h = f"{hash(comb) % 2**64:x}"
        floe = get_floe(comb)

        for att in attributes:
            if att not in problematic_attributes or att == "DispType":
                df_dict[att][i] = getattr(floe, att)
            else:
                if att == "xF":
                    np.savetxt(f"{DIR_TARGET}/att_{att}_{_h}.csv", floe.xF)
                else:
                    is_sp = False
                    try:
                        arr = floe.A
                    except AttributeError:
                        is_sp = True
                        arr = floe.Asp
                    if is_sp:
                        sparse.save_npz(f"{DIR_TARGET}/att_{att}_{_h}.npz", arr)
                    else:
                        np.savetxt(f"{DIR_TARGET}/att_{att}_{_h}.csv", arr)

    df = pl.from_dict(df_dict)
    df.write_parquet(f"{DIR_TARGET}/attributes_reference.parquet")


def gen_method_arguments():
    method_arguments = {
        "kw": (0, 0.6, 1),  # 0: default parameter
        # "wvf": tuple(map(np.array, ((0.5, -0.15, 0.6), (-1, 0, 1, 0)))),
        "iFracs": (20, (10, 15, 21)),
        "wave": (Wave(1, 50), Wave(0.6, 60)),
        "t": (465, 987, 3654),
        "EType": ("Disp", "Flex"),
        "verbose": (False,),
        "recompute": (False,),
        "istart": (5, 9),
        "iend": (13, 19),
        "multiFrac": (False, True),
        "maxFracs": (1, 2, 3),
        "x_fracs": (46, 49.3, (60, 70.1)),
    }

    return method_arguments


def gen_test_methods(
    constructor_arguments: dict, constructor_combinations: list, methods: dict
):
    def init_eel(floe, wvf, _args):
        floe.calc_w(wvf)
        _kwargs = dict(zip(("wvf", "EType"), (wvf, _args[-1])))
        floe.calc_Eel(**_kwargs)
        return floe

    n_combinations = len(constructor_combinations)
    # iteration on the methods
    for method, args in methods.items():
        # print(method)
        # Cartesian product of the method arguments
        argument_combinations = list(itertools.product(*args.values()))
        n_met_args = len(argument_combinations)
        df_keys = sum(
            map(lambda _d: tuple(_d.keys()), (constructor_arguments, args)), tuple()
        ) + (method,)
        df_dict = {
            k: [None] * n_combinations * len(argument_combinations) for k in df_keys
        }

        # iteration on different instances
        for i, cstr_args in enumerate(constructor_combinations):
            floe = get_floe(cstr_args, DUM_WAVE)
            wvf = 0.1 * np.sin(0.4 * floe.xF + 0.36)  # arbitrary "wave realisation"
            floe.calc_w(wvf)

            # iteration on the combinations of method arguments
            for j, _args in enumerate(argument_combinations):
                idx = i * n_met_args + j
                # populate the dataframe keys: constructor and method parameters
                for k, v in zip(constructor_arguments, cstr_args):
                    df_dict[k][idx] = v
                for k, v in zip(args, _args):
                    if k == "x_fracs":
                        # cast to list necessary for Polars not to scream
                        df_dict[k][idx] = list(np.atleast_1d(v))
                    else:
                        df_dict[k][idx] = v

                # populate the dataframe values: method output
                try:
                    df_dict[method][idx] = getattr(floe, method)(*_args)
                except AttributeError as e:
                    if method in ("calc_curv", "calc_du", "calc_strain"):
                        floe.calc_w(wvf)
                        df_dict[method][idx] = getattr(floe, method)()
                        if method == "calc_strain":
                            # call to getattr first so the attribute is created
                            df_dict[method][idx] = floe.strain
                    else:
                        raise e
                except TypeError as e:
                    if method == "calc_w":
                        floe.calc_w(wvf)
                        df_dict[method][idx] = floe.w
                    elif method == "mslf_int":
                        df_dict[method][idx] = getattr(floe, method)(wvf)
                    elif method == "calc_Eel":
                        # floe.calc_w(wvf)
                        # _kwargs = dict(zip(("wvf", "EType"), (wvf, _args[-1])))
                        # getattr(floe, method)(**_kwargs)
                        floe = init_eel(floe, wvf, _args)
                        df_dict[method][idx] = floe.Eel
                    elif method == "FindE_minVerbose":
                        floe = init_eel(floe, wvf, _args)
                        # floe.calc_w(wvf)
                        # _kwargs = dict(zip(("wvf", "EType"), (wvf, _args[0])))
                        # floe.calc_Eel(**_kwargs)
                        df_dict[method][idx] = getattr(floe, method)(*_args[:-1])[2]
                    else:
                        raise e
                if method == "fracture":
                    df_dict[method][idx] = [
                        [floe.h, floe.x0, floe.L] for floe in df_dict[method][idx]
                    ]
                # elif method == "FindE_min":

                #     df_dict[method][idx] = [_v[2] for _v in df_dict[method][idx]]

        df = pl.from_dict(df_dict)
        df.write_parquet(f"{DIR_TARGET}/met_{method}_reference.parquet")


def main():
    attributes = [
        "h",
        "x0",
        "L",
        "dx",
        "hw",
        "ha",
        "I",
        "k",
        "E",
        "v",
        "DispType",
        "xF",
        "A",
    ]
    method_arguments = gen_method_arguments()
    # FindE_min and FindE_minVerbose both call computeEnergyIfFrac.
    # In the latter, there seems to be a call to the floe.a0 before it is
    # initialised. These three methods thus systematically fail, and might be
    # broken at least for isolated Floe objects.
    # Idem for computeEnergySubFloe, called by FindE_min.
    methods = {
        "calc_alpha": sub_dict(method_arguments, ("kw",)),
        "calc_curv": dict(),
        "calc_du": dict(),
        "calc_Eel": sub_dict(method_arguments, ("EType",)),
        "calc_strain": dict(),
        "calc_w": dict(),
        # "computeEnergyIfFrac": sub_dict(
        #     method_arguments, ("iFracs", "wave", "t", "EType", "verbose", "recompute")
        # ),
        # "computeEnergySubFloe": sub_dict(
        #     method_arguments, ("istart", "iend", "wave", "t", "EType")
        # ),
        # "FindE_min": sub_dict(method_arguments, ("wave", "t", "multiFrac", "EType")),
        # "FindE_minVerbose": sub_dict(
        #     method_arguments, ("maxFracs", "wave", "t", "EType")
        # ),
        "fracture": sub_dict(method_arguments, ("x_fracs",)),
        "mslf_int": dict(),
    }

    constructor_arguments = {
        "thickness": (1, 1.1),
        "left_edge": (-10, 40, 45.2),
        "length": (100, 120.1),
        "dispersion": (None, "Open", "ML", "El", "ElML"),
        "dx": (None, 0.1, 1),
    }
    constructor_combinations = list(itertools.product(*constructor_arguments.values()))

    gen_test_attributes(
        attributes, constructor_arguments.keys(), constructor_combinations
    )
    gen_test_methods(constructor_arguments, constructor_combinations, methods)


if __name__ == "__main__":
    main()
