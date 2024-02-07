#!/usr/bin/env python3

import itertools
import primefac
import pytest
from hypothesis import assume
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst
import numpy as np

from flexfrac1d.flexfrac1d import DiscreteSpectrum


float_kw = {"allow_nan": False, "allow_infinity": False}
number = st.one_of(
    st.floats(**float_kw),
    st.integers(),
)

non_neg_float_kw = float_kw | {"min_value": 0}
non_negative_number = st.one_of(
    st.floats(**non_neg_float_kw),
    st.integers(min_value=0),
)

pos_float_kw = non_neg_float_kw | {"allow_subnormal": False, "exclude_min": True}
positive_number = st.one_of(
    st.floats(**pos_float_kw),
    st.integers(min_value=1),
)


def get_optional_kwargs(*args):
    combinations = [
        _c for n in range(len(args) + 1) for _c in itertools.combinations(args, n)
    ]
    return st.sampled_from(combinations)


@st.composite
def comp_shapes(draw, size, max_dims=None):
    if size == 0:
        return tuple()
    if size in (1, 2, 3, 5, 7, 11, 13, 17, 19, 23):
        return (size,)

    factors = draw(st.permutations(list(primefac.primefac(size))))
    if max_dims is None:
        max_dims = len(factors)

    ndim = draw(st.integers(min_value=1, max_value=max_dims))
    while len(factors) > ndim:
        factors[0] = factors[0] * factors[-1]
        factors = factors[:-1]

    return tuple(factors)


@st.composite
def broadcastable(draw):
    size = draw(st.integers(min_value=0, max_value=255))
    shape_st = comp_shapes(size)
    ds_kw = dict()

    amplitudes = draw(get_number_or_array(shape_st, False, "non_neg"))
    frequencies = draw(get_number_or_array(shape_st, False, "pos"))
    opt = draw(get_optional_kwargs("phases", "shapes"))
    if "phases" in opt:
        ds_kw["phases"] = draw(get_number_or_array(shape_st, False))
    if "betas" in opt:
        ds_kw["betas"] = draw(get_number_or_array(shape_st, False, "non_neg"))

    return amplitudes, frequencies, ds_kw


def get_number_or_array(shape, strict=False, constraints=None):
    if constraints == "non_neg":
        st = npst.arrays(
            npst.floating_dtypes(),
            shape=shape,
            elements=non_neg_float_kw,
        )
        if not strict:
            return non_negative_number | st
    elif constraints == "pos":
        st = npst.arrays(
            npst.floating_dtypes(),
            shape=shape,
            elements=pos_float_kw,
        )
        if not strict:
            return positive_number | st
    else:
        st = npst.arrays(
            npst.floating_dtypes(),
            shape=shape,
            elements=float_kw,
        )
        if not strict:
            return number | st
    return st


@st.composite
def not_broadcastable(draw):
    def inc_n_arr(arr):
        return isinstance(arr, np.ndarray) and arr.size > 1

    ds_kw = dict()
    opt = draw(get_optional_kwargs("phases", "betas"))
    nb_args = len(opt) + 2
    n_arr = 0

    sizes = draw(
        st.lists(
            st.integers(min_value=0, max_value=255), min_size=nb_args, max_size=nb_args
        ).filter(lambda lst: np.unique(lst).size > 1)
    )
    shape_st = [comp_shapes(size) for size in sizes]

    # The assignation order is randomised, to give each argument a chance of
    # having the option of being scalar
    idxs = draw(st.permutations(range(nb_args)))

    for i, idx in enumerate(idxs):
        strict = n_arr + len(idxs) - i
        if idx == 0:
            amplitudes = draw(get_number_or_array(shape_st[i], strict, "non_neg"))
            if inc_n_arr(amplitudes):
                n_arr += 1
        elif idx == 1:
            frequencies = draw(get_number_or_array(shape_st[i], strict, "pos"))
            if inc_n_arr(frequencies):
                n_arr += 1
        elif idx == 2:
            if "phases" in opt:
                ds_kw["phases"] = draw(get_number_or_array(shape_st[i], strict))
            else:
                ds_kw["betas"] = draw(
                    get_number_or_array(shape_st[i], strict, "non_neg")
                )
        else:
            ds_kw["betas"] = draw(get_number_or_array(shape_st[i], strict, "non_neg"))

    # Should not be necessarry, here as a final "just in case" filter
    assume(
        np.unique(
            list(
                map(
                    lambda arr: arr.size,
                    map(np.ravel, [amplitudes, frequencies] + list(ds_kw.values())),
                )
            )
        ).size
        > 2
    )
    return amplitudes, frequencies, ds_kw


@given(args=broadcastable())
def test_sanitised(args):
    amplitudes, frequencies, kwargs = args
    DiscreteSpectrum(amplitudes, frequencies, **kwargs)


@given(args=not_broadcastable())
def test_unsanitised(args):
    amplitudes, frequencies, kwargs = args
    with pytest.raises(ValueError):
        DiscreteSpectrum(amplitudes, frequencies, **kwargs)
