import itertools
import primefac
import pytest
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

from flexfrac1d.model.model import DiscreteSpectrum


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

    amplitudes = draw(get_number_or_array(shape_st, False, "pos"))
    frequencies = draw(get_number_or_array(shape_st, False, "pos"))
    opt = draw(get_optional_kwargs("phases", "shapes"))
    if "phases" in opt:
        ds_kw["phases"] = draw(get_number_or_array(shape_st, False))

    return amplitudes, frequencies, ds_kw


def get_number_or_array(shape, strict=False, constraints=None):
    if constraints == "non_neg":
        strategy = npst.arrays(
            npst.floating_dtypes(),
            shape=shape,
            elements=non_neg_float_kw,
        )
        if not strict:
            return non_negative_number | strategy
    elif constraints == "pos":
        strategy = npst.arrays(
            npst.floating_dtypes(),
            shape=shape,
            elements=pos_float_kw,
        )
        if not strict:
            return positive_number | strategy
    else:
        strategy = npst.arrays(
            npst.floating_dtypes(),
            shape=shape,
            elements=float_kw,
        )
        if not strict:
            return number | strategy
    return strategy


@st.composite
def not_broadcastable(draw):
    # 1 is excluded as it will be compatible with anything, and we want to include 0
    sizes = draw(
        st.lists(
            st.integers(min_value=0, max_value=255).filter(lambda n: n != 1),
            min_size=2,
            max_size=3,
            unique=True,
        )
    )
    if len(sizes) == 2:
        with_phase = draw(st.booleans())
        if with_phase:
            idx_scalar = draw(st.integers(min_value=0, max_value=2))
            if idx_scalar == 0:
                amplitudes = draw(get_number_or_array(1, constraints="pos"))
                frequencies = draw(
                    get_number_or_array(sizes[0], True, constraints="pos")
                )
                phases = draw(get_number_or_array(sizes[1], True))
            elif idx_scalar == 1:
                amplitudes = draw(
                    get_number_or_array(sizes[0], True, constraints="pos")
                )
                frequencies = draw(get_number_or_array(1, constraints="pos"))
                phases = draw(get_number_or_array(sizes[1], True))
            else:
                amplitudes = draw(
                    get_number_or_array(sizes[0], True, constraints="pos")
                )
                frequencies = draw(
                    get_number_or_array(sizes[1], True, constraints="pos")
                )
                phases = draw(get_number_or_array(1, False))
        else:
            amplitudes = draw(get_number_or_array(sizes[0], True, constraints="pos"))
            frequencies = draw(get_number_or_array(sizes[1], True, constraints="pos"))
            phases = None
    else:
        amplitudes = draw(get_number_or_array(sizes[0], True, constraints="pos"))
        frequencies = draw(get_number_or_array(sizes[1], True, constraints="pos"))
        phases = draw(get_number_or_array(sizes[2], True))

    return amplitudes, frequencies, phases


@given(args=broadcastable())
def test_sanitised(args):
    amplitudes, frequencies, kwargs = args
    DiscreteSpectrum(amplitudes, frequencies, **kwargs)


@given(args=not_broadcastable())
def test_unsanitised(args):
    amplitudes, frequencies, phases = args
    with pytest.raises(ValueError):
        if phases is None:
            DiscreteSpectrum(amplitudes, frequencies)
        else:
            DiscreteSpectrum(amplitudes, frequencies, phases)
