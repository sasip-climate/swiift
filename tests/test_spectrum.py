import itertools

from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import primefac
import pytest

from swiift.model.model import DiscreteSpectrum
from tests.conftest import float_kw

SpectrumParameters = tuple[
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray | None,
]
SpectrumParametersKWA = tuple[
    float | np.ndarray,
    float | np.ndarray,
    dict[
        str,
        float | np.ndarray,
    ],
]

# `test_properties` breaks if max_value is left open (overflow error).
local_float_kw = float_kw | {"allow_infinity": False, "max_value": 1e9}
number = st.one_of(
    st.floats(**local_float_kw),
    st.integers(),
)

non_neg_float_kw = local_float_kw | {"min_value": 0}
non_negative_number = st.one_of(
    st.floats(**non_neg_float_kw),
    st.integers(min_value=0),
)

pos_float_kw = non_neg_float_kw | {"allow_subnormal": False, "exclude_min": True}
# Use of int max_value to prevent C int overflow problems
positive_number = st.one_of(
    st.floats(**pos_float_kw),
    st.integers(min_value=1, max_value=2**32 - 1),
)


def get_optional_kwargs(*args):
    combinations = [
        _c for n in range(len(args) + 1) for _c in itertools.combinations(args, n)
    ]
    return st.sampled_from(combinations)


@st.composite
def comp_shapes(draw: st.DrawFn, size: int, max_dims: int | None = None) -> tuple:
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
def broadcastable(
    draw: st.DrawFn,
) -> SpectrumParametersKWA:
    size = draw(st.integers(min_value=0, max_value=255))
    shape_st = comp_shapes(size)
    ds_kw = dict()

    amplitudes = draw(get_number_or_array(shape_st, False, "pos"))
    frequencies = draw(get_number_or_array(shape_st, False, "pos"))
    opt = draw(get_optional_kwargs("phases", "shapes"))
    if "phases" in opt:
        ds_kw["phases"] = draw(get_number_or_array(shape_st, False))

    return amplitudes, frequencies, ds_kw


def get_number_or_array(
    shape: st.SearchStrategy | int,
    strict: bool = False,
    constraints: str | None = None,
) -> st.SearchStrategy:
    if constraints == "non_neg":
        strategy = npst.arrays(
            npst.floating_dtypes(sizes=64),
            shape=shape,
            elements=non_neg_float_kw,
        )
        if not strict:
            return non_negative_number | strategy
    elif constraints == "pos":
        strategy = npst.arrays(
            npst.floating_dtypes(sizes=64),
            shape=shape,
            elements=pos_float_kw,
        )
        if not strict:
            return positive_number | strategy
    else:
        strategy = npst.arrays(
            npst.floating_dtypes(sizes=64),
            shape=shape,
            elements=local_float_kw,
        )
        if not strict:
            return number | strategy
    return strategy


@st.composite
def not_broadcastable(
    draw: st.DrawFn,
) -> SpectrumParameters:
    # 1 is excluded as it will be compatible with anything, and we want to include 0
    sizes = draw(
        st.lists(
            st.just(0) | st.integers(min_value=2, max_value=255),
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
def test_sanitised(args: SpectrumParametersKWA):
    amplitudes, frequencies, kwargs = args
    spectrum = DiscreteSpectrum(amplitudes, frequencies, **kwargs)

    assert spectrum.amplitudes.size > 0
    assert len(spectrum.amplitudes.shape) == 1
    assert spectrum.amplitudes.shape == spectrum.frequencies.shape
    assert spectrum.amplitudes.shape == spectrum.phases.shape


@given(args=not_broadcastable())
def test_unsanitised(args: SpectrumParameters):
    amplitudes, frequencies, phases = args
    with pytest.raises(ValueError):
        if phases is None:
            DiscreteSpectrum(amplitudes, frequencies)
        else:
            DiscreteSpectrum(amplitudes, frequencies, phases)


@given(args=broadcastable())
@pytest.mark.parametrize(
    "property", ("periods", "angular_frequencies", "_ang_freqs_pow2", "nf", "energy")
)
def test_properties(args, property: str):
    amplitudes, frequencies, kwargs = args
    spectrum = DiscreteSpectrum(amplitudes, frequencies, **kwargs)
    if property == "periods":
        assert np.allclose(1 / spectrum.frequencies, spectrum.periods)
    elif property == "angular_frequencies":
        assert np.allclose(
            2 * np.pi * spectrum.frequencies, spectrum.angular_frequencies
        )
    elif property == "_ang_freqs_pow2":
        assert np.allclose(
            (2 * np.pi * spectrum.frequencies) ** 2, spectrum._ang_freqs_pow2
        )
    elif property == "nf":
        nf = max(np.ravel(amplitudes).size, np.ravel(frequencies).size)
        if "phases" in kwargs:
            nf = max(nf, np.ravel(kwargs["phases"]).size)
        assert nf == spectrum.nf
    elif property == "energy":
        assert np.allclose(np.sum(spectrum.amplitudes**2) / 2, spectrum.energy)
    else:
        raise ValueError(f"DiscreteSpectrum objects have no property {property}.")
