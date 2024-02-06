#!/usr/bin/env python3

import itertools
import pytest
from hypothesis import assume, given, strategies as st
import hypothesis.extra.numpy as npst
import numpy as np

from flexfrac1d.flexfrac1d import DiscreteSpectrum

# @pytest.fixture
# def frequencies():
#     frequencies = 1, [3, 4, 5], [[3, 4], [2, 1]], np.array((1, 2, 3))


# @st.composite
# def scalars(draw, n=2):
#     return [
#         draw(st.floats(min_value=1e-14, allow_nan=False, allow_infinity=False))
#         for _ in range(n)
#     ]
#     amplitudes = draw(st.floats(min_value=1e-14, allow_nan=False, allow_infinity=False))
#     return amplitudes, frequencies


number = st.one_of(
    st.floats(allow_nan=False, allow_infinity=False),
    st.integers(),
)
non_negative_number = st.one_of(
    st.floats(min_value=0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=0),
)

positive_number = st.one_of(
    st.floats(
        min_value=np.finfo(np.float64).eps, allow_nan=False, allow_infinity=False
    ),
    st.integers(min_value=1),
)

positive_array_or_number = st.one_of(
    positive_number,
    st.lists(positive_number, min_size=1),
    npst.arrays(
        np.float64,
        shape=st.integers(min_value=1),
        elements=positive_number,
    ),
)

# min_size = st.shared(st.integers(min_value=1), key="size")
shape = st.shared(npst.array_shapes(max_dims=1), key="size")
dtype = st.shared(npst.floating_dtypes(sizes=(32, 64)), key="dtype")
# .filter(
#     lambda s: str(s)[0] == "f" or str(s)[1] == "f"
# )


@given(
    amplitudes=(
        non_negative_number
        | npst.arrays(
            npst.floating_dtypes(),
            shape=shape,
            elements={"min_value": 0, "allow_infinity": False},
        )
    ),
    frequencies=(
        positive_number
        | npst.arrays(
            dtype,
            shape=shape,
            elements={"min_value": 0, "allow_infinity": False},
        )
    ),
    phases=(
        number
        | npst.arrays(
            npst.floating_dtypes(), shape=shape, elements={"allow_infinity": False}
        )
    ),
    betas=(
        non_negative_number
        | npst.arrays(
            npst.floating_dtypes(),
            shape=shape,
            elements={"min_value": 0, "allow_infinity": False},
        )
    ),
)
@pytest.mark.parametrize(
    "fph, fb", (e for e in itertools.product((False, True), repeat=2))
)
def test_sanitised(amplitudes, frequencies, phases, betas, fph, fb):
    kwargs = dict()
    if not fph:
        kwargs["phases"] = phases
    if not fb:
        kwargs["betas"] = betas
    # assume(len(np.atleast_1d(amplitudes)) == len(np.atleast_1d(frequencies)))
    if isinstance(frequencies, np.ndarray):
        frequencies += np.finfo(frequencies.dtype).eps
    DiscreteSpectrum(amplitudes, frequencies, **kwargs)


# @given(
#     amplitudes=(
#         non_negative_number
#         | npst.arrays(
#             npst.floating_dtypes(),
#             shape=npst.array_shapes(max_dims=1),
#             elements={"min_value": 0, "allow_infinity": False},
#         )
#     ),
#     frequencies=(
#         positive_number
#         | npst.arrays(
#             dtype,
#             shape=npst.array_shapes(max_dims=1),
#             elements={"min_value": 0, "allow_infinity": False},
#         )
#     ),
#     phases=(
#         number
#         | npst.arrays(
#             npst.floating_dtypes(),
#             shape=npst.array_shapes(max_dims=1),
#             elements={"allow_infinity": False},
#         )
#     ),
#     betas=(
#         non_negative_number
#         | npst.arrays(
#             npst.floating_dtypes(),
#             shape=npst.array_shapes(max_dims=1),
#             elements={"min_value": 0, "allow_infinity": False},
#         )
#     ),
# )
# @pytest.mark.parametrize(
#     "fph, fb", (e for e in itertools.product((False, True), repeat=2))
# )
# def test_unsanitised(amplitudes, frequencies, phases, betas, fph, fb):
#     cond = np.unique(
#         [arr.size for arr in map(np.ravel, (amplitudes, frequencies, phases, betas))]
#     )
#     assume(cond.size > 1 and cond[0] != 1)
#     # assume(len(np.atleast_1d(amplitudes)) == len(np.atleast_1d(frequencies)) or
#     #        )
#     if isinstance(frequencies, np.ndarray):
#         frequencies += np.finfo(frequencies.dtype).eps
#     kwargs = dict()
#     if not fph:
#         kwargs["phases"] = phases
#     if not fb:
#         kwargs["betas"] = betas
#     with pytest.raises(ValueError):
#         DiscreteSpectrum(amplitudes, frequencies, **kwargs)


# # same_len_lists = ints(min_value=1, max_value=100).flatmap(lambda n: st.lists(st.lists(ints(), min_size=n, max_size=n), min_size=2, max_size=2))
# # non_negative_number.flatmap(lambda n:
