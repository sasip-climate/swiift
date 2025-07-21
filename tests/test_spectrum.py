from hypothesis import given, strategies as st
import numpy as np
import pytest

from swiift.model.model import DiscreteSpectrum
from tests.physical_strategies import PHYSICAL_STRATEGIES

# Valid cases: all sizes the same, or some are 1
valid_cases = [
    # (amplitude, period, phase)
    (1, 1, 7),  # (1, 1, 1)
    ([1], [1], [7]),  # (1, 1, 1)
    ([1, 2, 9, 6, 1], [1], [7]),  # (5, 1, 1)
    ([1, 3, 4, 2, 1], [1, 1, 1, 1, 1], 7),  # (5, 5, 1)
    ([1, 3, 1, 4, 1], [1, 8, 0.1, 1, 1], [7]),  # (5, 5, 1)
    ([1, 1, 2, 1, 1], [1.5, 1, 0.2, 1, 1], [7, 1, 7, 1, 1]),  # (5, 5, 5)
]
valid_shapes = [
    tuple(
        [
            len(subsequence) if isinstance(subsequence, list) else 0
            for subsequence in sequence
        ]
    )
    for sequence in valid_cases
]

invalid_cases = [
    ([1, 1, 1], [1, 1], 1),  # (3, 2, 1)
    ([1, 1, 1], [1, 1], [1]),  # (3, 2, 1)
    ([1, 1, 1], [1, 1, 1, 1], [1]),  # (3, 4, 1)
    ([1, 1, 1], [1, 1], [1, 1, 1]),  # (3, 2, 3)
    ([1, 1, 1], [1, 1], [1, 1, 1, 1]),  # (3, 2, 4)
]


def parse_shape(
    size: int, strategy: st.SearchStrategy[float]
) -> st.SearchStrategy[float | list[float]]:
    if size == 0:
        return strategy
    else:
        return st.lists(
            strategy,
            min_size=size,
            max_size=size,
        )


@st.composite
def spectrum_arguments(
    draw: st.DrawFn, sizes: tuple[int, int, int]
) -> list[float | list[float]]:
    return [
        draw(parse_shape(size, strategy))
        for size, strategy in zip(
            sizes,
            (
                PHYSICAL_STRATEGIES[("wave", "amplitude")],
                PHYSICAL_STRATEGIES[("wave", "frequency")],
                PHYSICAL_STRATEGIES[("wave", "phase")],
            ),
        )
    ]


@st.composite
def spectrum_arguments_periods(
    draw: st.DrawFn, sizes: tuple[int, int, int]
) -> list[float | list[float]]:
    return [
        draw(parse_shape(size, strategy))
        for size, strategy in zip(
            sizes,
            (
                PHYSICAL_STRATEGIES[("wave", "amplitude")],
                PHYSICAL_STRATEGIES[("wave", "period")],
                PHYSICAL_STRATEGIES[("wave", "phase")],
            ),
        )
    ]


@pytest.mark.parametrize("args", valid_cases)
@pytest.mark.parametrize("with_shape", (True, False))
@pytest.mark.parametrize("as_array", (True, False))
def test_valid_shapes(args, with_shape: bool, as_array: bool):
    passed_args = args[:-1] if not with_shape else args
    if as_array:
        passed_args = (np.asarray(_a) for _a in passed_args)
    spectrum = DiscreteSpectrum(*passed_args)
    assert spectrum.amplitudes.size > 0
    assert len(spectrum.amplitudes.shape) == 1
    assert spectrum.amplitudes.shape == spectrum.frequencies.shape
    assert spectrum.amplitudes.shape == spectrum.phases.shape
    assert np.all(spectrum.phases >= 0) and np.all(spectrum.phases < 2 * np.pi)


@pytest.mark.parametrize("args", invalid_cases)
@pytest.mark.parametrize("with_phase", (True, False))
@pytest.mark.parametrize("as_array", (True, False))
def test_invalid_shapes(args, with_phase: bool, as_array: bool):
    passed_args = args if with_phase else args[:-1]
    if as_array:
        passed_args = (np.asarray(_a) for _a in passed_args)
    with pytest.raises(ValueError):
        DiscreteSpectrum(*passed_args)


@pytest.mark.parametrize("shapes", valid_shapes)
@pytest.mark.parametrize("with_phase", (True, False))
@given(data=st.data())
def test_from_periods(
    data: st.DataObject, shapes: tuple[int, int, int], with_phase: bool
):
    amplitudes, periods, phases = data.draw(spectrum_arguments_periods(shapes))
    if with_phase:
        spectrum = DiscreteSpectrum.from_periods(amplitudes, periods, phases)
    else:
        spectrum = DiscreteSpectrum.from_periods(amplitudes, periods)
    assert np.allclose(1 / spectrum.frequencies, spectrum.periods)
    # Test frequencies rather than periods, because of floating point errors
    assert np.isin(
        1 / np.array(periods), spectrum.frequencies, assume_unique=False
    ).all()


@pytest.mark.parametrize("shapes", valid_shapes)
@pytest.mark.parametrize("with_phase", (True, False))
@pytest.mark.parametrize(
    "property", ("periods", "angular_frequencies", "_ang_freqs_pow2", "nf", "energy")
)
@given(data=st.data())
def test_properties(
    data: st.DataObject,
    shapes: tuple[int, int, int],
    with_phase: bool,
    property: str,
):
    amplitudes, frequencies, phases = data.draw(spectrum_arguments(shapes))
    if with_phase:
        spectrum = DiscreteSpectrum(amplitudes, frequencies, phases)
    else:
        spectrum = DiscreteSpectrum(amplitudes, frequencies)

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
        if with_phase:
            nf = max(nf, np.ravel(phases).size)
        assert nf == spectrum.nf
    elif property == "energy":
        assert np.allclose(np.sum(spectrum.amplitudes**2) / 2, spectrum.energy)
    else:
        raise ValueError(f"DiscreteSpectrum objects have no property {property}.")
