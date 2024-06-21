#!/usr/bin/env python3

from hypothesis import given, strategies as st
import numpy as np
from pydantic import ValidationError
import pytest

from flexfrac1d.flexfrac1d import Wave
from flexfrac1d.lib.constants import PI_2

from .conftest import physical_strategies


def build_kwargs(phase):
    if phase is None:
        kwargs = {}
    else:
        kwargs = {"phase": phase}
    return kwargs


@given(
    amplitude=physical_strategies["wave"]["wave_amplitude"],
    period=physical_strategies["wave"]["wave_period"],
    frequency=physical_strategies["wave"]["wave_frequency"],
    phase=st.one_of(st.none(), physical_strategies["wave"]["wave_phase"]),
)
def test_period_xor_frequency(amplitude, period, frequency, phase):
    kwargs = build_kwargs(phase)

    wave = Wave(amplitude, period=period, **kwargs)
    assert wave.amplitude == amplitude
    assert wave.period == period
    assert wave.frequency == 1 / period
    if phase is not None:
        assert wave.phase == phase
    else:
        assert wave.phase == 0

    wave = Wave(amplitude, frequency=frequency, **kwargs)
    assert wave.amplitude == amplitude
    # In this case, wave.frequency := 1 / (1 / frequency)
    # and floating point errors may occur
    assert np.allclose(wave.frequency - frequency, 0)
    assert wave.period == 1 / frequency
    if phase is not None:
        assert wave.phase == phase
    else:
        assert wave.phase == 0

    with pytest.raises(ValueError):
        Wave(amplitude, **kwargs)

    with pytest.warns(UserWarning):
        wave = Wave(amplitude, period=period, frequency=frequency, **kwargs)
        assert wave.amplitude == amplitude
        assert wave.period == period
        assert wave.frequency == 1 / period
        if phase is not None:
            assert wave.phase == phase
        else:
            assert wave.phase == 0


@given(
    amplitude=physical_strategies["wave"]["wave_amplitude"],
    period=physical_strategies["wave"]["wave_period"],
)
def test_cached_properties(amplitude, period):
    wave = Wave(amplitude, period=period)
    assert np.allclose(wave.period - PI_2 / wave.angular_frequency, 0)
    assert wave.angular_frequency2 == wave.angular_frequency**2


@given(
    amplitude=physical_strategies["wave"]["wave_amplitude"],
    period=physical_strategies["wave"]["wave_period"],
)
def test_immutable(amplitude, period):
    wave = Wave(amplitude, period=period)
    with pytest.raises(ValidationError):
        wave.amplitude = 1
        wave.period = 1
        wave.phase = 1


@given(
    amplitude=physical_strategies["wave"]["wave_amplitude"],
    period=physical_strategies["wave"]["wave_period"],
    alt_phase=st.floats(
        min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
    ),
)
def test_bounded_phase(amplitude, period, alt_phase):
    wave = Wave(amplitude, period=period, phase=alt_phase)
    assert np.allclose(wave.phase - alt_phase % PI_2, 0)
