#!/usr/bin/env python3

from attrs.exceptions import FrozenInstanceError
from hypothesis import given, strategies as st
import numpy as np
import pytest

from flexfrac1d.flexfrac1d import Wave
from flexfrac1d.lib.constants import PI_2

from .conftest import physical_strategies


@given(
    amplitude=physical_strategies["wave"]["amplitude"],
    period=physical_strategies["wave"]["period"],
    phase=(physical_strategies["wave"]["phase"]),
)
def test_basic(amplitude, period, phase):
    wave_default, wave_phase = Wave(amplitude, period), Wave(amplitude, period, phase)
    for wave in wave_default, wave_phase:
        assert wave.amplitude == amplitude
        assert wave.period == period
    assert wave_default.phase == 0
    assert wave_phase.phase == phase


@given(
    amplitude=physical_strategies["wave"]["amplitude"],
    frequency=physical_strategies["wave"]["frequency"],
    phase=(physical_strategies["wave"]["phase"]),
)
def test_frequency(amplitude, frequency, phase):
    wave_default, wave_phase = Wave.from_frequency(
        amplitude, frequency
    ), Wave.from_frequency(amplitude, frequency, phase)
    for wave in wave_default, wave_phase:
        assert wave.amplitude == amplitude
        assert np.isclose(wave.period - 1 / frequency, 0)
    assert wave_default.phase == 0
    assert wave_phase.phase == phase


@given(
    amplitude=physical_strategies["wave"]["amplitude"],
    period=physical_strategies["wave"]["period"],
)
def test_cached_properties(amplitude, period):
    wave = Wave(amplitude, period=period)
    assert np.isclose(wave.frequency - 1 / period, 0)
    assert np.isclose(wave.period - PI_2 / wave.angular_frequency, 0)
    assert wave.angular_frequency2 == wave.angular_frequency**2


@given(
    amplitude=physical_strategies["wave"]["amplitude"],
    period=physical_strategies["wave"]["period"],
)
def test_immutable(amplitude, period):
    wave = Wave(amplitude, period=period)
    with pytest.raises(FrozenInstanceError):
        wave.amplitude = 1
    with pytest.raises(FrozenInstanceError):
        wave.period = 1
    with pytest.raises(FrozenInstanceError):
        wave.phase = 1


@given(
    amplitude=physical_strategies["wave"]["amplitude"],
    period=physical_strategies["wave"]["period"],
    alt_phase=st.floats(
        min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
    ),
)
def test_bounded_phase(amplitude, period, alt_phase):
    wave = Wave(amplitude, period=period, phase=alt_phase)
    assert np.allclose(wave.phase - alt_phase % PI_2, 0)
