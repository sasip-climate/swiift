#!/usr/bin/env python3

from hypothesis import given, strategies as st
import numpy as np
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
    assert wave.frequency == frequency
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
    frequency=physical_strategies["wave"]["wave_frequency"],
    phase=st.one_of(st.none(), physical_strategies["wave"]["wave_phase"]),
)
def test_neg(amplitude, period, frequency, phase):
    kwargs = build_kwargs(phase)

    wave = Wave(-amplitude, period=period, **kwargs)
    assert wave.amplitude == amplitude

    wave = Wave(amplitude, period=-period, **kwargs)
    assert wave.period == period

    wave = Wave(-amplitude, period=-period, **kwargs)
    assert wave.amplitude == amplitude
    assert wave.period == period

    wave = Wave(amplitude, frequency=-frequency, **kwargs)
    assert wave.frequency == frequency

    wave = Wave(-amplitude, frequency=-frequency, **kwargs)
    assert wave.amplitude == amplitude
    assert wave.frequency == frequency

    if phase is not None:
        wave = Wave(amplitude, period=period, phase=PI_2 - phase)
        # This should fail if |phase| is forced by Wave.__init__, which is not
        # the expected behaviour. In the general case, test_val should evaluate
        # to PI_2, it evaluates to 0 in the corner case where phase := 0, and
        # to phase when phase is close to 0 because of floating point errors.
        test_val = wave.phase % PI_2 + phase
        assert np.allclose(test_val % PI_2, 0)


@given(
    amplitude=physical_strategies["wave"]["wave_amplitude"],
    period=physical_strategies["wave"]["wave_period"],
    frequency=physical_strategies["wave"]["wave_frequency"],
    phase=st.one_of(st.none(), physical_strategies["wave"]["wave_phase"]),
)
def test_complex(amplitude, period, frequency, phase):
    kwargs = build_kwargs(phase)

    with pytest.warns(np.ComplexWarning):
        wave = Wave(amplitude + 1j, period=period, **kwargs)
        assert wave.amplitude == amplitude

    with pytest.warns(np.ComplexWarning):
        wave = Wave(amplitude, period=period + 1j, **kwargs)
        assert wave.period == period

    with pytest.warns(np.ComplexWarning):
        wave = Wave(amplitude, frequency=frequency + 1j, **kwargs)
        assert wave.frequency == frequency

    if phase is not None:
        with pytest.warns(np.ComplexWarning):
            wave = Wave(amplitude, frequency=frequency, phase=phase + 1j)
            assert wave.phase == phase


@given(
    amplitude=physical_strategies["wave"]["wave_amplitude"],
    period=physical_strategies["wave"]["wave_period"],
)
def test_cached_properties(amplitude, period):
    wave = Wave(amplitude, period=period)
    assert np.allclose(wave.period - PI_2 / wave.angular_frequency, 0)
    assert wave.angular_frequency2 == wave.angular_frequency**2
