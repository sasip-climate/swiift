from attrs.exceptions import FrozenInstanceError
from hypothesis import given, strategies as st
import numpy as np
import pytest

from swiift.lib.constants import PI_2
import swiift.model.model as md

from .conftest import physical_strategies


class TestWave:
    @staticmethod
    @given(
        amplitude=physical_strategies["wave"]["amplitude"],
        period=physical_strategies["wave"]["period"],
        phase=(physical_strategies["wave"]["phase"]),
    )
    def test_basic(amplitude, period, phase):
        wave_default, wave_phase = md.Wave(amplitude, period), md.Wave(
            amplitude, period, phase
        )
        for wave in wave_default, wave_phase:
            assert wave.amplitude == amplitude
            assert wave.period == period
        assert wave_default.phase == 0
        assert wave_phase.phase == phase

    @staticmethod
    @given(
        amplitude=physical_strategies["wave"]["amplitude"],
        frequency=physical_strategies["wave"]["frequency"],
        phase=(physical_strategies["wave"]["phase"]),
    )
    def test_frequency(amplitude, frequency, phase):
        wave_default, wave_phase = md.Wave.from_frequency(
            amplitude, frequency
        ), md.Wave.from_frequency(amplitude, frequency, phase)
        for wave in wave_default, wave_phase:
            assert wave.amplitude == amplitude
            assert np.isclose(wave.period - 1 / frequency, 0)
        assert wave_default.phase == 0
        assert wave_phase.phase == phase

    @staticmethod
    @given(
        amplitude=physical_strategies["wave"]["amplitude"],
        period=physical_strategies["wave"]["period"],
    )
    def test_cached_properties(amplitude, period):
        wave = md.Wave(amplitude, period=period)
        assert np.isclose(wave.frequency - 1 / period, 0)
        assert np.isclose(wave.period - PI_2 / wave.angular_frequency, 0)
        assert wave._angular_frequency_pow2 == wave.angular_frequency**2

    @staticmethod
    @given(
        amplitude=physical_strategies["wave"]["amplitude"],
        period=physical_strategies["wave"]["period"],
    )
    def test_immutable(amplitude, period):
        wave = md.Wave(amplitude, period=period)
        with pytest.raises(FrozenInstanceError):
            wave.amplitude = 1
        with pytest.raises(FrozenInstanceError):
            wave.period = 1
        with pytest.raises(FrozenInstanceError):
            wave.phase = 1

    @staticmethod
    @given(
        amplitude=physical_strategies["wave"]["amplitude"],
        period=physical_strategies["wave"]["period"],
        alt_phase=st.floats(
            min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
        ),
    )
    def test_bounded_phase(amplitude, period, alt_phase):
        wave = md.Wave(amplitude, period=period, phase=alt_phase)
        assert np.allclose(wave.phase - alt_phase % PI_2, 0)


class TestSubdomain:
    @staticmethod
    def test_ordering():
        left_edge, length = 0, 100
        sd = md._Subdomain(left_edge, length)

        left_edge_further_right = left_edge + length + 1
        assert left_edge_further_right > sd
        sd_right = md._Subdomain(left_edge_further_right, length)
        assert sd_right > sd

        left_edge_further_left = -200
        assert left_edge_further_left < sd
        sd_left = md._Subdomain(left_edge_further_left, length)
        assert sd_left < sd

        assert sd == left_edge
        sd_same = md._Subdomain(left_edge, length)
        assert sd_same == sd
        sd_same_but_different_length = md._Subdomain(left_edge, 2 * length)
        assert sd_same_but_different_length == sd

        with pytest.raises(TypeError):
            sd > []  # noqa: B015

    @staticmethod
    @pytest.mark.parametrize("left_edge", (-100, 0, 0.8, 45, 34.2))
    @pytest.mark.parametrize("length", (10, 13.8, 101.9))
    def test_right_edge(left_edge, length):
        sd = md._Subdomain(left_edge, length)
        assert sd.right_edge == left_edge + length
