from hypothesis import given, strategies as st
import numpy as np
import pytest
import scipy.integrate as integrate

import swiift.api.spectra as spectra
from swiift.lib.constants import PI_2
from tests.physical_strategies import PHYSICAL_STRATEGIES

_freqencies = (
    np.array([1, 2]),
    np.array([0.5, 1.2]),
    np.linspace(0.3, 0.7, 10),
    0.3,
)

generic_spectra = (
    spectra.PiersonMoskowitz.from_peak_frequency(0.2, 9.8),
    spectra.Bretschneider.from_peak_frequency_swh(0.18, 2.1),
    spectra.JONSWAP.from_parameters(0.21, 3.3, 9.8),
)


@pytest.mark.parametrize("frequencies", _freqencies)
@pytest.mark.parametrize("spectrum", generic_spectra)
def test_angular_density(frequencies, spectrum: spectra._ParametrisedSpectrum):
    density_from_frequency = spectrum.density(frequencies)
    angular_frequencies = PI_2 * frequencies
    density_from_ang_frequency = spectrum._density_ang(angular_frequencies)

    # The density should be equal up to the scaling factor 2pi.
    assert np.allclose(density_from_frequency, density_from_ang_frequency * PI_2)

    # Integrating the densities wrt their arguments
    # should yield identical results.
    if len(np.atleast_1d(frequencies)) > 1:
        assert np.isclose(
            spectrum.discrete_energy(frequencies),
            np.trapezoid(density_from_ang_frequency, angular_frequencies),
        )


@pytest.mark.parametrize("spectrum", generic_spectra)
def test_total_energy(spectrum):
    res, _ = integrate.quad(spectrum.density, 0, np.inf)
    swh_num = 4 * np.sqrt(res)
    target = spectrum.swh

    if isinstance(spectrum, spectra.JONSWAP):
        # Numerically integrating the JONSWAP density (with `tanhsinh`
        # and `maxlevel=12`) gives an absolute error on the cubic
        # approximation bounded by 4.6e-3 (assuming the numerical value
        # is the truth). Peakedness discretised on 500 linearly spaced
        # points between 1 and 7. We set a slighlty looser bound here to
        # be sure that the test passes with random input.
        jonswap_atol = 6.5e-3
        assert np.isclose(swh_num, target, atol=jonswap_atol)
    else:
        assert np.isclose(swh_num, target)


@pytest.mark.parametrize("spectrum", generic_spectra)
def test_properties(spectrum):
    fp = spectrum.peak_frequency

    assert np.isclose(spectrum.peak_period, 1 / fp)
    assert np.isclose(spectrum.peak_ang_frequency, PI_2 * fp)


class TestPiersonMoskowitz:
    @staticmethod
    @given(
        swh=PHYSICAL_STRATEGIES[("wave", "amplitude")],
        gravity=PHYSICAL_STRATEGIES[("gravity",)],
    )
    def test_recover_swh(swh: float, gravity: float):
        spectrum = spectra.PiersonMoskowitz.from_swh(swh, gravity)
        assert np.isclose(spectrum.swh, swh)

    @staticmethod
    @given(
        peak_frequency=PHYSICAL_STRATEGIES[("wave", "frequency")],
        gravity=PHYSICAL_STRATEGIES[("gravity",)],
    )
    def test_recover_peak_frequency(peak_frequency: float, gravity: float):
        spectrum = spectra.PiersonMoskowitz.from_peak_frequency(peak_frequency, gravity)
        assert np.isclose(spectrum.peak_frequency, peak_frequency)


class TestBretschneider:
    @staticmethod
    @given(
        peak_frequency=PHYSICAL_STRATEGIES[("wave", "frequency")],
        swh=PHYSICAL_STRATEGIES[("wave", "amplitude")],
    )
    def test_recover_physicak_values(peak_frequency, swh):
        spectrum = spectra.Bretschneider.from_peak_frequency_swh(peak_frequency, swh)
        assert np.isclose(spectrum.peak_frequency, peak_frequency)
        assert np.isclose(spectrum.swh, swh)
