import abc
import functools

import attrs
import numpy as np

from ..lib.constants import PI_2

_ALPHA = 8.1e-3
_peak_width_leq = 0.07
_peak_width_gt = 0.09


class _ParametrisedSpectrum(abc.ABC):
    @abc.abstractmethod
    def density(self, frequency):
        pass

    def discrete_energy(self, frequency):
        return np.trapezoid(self.density(frequency), frequency)

    def _density_ang(self, angular_frequency):
        return self.density(angular_frequency / PI_2) / PI_2

    def __call__(self, frequency):
        return self.density(frequency)


@attrs.define
class _PMFamily(_ParametrisedSpectrum):
    scale: float
    exp_scale: float

    @functools.cached_property
    def peak_period(self):
        return 1 / self.peak_frequency

    @functools.cached_property
    def peak_ang_frequency(self):
        return PI_2 * self.peak_frequency

    @functools.cached_property
    def swh(self):
        return 2 * (self.scale / self.exp_scale) ** 0.5

    @functools.cached_property
    def peak_frequency(self):
        return (4 * self.exp_scale / 5) ** 0.25

    def density(self, frequency):
        return self.scale * frequency**-5 * np.exp(-self.exp_scale * frequency**-4)


@attrs.define
class PiersonMoskowitz(_PMFamily):
    @staticmethod
    def _make_scale(gravity):
        return _ALPHA * (gravity / PI_2**2) ** 2

    @classmethod
    def from_swh(cls, swh, gravity):
        scale = cls._make_scale(gravity)
        exp_scale = 4 * scale / swh
        return cls(scale, exp_scale)

    @classmethod
    def from_peak_frequency(cls, peak_frequency, gravity):
        scale = cls._make_scale(gravity)
        exp_scale = 5 / 4 * peak_frequency**4
        return cls(scale, exp_scale)


class Bretschneider(_PMFamily):
    @classmethod
    def from_peak_frequency_swh(cls, peak_frequency, swh):
        scale = 5 * swh**2 * peak_frequency**4 / 16
        exp_scale = 5 / 4 * peak_frequency**4
        return cls(scale, exp_scale)


@attrs.define
class JONSWAP(_ParametrisedSpectrum):
    peakedness: float
    _base_spectrum: PiersonMoskowitz

    @property
    def peak_frequency(self):
        return self._base_spectrum.peak_frequency

    @functools.cached_property
    def swh(self):
        return (
            (
                1.555
                + 0.2596 * self.peakedness
                - 0.02231 * self.peakedness**2
                + 0.001142 * self.peakedness**3
            )
            * self._base_spectrum.scale**0.5
            / self.peak_frequency**2
        )

    @classmethod
    def from_parameters(cls, peak_frequency, peak_control, gravity):
        pm_spectrum = PiersonMoskowitz.from_peak_frequency(peak_frequency, gravity)
        return cls(peak_control, pm_spectrum)

    def density(self, frequency):
        peak_width = np.ones_like(frequency) * _peak_width_leq
        peak_width[frequency > self._base_spectrum.peak_frequency] = _peak_width_gt
        peak_enhancement = self.peakedness ** np.exp(
            -((frequency - self.peak_frequency) ** 2)
            / (2 * peak_width**2 * self.peak_frequency**2)
        )
        return self._base_spectrum(frequency) * peak_enhancement
