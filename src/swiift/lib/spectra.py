import abc
import functools

from ..lib.constants import PI_2

_alpha = 8.1e-3


class ParametrisedSpectrum(abc.ABC):
    @abc.abstractmethod
    def density(self, frequency):
        pass

    def _density_ang(self, angular_frequency):
        return self.density(angular_frequency / PI_2) / PI_2


class PMFamily(ParametrisedSpectrum):
    scale: float
    exp_scale: float
    peak_control: float | None = None

    @functools.cached_property
    def peak_period(self):
        return 1 / self.peak_frequency

    @functools.cached_property
    def peak_ang_frequency(self):
        return PI_2 * self.peak_frequency


class PiersonMoskowitz(ParametrisedSpectrum):
    @staticmethod
    def _make_scale(gravity):
        return _alpha * gravity**2 * PI_2**-4

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

    @classmethod
    def from_peak_ang_frequency(cls, peak_ang_frequency, gravity):
        return cls.from_peak_frequency(peak_ang_frequency / PI_2, gravity)

    @classmethod
    def from_peak_period(cls, peak_period, gravity):
        return cls.from_peak_frequency(1 / peak_period, gravity)

    @functools.cached_property
    def swh(self):
        return 2 * (self.scale / self.exp_scale) ** 0.5

    @functools.cached_property
    def peak_frequency(self):
        return (4 * self.exp_scale / 5) ** 0.25


class JONSWAP(PMFamily):
    pass
