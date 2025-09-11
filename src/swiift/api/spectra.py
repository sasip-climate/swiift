"""Consistant implementations of usual spectral formulations.

Spectra are defined as function of the frequency. Definitions are taken
from [1]_.

References
----------
.. [1] Stansberg, Carl Trygve & Contento, G. & Hong, S.W. & Irani, M. &
   Ishida, S. & Mercier, R.. (2002). The specialist committee on waves
   final report and recommendations to the 23rd ITTC. Proceedings of the
   23rd ITTC. 505-736.

"""

import abc
import functools
import typing

import attrs
import numpy as np

from ..lib.constants import PI_2
from ..lib.physics import _demote_to_scalar

"""Constant appearing in the definition of the PM family spectra."""
_ALPHA = 8.1e-3
"""JONSWAP peak width, below and beyond peak frequency."""
_PEAK_WIDTH_LEQ = 0.07
_PEAK_WIDTH_GT = 0.09

T = typing.TypeVar("T", float, np.ndarray[tuple[int], np.dtype[np.floating]])


class _ParametrisedSpectrum(abc.ABC):
    @abc.abstractmethod
    def density(self, frequency: T) -> T:
        """Spectral density as a function of frequency.

        Parameters
        ----------
        frequency: array_like
            Frequency at which to evaluate the density.

        Returns
        -------
        density: float or np.ndarray
           The density in m^2 s.

        """

    def discrete_energy(self, frequencies: np.ndarray) -> float:
        r"""Spectral density integrated over the input frequencies.

        The full spectral energy is obtained by integrating the
        density over the positive half-line: this method computes
        energy for the frequency band spanned by the input array.

        The spectral density is computed for the input frequencies,
        and integrated over these same frequencies using the
        trapezoid rule, so that for an input of size :math:`N+1`:

        .. math::

            E(f_0, f_{N-1}) \approx \frac{1}{2} \sum_j=0^{N-1} {
                (S(f_j) + S(f_{j+1})) (f_{j+1} - f_j)
            }.


        It is assumed the input array is sorted.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies to use for computing the energy.

        Returns
        -------
        float
            Wave spectral energy in m^2.

        """
        return np.trapezoid(self.density(frequencies), frequencies)

    def _density_ang(self, angular_frequency: T) -> T:
        # Helper method to use angular frequency as input to the density.
        return self.density(angular_frequency / PI_2) / PI_2

    def __call__(self, frequency: T) -> T:
        """Wrapper for `density`.

        Parameters
        ----------
        frequency : T
            Frequency in Hz.

        """
        return self.density(frequency)


@attrs.define
class _UnimodalSpectrum(_ParametrisedSpectrum):
    @property
    @abc.abstractmethod
    def peak_frequency(self) -> float:
        """Peak frequency.

        Returns
        -------
        float
            Peak frequency, in Hz.

        """

    @property
    @abc.abstractmethod
    def swh(self) -> float:
        r"""Significant wave height.

        We use the spectral definition,

        .. math:: H_s = 4\sqrt{\int_0^{+\infty} S(f) df}.

        Returns
        -------
        float
            Significant wave height, in m.

        """

    @functools.cached_property
    def peak_period(self) -> float:
        """Spectral peak (modal) period.

        Returns
        -------
        float
            Peak period, in s.

        """
        return 1 / self.peak_frequency

    @functools.cached_property
    def peak_ang_frequency(self) -> float:
        """Spectral peak angular frequency.

        Returns
        -------
        float
            Peak angular frequency, in rad s^-1.

        """
        return PI_2 * self.peak_frequency


@attrs.define
class _PMFamily(_UnimodalSpectrum):
    r"""Base class for spectra of the Bretschneider family.

    These spectra are of the form

    .. math:: S(f) = \frac{A}{f^5}\exp{\Bigl(-\frac{B}{f^4}\Bigr)}

    with the definitions of the parameters :math:`A` and :math:`B`
    identifying particular forms. For clarity, we call them
    respectively `scale` and `exp_scale`.

    In practice, spectra are parameterised by physical variables
    which can be linked back to these parameters. In particular,
    the peak frequency is given by

    .. math:: f_p = {\Bigl(\frac{4B}{5}\Bigr)}^{1/4}

    and the significant wave height by

    .. math:: H_s = 2\sqrt{\frac{A}{B}}.

    Attributes
    ----------
    scale : float
    exp_scale : float

    Methods
    -------
    __call__

    """

    scale: float
    exp_scale: float

    @functools.cached_property
    def swh(self) -> float:
        return 2 * (self.scale / self.exp_scale) ** 0.5

    @functools.cached_property
    def peak_frequency(self) -> float:
        return (4 * self.exp_scale / 5) ** 0.25

    def density(self, frequency):
        return self.scale / frequency**5 * np.exp(-self.exp_scale / frequency**4)


@attrs.define
class PiersonMoskowitz(_PMFamily):
    """Class encapsulating the Pierson--Moskowitz spectrum."""

    @staticmethod
    def _make_scale(gravity):
        return _ALPHA * (gravity / PI_2**2) ** 2

    @classmethod
    def from_swh(cls, swh: float, gravity: float) -> typing.Self:
        """Build a spectrum parameterised by the SWH.

        Parameters
        ----------
        swh : float
            Significant wave height, in m.
        gravity : float
           Acceleration of gravity, in m s**-2.

        Returns
        -------
        PiersonMoskowitz
            Initialised spectrum object.

        """
        scale = cls._make_scale(gravity)
        exp_scale = 4 * scale / swh**2
        return cls(scale, exp_scale)

    @classmethod
    def from_peak_frequency(cls, peak_frequency: float, gravity: float) -> typing.Self:
        """Build a spectrum parameterised by the peak frequency.

        Parameters
        ----------
        peak_frequency : float
            Peak frequency, in Hz.
        gravity : float
            Acceleration of gravity, in m s^-2.

        Returns
        -------
        PiersonMoskowitz
            Initialised spectrum object.

        """
        scale = cls._make_scale(gravity)
        exp_scale = 5 / 4 * peak_frequency**4
        return cls(scale, exp_scale)


class Bretschneider(_PMFamily):
    """Class encapsulating the Bretschneider spectrum."""

    @classmethod
    def from_peak_frequency_swh(cls, peak_frequency: float, swh: float) -> typing.Self:
        """Build a spectrum parameterised by peak frequency and SWH.

        Parameters
        ----------
        peak_frequency : float
            Peak frequency, in Hz.
        swh : float
            Significant wave height, in m.

        Returns
        -------
        Bretschneider
            Initialised spectrum.

        """
        scale = 5 * swh**2 * peak_frequency**4 / 16
        exp_scale = 5 / 4 * peak_frequency**4
        return cls(scale, exp_scale)


@attrs.define
class JONSWAP(_UnimodalSpectrum):
    """Class encapsulating the JONSWAP spectrum.

    The JONSWAP spectrum modifies the PM spectrum by introducing a peak
    enhancement factor.

    Attributes
    ----------
    peakedness : float
    _base_spectrum : PiersonMoskowitz

    """

    peakedness: float
    _base_spectrum: PiersonMoskowitz

    @property
    def peak_frequency(self):
        """Peak frequency.

        Returns
        -------
        float
            Peak frequency, in m.

        """
        return self._base_spectrum.peak_frequency

    @functools.cached_property
    def swh(self):
        r"""Significant wave height.

        The SWH is computed with a cubic approximation.

        Returns
        -------
        float
            Significant wave height, in m.

        """
        return (
            self._base_spectrum.scale**0.5
            * (
                1.555
                + 0.2596 * self.peakedness
                - 0.02231 * self.peakedness**2
                + 0.001142 * self.peakedness**3
            )
            / self.peak_frequency**2
        )

    @classmethod
    def from_parameters(
        cls,
        peak_frequency: float,
        peakedness: float,
        gravity: float,
    ) -> typing.Self:
        """Build a spectrum.

        Peak frequency and gravity are used to instantiate the
        underlying Pierson--Moskowitz spectrum, while peakedness is
        specific to the JONSWAP formulation.

        Parameters
        ----------
        peak_frequency : float
            Peak frequency, in Hz.
        peakedness : float
            Peakedness (base of the peak enhancement factor).
        gravity : float
            Acceleration of gravity, in m s^-2.

        Returns
        -------
        JONSWAP
            Initialised spectrum.

        """
        pm_spectrum = PiersonMoskowitz.from_peak_frequency(peak_frequency, gravity)
        return cls(peakedness, pm_spectrum)

    # Decoratar actually mandatory for integrate.quad not to error out
    # on this method.
    @_demote_to_scalar
    def density(self, frequency):
        # The dimension of `frequency` cannot be 0,
        # to allow for masking/assignement.
        peak_width = np.ones_like(np.atleast_1d(frequency)) * _PEAK_WIDTH_LEQ
        peak_width[frequency > self.peak_frequency] = _PEAK_WIDTH_GT
        peak_enhancement = self.peakedness ** np.exp(
            -((frequency - self.peak_frequency) ** 2)
            / (2 * peak_width**2 * self.peak_frequency**2)
        )
        return self._base_spectrum(frequency) * peak_enhancement
