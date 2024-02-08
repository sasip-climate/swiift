#!/usr/bin/env python3


import functools
import itertools
import warnings
import numpy as np
import scipy.optimize as optimize

from .libraries.WaveUtils import PM, Jonswap, PowerLaw, SpecVars, calc_k
from .pars import g


PI = np.pi


class Wave:
    """Represents a monochromatic wave."""

    def __init__(
        self,
        amplitude: float,
        *,
        period: float = None,
        frequency: float = None,
        phase: float = 0,
        beta: float = 0,
    ):
        """Initialise self."""
        if period is None and frequency is None:
            raise ValueError("Either period or frequency must be specified.")
        elif period is not None:
            self.__period = period
            if frequency is not None:
                warnings.warn(
                    (
                        "Both period and frequency were specified, "
                        "frequency will be ignored"
                    ),
                    stacklevel=2,
                )
            self.__frequency = 1 / period
        else:
            self.__frequency = frequency
            self.__period = 1 / frequency

        self.__amplitude = amplitude
        self.__phase = phase
        self.__beta = beta

    @property
    def amplitude(self) -> float:
        """Wave amplitude in m."""
        return self.__amplitude

    @property
    def period(self) -> float:
        """Wave period in s."""
        return self.__period

    @property
    def phase(self) -> float:
        return self.__phase

    @property
    def frequency(self) -> float:
        """Wave frequency in Hz."""
        return self.__frequency

    @functools.cached_property
    def angular_frequency(self) -> float:
        """Wave angular frequency in rad s**-1."""
        return 2 * PI / self.period

    @functools.cached_property
    def angular_frequency2(self) -> float:
        """Squared wave angular frequency, for convenience."""
        return self.angular_frequency**2


class Ocean:
    """Represents the fluid bearing the floes.

    Encapsulates the properties of an incompressible ocean of finite,
    constant depth and given density.

    Parameters
    ----------
    depth : float
        Ocean constant and finite depth in m.
    density : float
        Ocean constant density in kg m**-3.

    Attributes
    ----------
    density : float
    depth : float

    """

    def __init__(self, depth: float = 1000, density: float = 1025) -> None:
        """Initialise self."""
        self.__depth = depth
        self.__density = density

    @property
    def density(self) -> float:
        """Ocean density in kg m**-3."""
        return self.__density

    @property
    def depth(self) -> float:
        """Ocean depth in m."""
        return self.__depth


class DiscreteSpectrum:
    def __init__(self, amplitudes, frequencies, phases=0, betas=0):

        # np.ravel force precisely 1D-arrays
        # Promote the map to list so the iterator can be used several times
        args = list(map(np.ravel, (amplitudes, frequencies, phases, betas)))
        (size,) = np.broadcast_shapes(*(arr.shape for arr in args))

        if size != 1:
            for i, arr in enumerate(args):
                if arr.size == 1:
                    args[i] = itertools.repeat(arr[0], size)

        self.__waves = [
            Wave(_a, frequency=_f, phase=_ph, beta=_b) for _a, _f, _ph, _b in zip(*args)
        ]

    @property
    def waves(self):
        return self.__waves


class _GenericSpectrum:
    def __init__(
        self,
        *,
        u=None,
        swh=None,
        peak_period=None,
        peak_frequency=None,
        peak_wavelength=None,
    ):
        if u is not None:
            values = SpecVars(u)
            for k, v in zip(
                (
                    "swh",
                    "peak_period",
                    "peak_frequency",
                    "peak_wavenumber",
                    "peak_wavelength",
                ),
                values,
            ):
                setattr(self, f"__{k}", v)
        else:
            ...

    @property
    def waves(self):
        return self.__waves


# def SpecVars(u=10, v=False):
#     Tp = 2 * np.pi * u / (0.877 * g)
#     fp = 1 / Tp
#     Hs = 0.0246 * u**2
#     k = (2 * np.pi * fp)**2 / g
#     wl = 2 * np.pi / k

#     if v:
#         print(f'For winds of {u:.2f}m/s, expected waves are {Hs:.2f}m high,\n'
#               f'with a period of {Tp:.2f}s, corresponding to a frequency of {fp:.2f}Hz,\n'
#               f'and wavenumber of {k:.2f}/m or wavelength of {wl:.2f}m.')
#     else:
#         return(Hs, Tp, fp, k, wl)


class PiersonMoskowitz(_GenericSpectrum):
    def __init__(
        self,
        *,
        wind_speed=None,
        swh=None,
        peak_frequency=None,
        peak_period=None,
        peak_wavelength=None,
    ):
        kwargs = locals()
        print(kwargs)
        kwargs.pop("self")
        keys = list(kwargs.keys())

        def filter_dict(dct):
            return (
                item[0]
                for item in filter(lambda item: item[1] is not None, kwargs.items())
            )

        if wind_speed is not None:
            self._init_from_u(kwargs.pop("wind_speed"))
            for k in filter_dict(kwargs):
                warnings.warn(
                    f"Wind speed was specified, {k} will be ignored", stacklevel=1
                )
        elif swh is not None:
            for k in keys[:1]:
                kwargs.pop(k)
            self._init_from_swh(kwargs.pop("swh"))
            for k in filter_dict(kwargs):
                warnings.warn(
                    f"Significant wave height was specified, {k} will be ignored",
                    stacklevel=1,
                )
        elif peak_frequency is not None:
            for k in keys[:2]:
                kwargs.pop(k)
            self._init_from_frequency(kwargs.pop("peak_frequency"))
            for k in filter_dict(kwargs):
                warnings.warn(
                    f"Peak frequency was specified, {k} will be ignored",
                    stacklevel=1,
                )
        elif peak_period is not None:
            for k in keys[:3]:
                kwargs.pop(k)
            self._init_from_period(kwargs.pop("peak_period"))
            for k in filter_dict(kwargs):
                warnings.warn(
                    f"Peak frequency was specified, {k} will be ignored",
                    stacklevel=1,
                )
        elif peak_wavelength is not None:
            for k in keys[:4]:
                kwargs.pop(k)
            self._init_from_frequency(kwargs.pop("peak_wavelength"))
        else:
            raise ValueError("At least one spectral parameter has to be provided")

    @property
    def wind_speed(self):
        return self.__wind_speed

    @property
    def swh(self):
        return self.__swh

    @property
    def peak_frequency(self):
        return self.__peak_frequency

    @property
    def peak_period(self):
        return self.__peak_period

    @property
    def peak_wavelength(self):
        return self.__peak_wavelength

    def _init_from_u(self, wind_speed):
        self.__wind_speed = wind_speed
        self.__swh = 0.0246 * wind_speed**2
        self.__peak_frequency = 0.877 * g / (2 * PI * wind_speed)
        self.__peak_period = 1 / self.peak_frequency

    def __call__(self, frequency):
        alpha_s = 0.2044
        beta_s = 1.2500

        frequency = np.asarray(frequency)

        return (
            alpha_s
            * self.swh**2
            * (self.peak_frequency**4 / frequency**5)
            * np.exp(-beta_s * (self.peak_frequency / frequency) ** 4)
        )


class Spectrum:
    def __init__(
        self,
        *,
        u=None,
        swh=None,
        peak_period=None,
        peak_frequency=None,
        peak_wavelength=None,
        beta=0,
        phi=np.nan,
        df=1.1,
        x=-5,
        y=16,
        n=2,
    ):
        # if u is None or (
        swh, peak_period, peak_frequency, peak_wavenumber, peak_wavelength = SpecVars(
            u
        )  # Parameters for a typical spectrum
        spec = SpecType
        tfac = tail_fac

        f = peak_frequency * df ** np.arange(x, y)

        for key, value in kwargs.items():
            if key == "u":
                u = value
                swh, peak_period, peak_frequency, peak_wavenumber, peak_wavelength = (
                    SpecVars(u)
                )
                f = peak_frequency * df ** np.arange(x, y)
            elif key == "Hs":
                swh = value
            elif key == "n0":
                swh = (2**1.5) * value
            elif key == "fp":
                peak_frequency = value
                peak_period = 1 / peak_frequency
                peak_wavenumber = (2 * np.pi * peak_frequency) ** 2 / g
                peak_wavelength = 2 * np.pi / peak_wavenumber
            elif key == "Tp":
                peak_period = value
                peak_frequency = 1 / peak_period
                peak_wavenumber = (2 * np.pi * peak_frequency) ** 2 / g
                peak_wavelength = 2 * np.pi / peak_wavenumber
            elif key == "wlp":
                peak_wavelength = value
                peak_wavenumber = 2 * np.pi / peak_wavelength
                peak_frequency = (g * peak_wavenumber) ** 0.5 / (2 * np.pi)
                peak_period = 1 / peak_frequency
            elif key == "beta":
                beta = value
            elif key == "phi":
                phi = value
            elif key == "df":
                df = value
                fac = np.log(1.1) / np.log(df)
                x = -np.ceil(5 * fac)
                y = np.ceil(15 * fac) + 1
            elif key == "spec" or key == "SpecType":
                spec = value
            elif key == "f":
                f = value
            elif key == "n":
                n = value
            elif key == "tail_fac":
                tfac = value
            else:
                print(f"Unknow input: {key}")

        self.type = "WaveSpec"
        self.SpecType = spec
        self.Hs = swh
        self.Tp = peak_period
        self.fp = peak_frequency
        self.kp = peak_wavenumber
        self.wlp = peak_wavelength
        self.beta = beta
        self.tail_fac = tfac

        if len(f) == 1 or spec == "Mono":
            self.f = np.array([peak_frequency])
            df_vec = np.array([1])
            self.Ei = np.array([swh**2 / 16])
        elif spec == "PowerLaw":
            if n > 0:
                self.f = peak_frequency * df ** (np.arange(-f.size, 0) + 1)
            else:
                self.f = peak_frequency * df ** (np.arange(0, f.size) + 1)
            df_vec = np.empty_like(f)
            df_vec[0] = f[1] - f[0]
            df_vec[1:-1] = (f[2:] - f[:-2]) / 2
            df_vec[-1] = f[-1] - f[-2]
            self.Ei = PowerLaw(swh, peak_frequency, self.f, df_vec, n)
        else:
            self.f = f
            df_vec = np.empty_like(f)
            df_vec[0] = f[1] - f[0]
            df_vec[1:-1] = (f[2:] - f[:-2]) / 2
            df_vec[-1] = f[-1] - f[-2]
            if spec == "JONSWAP":
                self.Ei = Jonswap(swh, peak_frequency, f)
            elif spec == "PM":
                self.Ei = PM(u, f)
            else:
                raise ValueError(f"Unknown spectrum type: {spec}")

        self.nf = f.size
        self.k = (2 * np.pi * self.f) ** 2 / g
        self.cgw = 0.5 * (g / self.k) ** 0.5
        self.nf = len(self.f)
        self.df = df_vec

        if type(phi) == np.ndarray:
            self.phi = phi
        else:
            self.phi = phi * np.ones_like(self.f)

        self.setWaves()
        self.af = [0] * self.nf


class OceanCoupled(Ocean):
    """Extend `Ocean` to include wave-dependent quantities."""

    def __init__(self, ocean: Ocean, ds: DiscreteSpectrum, gravity: float):
        super().__init__(ocean.depth, ocean.density)
        self.__wavenumbers = self.compute_wavenumbers(ds, gravity)

    @property
    def wavenumbers(self) -> np.ndarray:
        """Array of wave numbers in rad m**-1"""
        return self.__wavenumbers

    @functools.cached_property
    def wavelengths(self) -> np.ndarray:
        """Wavelengths in m"""
        return 2 * PI / self.wavenumbers

    def _comp_wns(self, angfreqs2: np.ndarray, gravity: float) -> np.ndarray:
        def f(kk: float, alpha: float) -> float:
            # Dispersion relation (form f(k) = 0), for a free surface,
            # admitting one positive real root.
            return kk * np.tanh(kk) - alpha

        def df_dk(kk: float, alpha: float) -> float:
            # Derivative of dr with respect to kk.
            tt = np.tanh(kk)
            return tt + kk * (1 - tt**2)

        alphas = angfreqs2 / gravity
        if np.isposinf(self.depth):
            return alphas

        alphas *= self.depth
        roots = np.full(len(alphas), np.nan)
        for i, alpha in enumerate(alphas):
            res = optimize.root_scalar(f, (alpha,), fprime=df_dk, x0=alpha)
            if res.converged:
                roots[i] = res.root
            else:
                print("ohno!")

        return roots / self.depth

    def compute_wavenumbers(self, ds: DiscreteSpectrum, gravity: float) -> np.ndarray:
        """ """
        return self._comp_wns(
            np.array([wave.angular_frequency2 for wave in ds.waves]), gravity
        )


class WaveSpectrum:
    pass


class Domain:
    """"""

    def __init__(
        self,
        spectrum: WaveSpectrum,
        ocean: Ocean,
        nf,
        phases,
        betas,
        fmin,
        fmax,
        frequencies,
    ) -> None:
        """"""
        self.__spectrum = spectrum
        self.__ocean = OceanCoupled(ocean, spectrum)

        self.__frozen_spectrum = DiscreteSpectrum(
            spectrum.amplitude(frequencies), frequencies, phases, betas
        )
        # TODO: doit avoir un attribut gravité si le spectre n'en a pas
