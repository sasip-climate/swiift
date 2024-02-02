#!/usr/bin/env python3


import functools
import warnings
import numpy as np

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
        beta: float | None = None,
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
        else:
            self.__frequency = frequency

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
    """Extend `Ocean` to include wave-dependent quantities.

    Parameters
    ----------
    ocean : Ocean
        `Ocean` instance to extend.
    inner_products : 1D array-like of float
        Inner products of the vertical modes in m.
    wave_numbers : 1D array-like of complex
        Wave numbers in rad m**-1.

    Attributes
    ----------
    inner_products : 1D array-like of complex
    wavelength : float
    wave_numbers : 1D array-like of complex

    """

    def __init__(self, ocean: Ocean, spectrum: Spectrum) -> None:
        """Initialise self."""
        super().__init__(ocean.depth, ocean.density)
        alpha = wave._c_alpha
        self.__wave_numbers = self.compute_wave_numbers(alpha, nk)
        self.__inner_products = self.compute_inner_products(alpha)

    @property
    def inner_products(self) -> np.ndarray:
        """Array of inner products of the vertical modes with themselves in m.

        An inner product is defined on the space spawned by the wave modes.
        This modes are orthogonal, with respect to that product, hence there is
        as many non-null products as there is modes.
        """
        return self.__inner_products

    @property
    def wave_numbers(self) -> np.ndarray:
        """Array of wave numbers in rad m**-1.

        Index 0 is real.
        Index 1 onwards are imaginary.
        """
        return self.__wave_numbers

    @functools.cached_property
    def wavelength(self) -> float:
        """Wavelength in m.

        It corresponds to the only real wave number.
        """
        return 2 * PI / np.real(self.wave_numbers[0])

    def compute_inner_products(self, alpha: float) -> np.ndarray:
        """Compute the inner product associated with each mode.

        Parameters
        ----------
        alpha : float
            Deep water positive, real wave number.
        wave_numbers : 1D array-like of complex
            The actual wave numbers.

        Returns
        -------
        1D array-like of float
            The computed inner products.

        """
        ip = (self.depth * (self.wave_numbers**2 - alpha**2) + alpha) / (
            2 * self.wave_numbers**2
        )
        assert np.allclose(np.imag(ip), 0)
        return np.real(ip)

    def compute_wave_numbers(self, alpha: float, nk: int) -> np.ndarray:
        """Determine nk+1 wave numbers.

        Parameters
        ----------
        alpha : float
            Deep water positive, real wave number.
        nk : int
            Number of evanescent modes to compute.

        Returns
        -------
        1D array-like of complex
            The computed wave-numbers.

        """
        alpha *= self.depth  # rescaling
        roots = np.empty(nk + 1, complex)

        def dr(kk: float) -> float:
            # Dispersion relation (form f(k) = 0), for a free surface,
            # admitting one positive real root.
            return kk * np.tanh(kk) - alpha

        def dr_i(kk: float) -> float:
            # Dispersion relation (form f(k) = 0) for a free surface,
            # admitting an infinity of positive real roots.
            return kk * np.sin(kk) + alpha * np.cos(kk)

        def dr_d(kk: float) -> float:
            # Derivative of dr with respect to kk.
            tt = np.tanh(kk)
            return tt + kk * (1 - tt**2)

        roots[0] = newton(dr, fprime=dr_d, x0=alpha, tol=1e-14)
        assert roots[0] > 0
        roots[1:] = [
            1j * brentq(dr_i, m * PI, (m + 1) * PI, xtol=1e-14) for m in range(nk)
        ]
        return roots / self.depth


class WaveSpectrum:
    pass


class Domain:
    """"""

    def __init__(self, spectrum: WaveSpectrum, ocean: Ocean) -> None:
        """"""
        self.__spectrum = spectrum
        self.__ocean = OceanCoupled(ocean, spectrum)
