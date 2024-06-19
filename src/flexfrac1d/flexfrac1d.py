#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Sequence
import copy
import functools
import itertools
from numbers import Real
import warnings
import numpy as np
import scipy.optimize as optimize
import scipy.signal as signal
from sortedcontainers import SortedList

# from .libraries.WaveUtils import SpecVars
from .lib.displacement import displacement
from .lib.curvature import curvature
from .lib.energy import energy
from .lib.numerical import free_surface
from .lib.graphics import plot_displacement
from .pars import g
from .lib.constants import PI, PI_2


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
        return 2 * PI * self.frequency

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


# TODO: look at @dataclass
class Ice:
    def __init__(
        self,
        density: float = 922.5,
        frac_toughness: float = 1e5,
        poissons_ratio: float = 0.3,
        thickness: float = 1.0,
        youngs_modulus: float = 6e9,
    ):
        self.__density = density
        self.__frac_toughness = frac_toughness
        self.__poissons_ratio = poissons_ratio
        self.__thickness = thickness
        self.__youngs_modulus = youngs_modulus

    def __hash__(self) -> int:
        return hash(
            (
                self.density,
                self.frac_toughness,
                self.poissons_ratio,
                self.thickness,
                self.youngs_modulus,
            )
        )

    @property
    def density(self):
        return self.__density

    @property
    def frac_toughness(self) -> float:
        """Ice fracture toughness in Pa m**1/2

        Returns
        -------
        frac_toughness: float

        """
        return self.__frac_toughness

    @property
    def poissons_ratio(self):
        return self.__poissons_ratio

    @property
    def thickness(self):
        return self.__thickness

    @property
    def youngs_modulus(self):
        return self.__youngs_modulus

    @functools.cached_property
    def quad_moment(self):
        return self.thickness**3 / (12 * (1 - self.poissons_ratio**2))

    @functools.cached_property
    def flex_rigidity(self):
        return self.quad_moment * self.youngs_modulus

    @functools.cached_property
    def frac_energy_rate(self) -> float:
        """Ice fracture energy release rate in J m**-2

        Returns
        -------
        frac_energy_rate: float

        """
        return (
            (1 - self.poissons_ratio**2) * self.frac_toughness**2 / self.youngs_modulus
        )


class IceCoupled(Ice):
    def __init__(
        self,
        ice: Ice,
        ocean: OceanCoupled,
        spec: DiscreteSpectrum,
        dispersion: str,
        gravity: float,
    ):
        super().__init__(
            ice.density,
            ice.frac_toughness,
            ice.poissons_ratio,
            ice.thickness,
            ice.youngs_modulus,
        )
        if not (dispersion is None or dispersion != ""):
            warnings.warn(
                "Dispersion is ignored for now and is always ElML", stacklevel=1
            )
        self.__draft = self.density / ocean.density * self.thickness
        self.__dud = ocean.depth - self.draft
        self._elastic_length_pow4 = self.flex_rigidity / (ocean.density * gravity)
        self.__elastic_length = self._elastic_length_pow4 ** (1 / 4)
        self.__wavenumbers = self.compute_wavenumbers(ocean, spec, gravity)

    @property
    def draft(self):
        return self.__draft

    @property
    def dud(self):
        return self.__dud

    @property
    def elastic_length(self):
        return self.__elastic_length

    @property
    def wavenumbers(self):
        return self.__wavenumbers

    @functools.cached_property
    def _c_wavenumbers(self):
        return self.wavenumbers + 1j * self.attenuations

    @functools.cached_property
    def freeboard(self):
        return self.thickness - self.draft

    @functools.cached_property
    def attenuations(self):
        return self.wavenumbers**2 * self.thickness / 4

    @functools.cached_property
    def _red_elastic_number(self):
        return 1 / (2**0.5 * self.elastic_length)

    def compute_wavenumbers(
        self, ocean: OceanCoupled, ds: DiscreteSpectrum, gravity: float
    ) -> np.ndarray:
        return self._comp_wns(ocean.density, ds._ang_freq2, gravity)

    def _comp_wns(
        self, density: float, angfreqs2: np.ndarray, gravity: float
    ) -> np.ndarray:
        def f(kk: float, d0: float, d1: float, rr: float) -> float:
            obj = (kk**5 + d1 * kk) * np.tanh(rr * kk) + d0
            return obj

        def df_dk(kk: float, d0: float, d1: float, rr: float) -> float:
            return (5 * kk**4 + d1 + rr * d0) * np.tanh(rr * kk) + rr * (
                kk**5 + d1 * kk
            )

        def extract_real_root(roots):
            mask = (np.imag(roots) == 0) & (np.real(roots) > 0)
            if mask.nonzero()[0].size != 1:
                raise ValueError("An approximate initial guess could not be found")
            return np.real(roots[mask][0])

        def find_k(k0, alpha, d0, d1, rr):
            res = optimize.root_scalar(
                f,
                args=(d0, d1, rr),
                fprime=df_dk,
                x0=k0,
                xtol=1e-10,
            )
            if not res.converged:
                warnings.warn(
                    f"Root finding did not converge: ice-covered surface, "
                    f"f={np.sqrt(alpha*gravity)/(2*PI):1.2g} Hz",
                    stacklevel=2,
                )
            return res.root

        scaled_ratio = self.dud / self.elastic_length

        alphas = angfreqs2 / gravity
        deg1 = 1 - alphas * self.draft
        deg0 = -alphas * self.elastic_length
        roots = np.full(angfreqs2.size, np.nan)

        for i, (alpha, _d0, _d1) in enumerate(zip(alphas, deg0, deg1)):
            find_k_i = functools.partial(
                find_k,
                alpha=alpha,
                d0=_d0,
                d1=_d1,
                rr=scaled_ratio,
            )

            # We always expect one positive real root,
            # and if _d1 < 0, eventually two additional negative real roots.
            roots_dw = np.polynomial.polynomial.polyroots([_d0, _d1, 0, 0, 0, 1])
            k0_dw = extract_real_root(roots_dw)
            if np.isposinf(self.dud):
                roots[i] = k0_dw
                continue
            # Use a DW initial guess if |1-1/tanh(rr*k_DW)| < 0.15
            # Use a SW initial guess if |1-rr*k_SW/tanh(rr*k_SW)| < 0.20
            thrsld_dw, thrsld_sw = 1.33, 0.79
            if scaled_ratio * k0_dw > thrsld_dw:
                roots[i] = find_k_i(k0_dw)
            else:
                roots_sw = np.polynomial.polynomial.polyroots(
                    [_d0 / scaled_ratio, 0, _d1, 0, 0, 0, 1]
                )
                k0_sw = extract_real_root(roots_sw)

                if scaled_ratio * k0_sw < thrsld_sw:
                    roots[i] = find_k_i(k0_sw)
                # Use an initial guess in the middle otherwise
                else:
                    k0_ = (k0_sw + k0_dw) / 2
                    roots[i] = find_k_i(k0_)

        return roots / self.elastic_length


@functools.total_ordering
class Floe:
    def __init__(
        self,
        left_edge: float,
        length: float,
        ice: Ice = None,
        dispersion: str = "ElML",
    ):
        self.__left_edge = left_edge
        self.__length = length
        self.__ice = ice

    def __eq__(self, other: [Floe, Real]) -> bool:
        match other:
            case Floe():
                return self.left_edge == other.left_edge
            case Real():
                return self.left_edge == other
            case _:
                raise NotImplementedError

    def __lt__(self, other: Floe) -> bool:
        match other:
            case Floe():
                return self.left_edge < other.left_edge
            case Real():
                return self.left_edge < other
            case _:
                raise NotImplementedError

    @property
    def left_edge(self):
        return self.__left_edge

    @functools.cached_property
    def right_edge(self):
        return self.left_edge + self.length

    @property
    def length(self):
        return self.__length

    @property
    def ice(self) -> Ice:
        return self.__ice

    @property
    def dispersion(self):
        return self.__dispersion


class FloeCoupled(Floe):
    def __init__(
        self,
        floe: Floe,
        ice: IceCoupled,
        phases: np.ndarray | list[float] | float,
        amp_coefficients: np.ndarray | list[float] | float,
        gen: int = 0,
        dispersion=None,
    ):
        super().__init__(floe.left_edge, floe.length, ice, dispersion)
        self.phases = np.asarray(phases)  # no dunder: uses the setter method
        self.__amp_coefficients = amp_coefficients
        self.__gen = gen

    @property
    def phases(self) -> np.ndarray:
        return self.__phases

    @phases.setter
    def phases(self, value):
        self.__phases = np.asarray(value) % PI_2

    @property
    def amp_coefficients(self):
        return self.__amp_coefficients

    @property
    def gen(self) -> int:
        return self.__gen

    @property
    def ice(self) -> IceCoupled:
        return self._Floe__ice

    @functools.cached_property
    def _adim(self):
        return self.length * self.ice._red_elastic_number

    def _pack(
        self, spectrum: DiscreteSpectrum
    ) -> tuple[tuple[float], tuple[np.ndarray]]:
        return (self.ice._red_elastic_number, self.length), (
            self.amp_coefficients * spectrum._amps,
            self.ice._c_wavenumbers,
            self.phases,
        )

    def forcing(self, x, spectrum, growth_params):
        return free_surface(x, self._pack(spectrum)[1], growth_params)

    def displacement(self, x, spectrum, growth_params, an_sol, num_params):
        """Complete solution of the displacement ODE

        `x` is expected to be relative to the floe, i.e. to be bounded by 0, L
        """
        return displacement(x, *self._pack(spectrum), growth_params, an_sol, num_params)

    def curvature(self, x, spectrum, growth_params, an_sol, num_params):
        """Curvature of the floe, i.e. second derivative of the vertical displacement"""
        return curvature(x, *self._pack(spectrum), growth_params, an_sol, num_params)

    def energy(self, spectrum: DiscreteSpectrum, growth_params, an_sol, num_params):
        factor = self.ice.flex_rigidity / (2 * self.ice.thickness)
        unit_energy = energy(*self._pack(spectrum), growth_params, an_sol, num_params)
        return factor * unit_energy
        # In case of a numerical solution, the result is the output of
        # integrate.quad, that is a (solution, bound on error) tuple.
        # We do not do anything with the latter at the moment.
        # return factor * unit_energy[0]

    def search_fracture(
        self, spectrum: DiscreteSpectrum, growth_params, an_sol, num_params
    ):
        return self.binary_fracture(spectrum, growth_params, an_sol, num_params)

    def binary_fracture(
        self, spectrum: DiscreteSpectrum, growth_params, an_sol, num_params
    ) -> float | None:
        coef_nd = 4
        base_energy = self.energy(spectrum, growth_params, an_sol, num_params)
        # No fracture if the elastic energy is below the threshold
        if base_energy < self.ice.frac_energy_rate:
            return None
        else:
            nd = (
                np.ceil(
                    4 * self.length * self.ice.wavenumbers.max() / (2 * np.pi)
                ).astype(int)
                + 2
            )
            lengths = np.linspace(0, self.length, nd * coef_nd)[1:-1]
            ener = np.full(lengths.shape, np.nan)
            for i, length in enumerate(lengths):
                ener[i] = self.ener_min(
                    length, spectrum, growth_params, an_sol, num_params
                )

            peak_idxs = np.hstack(
                (0, signal.find_peaks(np.log(ener), distance=2)[0], ener.size - 1)
            )

            # TODO: temp func for ener_min and its arguments
            opts = [
                optimize.minimize_scalar(
                    lambda length: self.ener_min(
                        length, spectrum, growth_params, an_sol, num_params
                    ),
                    bounds=lengths[peak_idxs[[i, i + 1]]],
                )
                for i in range(len(peak_idxs) - 1)
            ]
            opt = min(filter(lambda opt: opt.success, opts), key=lambda opt: opt.fun)
            # Minimisation is done on the log of energy
            if np.exp(opt.fun) + self.ice.frac_energy_rate < base_energy:
                return opt.x
            else:
                return None

    def ener_min(self, length, spectrum, growth_params, an_sol, num_params) -> float:
        floe_l = Floe(left_edge=self.left_edge, length=length)
        cf_l = FloeCoupled(floe_l, self.ice, self.phases, self.amp_coefficients)

        floe_r = Floe(left_edge=self.left_edge + length, length=self.length - length)
        phases_r = cf_l.phases + self.ice.wavenumbers * floe_l.length
        cf_r = FloeCoupled(
            floe_r,
            self.ice,
            phases_r,
            self.amp_coefficients * np.exp(-self.ice.attenuations * cf_l.length),
        )
        growth_params_r = (
            (growth_params[0] - length, growth_params[1])
            if growth_params is not None
            else None
        )

        en_l, en_r = (
            _f.energy(spectrum, _gp, an_sol, num_params)
            for _f, _gp in zip((cf_l, cf_r), (growth_params, growth_params_r))
        )
        return np.log(en_l + en_r)

    def fracture(
        self, xfs: np.ndarray | float
    ) -> tuple[FloeCoupled, list[FloeCoupled]]:
        xfs = np.asarray(xfs) + self.left_edge  # domain reference frame
        left_edges = np.hstack((self.left_edge, xfs))
        right_edges = np.hstack((xfs, self.right_edge))
        lengths = right_edges - left_edges
        phases = np.vstack(
            (self.phases, self.ice.wavenumbers * lengths[:-1, None])
        ).cumsum(axis=0)
        amp_coefficients = np.exp(
            np.vstack(
                (
                    np.log(self.amp_coefficients),
                    -self.ice.attenuations * lengths[:-1, None],
                )
            ).cumsum(axis=0)
        )

        # TODO instead of instantiating FloeCoupled objects, return iterators
        # on the parameters, so that the phases can be altered before
        # instantiation and the need for a setter can be removed
        return self, [
            FloeCoupled(Floe(left_edge, length), self.ice, phases_, coefs_, gen_)
            for left_edge, length, phases_, coefs_, gen_ in zip(
                left_edges, lengths, phases, amp_coefficients, (self.gen + 1, self.gen)
            )
        ]


class DiscreteSpectrum:
    def __init__(
        self,
        amplitudes,
        frequencies,
        phases=0,
        betas=0,
    ):

        # np.ravel to force precisely 1D-arrays
        # Promote the map to list so the iterator can be used several times
        args = list(map(np.ravel, (amplitudes, frequencies, phases, betas)))
        (size,) = np.broadcast_shapes(*(arr.shape for arr in args))

        # TODO: sort waves by frequencies or something
        # TODO: sanity checks on nan, etc. that could be returned
        #       by the Spectrum objects

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

    @functools.cached_property
    def _ang_freq2(self):
        return np.asarray([wave.angular_frequency2 for wave in self.waves])

    @functools.cached_property
    def _amps(self):
        return np.asarray([wave.amplitude for wave in self.waves])

    @functools.cached_property
    def _freqs(self):
        return np.asarray([wave.frequency for wave in self.waves])

    @functools.cached_property
    def _ang_freqs(self):
        return np.asarray([wave.angular_frequency for wave in self.waves])

    @functools.cached_property
    def _phases(self):
        return np.asarray([wave.phase for wave in self.waves])

    @functools.cached_property
    def nf(self):
        return len(self.waves)


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
        # if u is not None:
        #     values = SpecVars(u)
        #     for k, v in zip(
        #         (
        #             "swh",
        #             "peak_period",
        #             "peak_frequency",
        #             "peak_wavenumber",
        #             "peak_wavelength",
        #         ),
        #         values,
        #     ):
        #         setattr(self, f"__{k}", v)
        # else:
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
#               f'with a period of {Tp:.2f}s, '
#               f'corresponding to a frequency of {fp:.2f}Hz,\n'
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


# class Spectrum:
#     def __init__(
#         self,
#         *,
#         u=None,
#         swh=None,
#         peak_period=None,
#         peak_frequency=None,
#         peak_wavelength=None,
#         beta=0,
#         phi=np.nan,
#         df=1.1,
#         x=-5,
#         y=16,
#         n=2,
#     ):
#         # if u is None or (
#         swh, peak_period, peak_frequency, peak_wavenumber, peak_wavelength = SpecVars(
#             u
#         )  # Parameters for a typical spectrum
#         spec = SpecType
#         tfac = tail_fac

#         f = peak_frequency * df ** np.arange(x, y)

#         for key, value in kwargs.items():
#             if key == "u":
#                 u = value
#                 swh, peak_period, peak_frequency, peak_wavenumber, peak_wavelength = (
#                     SpecVars(u)
#                 )
#                 f = peak_frequency * df ** np.arange(x, y)
#             elif key == "Hs":
#                 swh = value
#             elif key == "n0":
#                 swh = (2**1.5) * value
#             elif key == "fp":
#                 peak_frequency = value
#                 peak_period = 1 / peak_frequency
#                 peak_wavenumber = (2 * np.pi * peak_frequency) ** 2 / g
#                 peak_wavelength = 2 * np.pi / peak_wavenumber
#             elif key == "Tp":
#                 peak_period = value
#                 peak_frequency = 1 / peak_period
#                 peak_wavenumber = (2 * np.pi * peak_frequency) ** 2 / g
#                 peak_wavelength = 2 * np.pi / peak_wavenumber
#             elif key == "wlp":
#                 peak_wavelength = value
#                 peak_wavenumber = 2 * np.pi / peak_wavelength
#                 peak_frequency = (g * peak_wavenumber) ** 0.5 / (2 * np.pi)
#                 peak_period = 1 / peak_frequency
#             elif key == "beta":
#                 beta = value
#             elif key == "phi":
#                 phi = value
#             elif key == "df":
#                 df = value
#                 fac = np.log(1.1) / np.log(df)
#                 x = -np.ceil(5 * fac)
#                 y = np.ceil(15 * fac) + 1
#             elif key == "spec" or key == "SpecType":
#                 spec = value
#             elif key == "f":
#                 f = value
#             elif key == "n":
#                 n = value
#             elif key == "tail_fac":
#                 tfac = value
#             else:
#                 print(f"Unknow input: {key}")

#         self.type = "WaveSpec"
#         self.SpecType = spec
#         self.Hs = swh
#         self.Tp = peak_period
#         self.fp = peak_frequency
#         self.kp = peak_wavenumber
#         self.wlp = peak_wavelength
#         self.beta = beta
#         self.tail_fac = tfac

#         if len(f) == 1 or spec == "Mono":
#             self.f = np.array([peak_frequency])
#             df_vec = np.array([1])
#             self.Ei = np.array([swh**2 / 16])
#         elif spec == "PowerLaw":
#             if n > 0:
#                 self.f = peak_frequency * df ** (np.arange(-f.size, 0) + 1)
#             else:
#                 self.f = peak_frequency * df ** (np.arange(0, f.size) + 1)
#             df_vec = np.empty_like(f)
#             df_vec[0] = f[1] - f[0]
#             df_vec[1:-1] = (f[2:] - f[:-2]) / 2
#             df_vec[-1] = f[-1] - f[-2]
#             self.Ei = PowerLaw(swh, peak_frequency, self.f, df_vec, n)
#         else:
#             self.f = f
#             df_vec = np.empty_like(f)
#             df_vec[0] = f[1] - f[0]
#             df_vec[1:-1] = (f[2:] - f[:-2]) / 2
#             df_vec[-1] = f[-1] - f[-2]
#             if spec == "JONSWAP":
#                 self.Ei = Jonswap(swh, peak_frequency, f)
#             elif spec == "PM":
#                 self.Ei = PM(u, f)
#             else:
#                 raise ValueError(f"Unknown spectrum type: {spec}")

#         self.nf = f.size
#         self.k = (2 * np.pi * self.f) ** 2 / g
#         self.cgw = 0.5 * (g / self.k) ** 0.5
#         self.nf = len(self.f)
#         self.df = df_vec

#         if type(phi) == np.ndarray:
#             self.phi = phi
#         else:
#             self.phi = phi * np.ones_like(self.f)

#         self.setWaves()
#         self.af = [0] * self.nf


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

        coefs_d0 = alphas * self.depth
        roots = np.full(len(coefs_d0), np.nan)
        for i, _d0 in enumerate(coefs_d0):
            if _d0 >= np.arctanh(np.nextafter(1, 0)):
                roots[i] = _d0
                continue
            res = optimize.root_scalar(f, (_d0,), fprime=df_dk, x0=_d0)
            if not res.converged:
                warnings.warn(
                    f"Root finding did not converge: free surface, "
                    f"f={np.sqrt(_d0/self.depth*gravity)/(2*PI):1.2g} Hz",
                    stacklevel=2,
                )
            roots[i] = res.root

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
        gravity,
        spectrum: WaveSpectrum,
        ocean: Ocean,
        growth_mean=None,
        growth_std=None,
    ) -> None:
        """"""
        self.__gravity = gravity
        self.__frozen_spectrum = spectrum
        self.__ocean = OceanCoupled(ocean, spectrum, gravity)
        self.__floes = SortedList()
        self.__ices = {}

        if growth_mean is None:
            if growth_std is not None:
                growth_mean = np.zeros((self.spectrum.nf, 1))
        else:
            growth_mean = np.asarray(growth_mean)
            if growth_mean.size == 1:
                # As `broadcast_to` returns a view,
                # copying is necessary to obtain a mutable array
                growth_mean = np.broadcast_to(growth_mean, (self.spectrum.nf, 1)).copy()
            if growth_std is None:
                growth_std = (
                    2 * np.pi / self.ocean.wavenumbers[self.spectrum._amps.argmax()]
                )
        self.__growth_mean = growth_mean
        self.__growth_std = growth_std

        # TODO: callable spectrum
        # self.__frozen_spectrum = DiscreteSpectrum(
        #     spectrum.amplitude(frequencies), frequencies, phases, betas
        # )

    @property
    def floes(self) -> SortedList[FloeCoupled]:
        return self.__floes

    @property
    def gravity(self) -> float:
        return self.__gravity

    @property
    def ices(self) -> dict[Ice, IceCoupled]:
        return self.__ices

    @property
    def ocean(self) -> OceanCoupled:
        return self.__ocean

    @property
    def spectrum(self) -> DiscreteSpectrum:
        return self.__frozen_spectrum

    @property
    def growth_mean(self):
        return self.__growth_mean

    @growth_mean.setter
    def growth_mean(self, value: np.ndarray):
        self.__growth_mean = value

    @property
    def growth_std(self):
        return self.__growth_std

    def _pack_growth(self, floe):
        if self.growth_mean is None:
            return None
        return self.growth_mean - floe.left_edge, self.growth_std

    def _init_from_f(self): ...

    def _couple_ice(self, ice):
        self.ices[ice] = IceCoupled(ice, self.ocean, self.spectrum, None, self.gravity)

    def _shift_phases(self, phases: np.ndarray):
        for i in range(len(self.floes)):
            self.floes[i].phases -= phases

    def _shift_growth_means(self, phases: np.ndarray):
        # TODO: refine to take into account subdomain transitions
        # and floes with variying properties
        mask = self.growth_mean < self.floes[0].left_edge
        if mask.any():
            self.growth_mean[mask] += (
                phases[mask[:, 0]] / self.ocean.wavenumbers[mask[:, 0]]
            )
        if not mask.all():
            self.growth_mean[~mask] += (
                phases[~mask[:, 0]] / self.floes[0].ice.wavenumbers[~mask[:, 0]]
            )

    def iterate(self, delta_time: float):
        # NOTE: delta_time is likely to not change between calls. The computed
        # phases could be cached, but the cost of the computation seems
        # independent of the size of the array up to about size := 100--500. It
        # is not advisable to cache methods, and the array of angular
        # frequencies would have to be cast to (for example) a tuple before
        # being passed to a cached function, as arrays are not hashable. The
        # cost of a back-and-forth cast is tenfold the cost of the product for
        # size := 100; more than fiftyfold for size := 1000.
        phase_shifts = delta_time * self.spectrum._ang_freqs
        self._shift_phases(phase_shifts)
        if self.growth_mean is not None:
            # Phases are only modulo'd in the setter
            self._shift_growth_means(phase_shifts)

    def _pop_c_floe(self, floe: FloeCoupled):
        self.floes.remove(floe)

    def _add_c_floes(self, floes: list[FloeCoupled]):
        # It is assume no overlap will occur, and phases have been properly
        # set, as these method should only be called after a fracture event
        self.floes.update(floes)

    def breakup(self, an_sol=None, num_params=None):
        dct = {}
        for i, floe in enumerate(self.floes):
            xf = floe.search_fracture(
                self.spectrum,
                self._pack_growth(floe),
                an_sol,
                num_params,
            )
            if xf is not None:
                dct[i] = floe.fracture(xf)
        for old, new in dct.values():
            self._pop_c_floe(old)
            self._add_c_floes(new)

    def plot(
        self,
        resolution: float,
        left_bound: float,
        ax=None,
        an_sol=None,
        add_surface=True,
        base=0,
        kw_dis=None,
        kw_sur=None,
    ):
        # TODO moved to Experiment
        plot_displacement(
            resolution, self, left_bound, ax, an_sol, add_surface, base, kw_dis, kw_sur
        )


class Experiment:
    def __init__(self, domain: Domain, floes: Floe | Sequence[Floe]):
        self.__time = 0
        self.__domain = domain
        match floes:
            case Floe():
                floes = (floes,)
            case Sequence():
                pass
            case _:
                ValueError(
                    "`floes` should be a `Floe` object or a sequence of such objects"
                )
        self.domain.floes.update(self._init_floes(floes))
        self.__history = {}
        self.save_step()

    @property
    def domain(self):
        return self.__domain

    @property
    def history(self):
        return self.__history

    @property
    def time(self):
        return self.__time

    @time.setter
    def time(self, time: float):
        self.__time = time

    def last_step(self):
        return self.history[next(reversed(self.history))]

    def save_step(self):
        self.history[self.time] = (
            copy.deepcopy(self.domain.floes),
            (self.domain.growth_mean.copy(), self.domain.growth_std),
        )

    # TODO: extract from class
    def _check_overlap(self, floes: Sequence[Floe]):
        l_edges, r_edges = map(
            np.array, zip(*((floe.left_edge, floe.right_edge) for floe in floes))
        )
        if not (r_edges[:-1] <= l_edges[1:]).all():
            raise ValueError("Floe overlap")  # TODO: dedicated exception

    def _init_floes(self, floes: Sequence[Floe]) -> list[FloeCoupled]:
        self._check_overlap(floes)

        for floe in floes:
            if floe.ice not in self.domain.ices:
                self.domain._couple_ice(floe.ice)

        phases = self._init_phases(
            floes,
            self.domain.ices,
            self.domain.ocean.wavenumbers,
            self.domain.spectrum._phases,
        )

        # If `len(floes) == 1`, the following expression evaluates to an empty
        # array. If the forcing is polychromatic, this empty array could not be
        # v-stacked with the existing, 1D-coefficients of the first floe. To
        # circumvent this, we treat the first floe separately, and use
        # `itertools.chain` when building the final list of floes. It is barely
        # more expensive than a test, and cheaper than `np.vstack`.
        coef_amps = np.exp(
            np.cumsum(
                [-self.ices[floe.ice].attenuations * floe.length for floe in floes[1:]],
                axis=0,
            )
        )

        c_floes = [
            FloeCoupled(floe, self.domain.ices[floe.ice], _phs, coefs, 0)
            for floe, _phs, coefs in zip(
                floes,
                phases,
                itertools.chain(np.ones((1, self.domain.spectrum.nf)), coef_amps),
            )
        ]
        return c_floes

        # self.floes.update(c_floes)
        # self._set_phases()

    # TODO: extract from class
    def _init_phases(
        self,
        floes: Sequence[Floe],
        ices: dict[Ice, IceCoupled],
        wavenumbers: np.ndarray,  # wavenumbers := free surface wavenumbers
        phases0: np.ndarray,
    ):
        # TODO: rewrite passing floes/ices as iterators on the quantities
        # actually needed?
        phases = [np.full(phases0.shape, np.nan) for _ in range(len(floes))]
        phases[0] = phases0 + floes[0].left_edge * wavenumbers
        for i, floe in enumerate(floes[1:], 1):
            prev = floes[i - 1]
            phases[i] = (
                phases[i - 1]
                + floe.length * ices[prev.ice].wavenumbers
                + (prev.right_edge - floe.left_edge) * wavenumbers
            )
            phases[i] %= PI_2

        return phases

    def step(self, delta_time: float, an_sol=None, num_params=None):
        self.domain.breakup(an_sol, num_params)
        self.domain.iterate(delta_time)
        self.time += delta_time
        self.save_step()
