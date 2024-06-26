#!/usr/bin/env python3

from __future__ import annotations

import attrs
from collections import namedtuple
from collections.abc import Sequence
import functools
import itertools
from numbers import Real
import numpy as np

import scipy.optimize as optimize
import scipy.signal as signal
from sortedcontainers import SortedList

# from .lib.displacement import displacement
# from .lib.curvature import curvature
from .lib import dr

# from .lib.energy import energy
from .lib.numerical import free_surface
from .lib.graphics import plot_displacement
from .lib.constants import PI_2, SQR2


@attrs.define(frozen=True)
class Wave:
    """Represents a monochromatic wave."""

    amplitude: float
    period: float
    phase: float = attrs.field(default=0, converter=lambda raw: raw % PI_2)

    @classmethod
    def from_frequency(cls, amplitude, frequency, phase=0):
        return cls(amplitude, 1 / frequency, phase)

    @functools.cached_property
    def frequency(self) -> float:
        """Wave frequency in Hz."""
        return 1 / self.period

    @functools.cached_property
    def angular_frequency(self) -> float:
        """Wave angular frequency in rad s**-1."""
        return PI_2 / self.period

    # TODO: rename to ..._pow2
    @functools.cached_property
    def angular_frequency2(self) -> float:
        """Squared wave angular frequency, for convenience."""
        return self.angular_frequency**2


@attrs.define(frozen=True)
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

    depth: float = np.inf
    density: float = 1025


@attrs.define(frozen=True)
class Ice:
    density: float = 922.5
    frac_toughness: float = 1e5
    poissons_ratio: float = 0.3
    thickness: float = 1.0
    youngs_modulus: float = 6e9

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


@attrs.define(kw_only=True, frozen=True)
class FloatingIce(Ice):
    draft: float
    dud: float
    elastic_length_pow4: float

    @classmethod
    def from_ice_ocean(cls, ice: Ice, ocean: Ocean, gravity: float):
        draft = ice.density / ocean.density * ice.thickness
        dud = ocean.depth - draft
        el_lgth_pow4 = ice.flex_rigidity / (ocean.density * gravity)
        return cls(
            density=ice.density,
            frac_toughness=ice.frac_toughness,
            poissons_ratio=ice.poissons_ratio,
            thickness=ice.thickness,
            youngs_modulus=ice.youngs_modulus,
            draft=draft,
            dud=dud,
            elastic_length_pow4=el_lgth_pow4,
        )

    @functools.cached_property
    def elastic_length(self):
        return self.elastic_length_pow4**0.25

    @functools.cached_property
    def freeboard(self):
        return self.thickness - self.draft

    @functools.cached_property
    def _elastic_number(self):
        return 1 / self.elastic_length

    @functools.cached_property
    def _red_elastic_number(self):
        return 1 / (SQR2 * self.elastic_length)


@attrs.define(frozen=True)
class WavesUnderIce:
    ice: FloatingIce
    wavenumbers: np.ndarray = attrs.field(repr=False)

    @wavenumbers.default
    def _dummy_array_factory(self):
        return np.full(1, np.nan)

    @classmethod
    def from_floating(
        cls, ice: FloatingIce, spectrum: DiscreteSpectrum, gravity: float
    ):
        alphas = spectrum._ang_freq2 / gravity
        deg1 = 1 - alphas * ice.draft
        deg0 = -alphas * ice.elastic_length
        scaled_ratio = ice.dud / ice.elastic_length

        solver = dr.ElasticMassLoadingSolver(alphas, deg1, deg0, scaled_ratio)
        wavenumbers = solver.compute_wavenumbers() / ice.elastic_length

        return cls(ice, wavenumbers)

    @classmethod
    def from_ocean(
        cls, ice: Ice, ocean: Ocean, spectrum: DiscreteSpectrum, gravity: float
    ):
        floating_ice = FloatingIce.from_ice_ocean(ice, ocean, gravity)
        return cls.from_floating(floating_ice, spectrum, gravity)

    @functools.cached_property
    def _c_wavenumbers(self):
        return self.wavenumbers + 1j * self.attenuations

    @functools.cached_property
    def attenuations(self):
        return self.wavenumbers**2 * self.ice.thickness / 4


@attrs.define(frozen=True)
class FreeSurfaceWaves:
    ocean: Ocean
    wavenumbers: np.ndarray

    @classmethod
    def from_ocean(cls, ocean: Ocean, spectrum: DiscreteSpectrum, gravity: float):
        alphas = spectrum._ang_freq2 / gravity
        solver = dr.FreeSurfaceSolver(alphas, ocean.depth)
        wavenumbers = solver.compute_wavenumbers()
        return cls(ocean, wavenumbers)

    @functools.cached_property
    def wavelengths(self) -> np.ndarray:
        """Wavelengths in m"""
        return PI_2 / self.wavenumbers


@attrs.define(frozen=True)
@functools.total_ordering
class Floe:
    left_edge: float
    length: float
    ice: FloatingIce
    generation: int = 0

    def __eq__(self, other: Floe | Real) -> bool:
        match other:
            case Floe():
                return self.left_edge == other.left_edge
            case Real():
                return self.left_edge == other
            case _:
                raise NotImplementedError

    def __lt__(self, other: Floe | Real) -> bool:
        match other:
            case Floe():
                return self.left_edge < other.left_edge
            case Real():
                return self.left_edge < other
            case _:
                raise NotImplementedError

    @functools.cached_property
    def right_edge(self):
        return self.left_edge + self.length


@attrs.define
class WavesUnderFloe:
    wui: WavesUnderIce
    floe: Floe
    edge_amplitudes: np.ndarray

    @functools.cached_property
    def _adim(self):
        return self.length * self.wui.ice._red_elastic_number

    def __eq__(self, other: WavesUnderFloe | Real) -> bool:
        match other:
            case WavesUnderFloe():
                return self.floe.left_edge == other.floe.left_edge
            case Real():
                return self.floe.left_edge == other
            case _:
                raise NotImplementedError

    def __lt__(self, other: WavesUnderFloe | Real) -> bool:
        match other:
            case WavesUnderFloe():
                return self.floe.left_edge < other.floe.left_edge
            case Real():
                return self.floe.left_edge < other
            case _:
                raise NotImplementedError


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
                ener[i] = self._ener_min(
                    length, spectrum, growth_params, an_sol, num_params
                )

            peak_idxs = np.hstack(
                (0, signal.find_peaks(np.log(ener), distance=2)[0], ener.size - 1)
            )

            local_ener_cost = functools.partial(
                self._ener_min,
                spectrum=spectrum,
                growth_params=growth_params,
                an_sol=an_sol,
                num_params=num_params,
            )
            opts = [
                optimize.minimize_scalar(
                    local_ener_cost,
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

    def _fracture_diagnostic(
        self, spectrum, res=0.1, growth_params=None, an_sol=False, num_params=None
    ):
        lengths = np.linspace(
            0, self.length, np.ceil(self.length / res).astype(int) + 1
        )[1:-1]
        energies = np.full((lengths.size, 2), np.nan)
        for i, length in enumerate(lengths):
            energies[i, :] = [
                _f.energy(spectrum, growth_params, an_sol, num_params)
                for _f in self._binary_split(length)
            ]
        frac_diag = namedtuple("FractureDiagnostic", ("length", "energy"))
        return frac_diag(lengths, energies)

    def _binary_split(self, length) -> tuple[FloeCoupled]:
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
        return cf_l, cf_r

    def _ener_min(self, length, spectrum, growth_params, an_sol, num_params) -> float:
        """Objective function to minimise for energy-based fracture"""
        cf_l, cf_r = self._binary_split(length)
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
    ):

        # np.ravel to force precisely 1D-arrays
        # Promote the map to list so the iterator can be used several times
        args = list(map(np.ravel, (amplitudes, frequencies, phases)))
        (size,) = np.broadcast_shapes(*(arr.shape for arr in args))

        # TODO: sort waves by frequencies or something
        # TODO: sanity checks on nan, etc. that could be returned
        #       by the Spectrum objects

        # If size is one, all the arguments are scalar and the "spectrum" is
        # monochromatic. Otherwise, there is at least one argument with
        # different components. The eventual other arguments with a single
        # component are repeated for instantiating as many Wave objects as
        # needed.
        if size != 1:
            for i, arr in enumerate(args):
                if arr.size == 0:
                    raise ValueError
                if arr.size == 1:
                    args[i] = itertools.repeat(arr[0], size)

        self.__waves = [
            Wave.from_frequency(_a, _f, phase=_ph) for _a, _f, _ph in zip(*args)
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


@attrs.define
class Domain:
    gravity: float
    spectrum: DiscreteSpectrum
    fsw: FreeSurfaceWaves
    growth_params: list[np.array, float] | None = None
    subdomains: SortedList = attrs.field(init=False, factory=SortedList)
    _cached_wuis: dict[Ice, WavesUnderIce] = attrs.field(factory=dict, init=False)

    @classmethod
    def from_discrete(cls, gravity, spectrum, ocean, growth_params):
        fsw = FreeSurfaceWaves.from_ocean(ocean, spectrum, gravity)
        return cls(gravity, spectrum, fsw, growth_params)

    def __attrs_post_init__(self):
        if self.growth_params is not None:
            if len(self.growth_params) != 2:
                raise ValueError
            growth_mean, growth_std = (
                np.asarray(self.growth_params[0]),
                self.growth_params[1],
            )
            if growth_mean.size == 1:
                # As `broadcast_to` returns a view,
                # copying is necessary to obtain a mutable array. It is easier
                # than dealing with 0-length and 1-length arrays seperately.
                growth_mean = np.broadcast_to(growth_mean, (self.spectrum.nf, 1)).copy()
            if growth_std is None:
                growth_std = self.fsw.wavelengths[self.spectrum._amps.argmax()]
            self.growth_params = [growth_mean, growth_std]

    # def __init__(
    #     self,
    #     gravity,
    #     spectrum: WaveSpectrum,
    #     ocean: Ocean,
    #     growth_mean=None,
    #     growth_std=None,
    # ) -> None:
    #     """"""
    #     self.__gravity = gravity
    #     self.__frozen_spectrum = spectrum
    #     self.__ocean = OceanCoupled(ocean, spectrum, gravity)
    #     self.__floes = SortedList()
    #     self.__ices = {}

    #     if growth_mean is None:
    #         if growth_std is not None:
    #             growth_mean = np.zeros((self.spectrum.nf, 1))
    #     else:
    #         growth_mean = np.asarray(growth_mean)
    #         if growth_mean.size == 1:
    #             # As `broadcast_to` returns a view,
    #             # copying is necessary to obtain a mutable array
    # growth_mean = np.broadcast_to(growth_mean,
    #                               (self.spectrum.nf, 1)).copy()
    #         if growth_std is None:
    #             growth_std = (
    #                 2 * np.pi / self.ocean.wavenumbers[self.spectrum._amps.argmax()]
    #             )
    #     self.__growth_mean = growth_mean
    #     self.__growth_std = growth_std

    # @property
    # def floes(self) -> SortedList[FloeCoupled]:
    #     return self.__floes

    # @property
    # def gravity(self) -> float:
    #     return self.__gravity

    # @property
    # def ices(self) -> dict[Ice, IceCoupled]:
    #     return self.__ices

    # @property
    # def ocean(self) -> OceanCoupled:
    #     return self.__ocean

    # @property
    # def spectrum(self) -> DiscreteSpectrum:
    #     return self.__frozen_spectrum

    # @property
    # def growth_mean(self):
    #     return self.__growth_mean

    # @growth_mean.setter
    # def growth_mean(self, value: np.ndarray):
    #     self.__growth_mean = value

    # @property
    # def growth_std(self):
    #     return self.__growth_std

    # def _pack_growth(self, floe):
    #     if self.growth_mean is None:
    #         return None
    #     return self.growth_mean - floe.left_edge, self.growth_std

    def _couple_ice(self, ice):
        self.ices[ice] = IceCoupled(ice, self.ocean, self.spectrum, None, self.gravity)

    def _compute_wui(self, ice: Ice):
        if ice not in cached_wuis:
            self._cached_wuis[ice] = WavesUnderIce.from_ocean(
                ice, self.fsw.ocean, self.spectrum, self.gravity
            )
        return self._cached_wuis[ice]

    def _shift_phases(self, phases: np.ndarray):
        for i in range(len(self.floes)):
            self.floes[i].phases -= phases

    def _shift_growth_means(self, phases: np.ndarray):
        # TODO: refine to take into account subdomain transitions
        # and floes with variying properties
        mask = self.growth_mean < self.floes[0].left_edge
        if mask.any():
            self.growth_params[0][mask] += (
                phases[mask[:, 0]] / self.fsw.wavenumbers[mask[:, 0]]
            )
        if not mask.all():
            self.growth_params[0][~mask] += (
                phases[~mask[:, 0]] / self.subdomains[0].wui.wavenumbers[~mask[:, 0]]
            )

    def add_floes(self, floes: Floe | Sequence[Floe]): ...

    @staticmethod
    def _promote_floe(floes: Floe | Sequence[Floe]):
        match floes:
            case Floe():
                return (floes,)
            case Sequence():
                return floes
            case _:
                ValueError(
                    "`floes` should be a `Floe` object or a sequence of such objects"
                )

    def _check_overlap(self, floes: Sequence[Floe]):
        # TODO: look for already existing floes
        l_edges, r_edges = map(
            np.array, zip(*((floe.left_edge, floe.right_edge) for floe in floes))
        )
        if not (r_edges[:-1] <= l_edges[1:]).all():
            raise ValueError("Floe overlap")  # TODO: dedicated exception

    # TODO: extract from class
    def _init_phases(
        self,
        floes: Sequence[Floe],
    ):
        phases0 = self.spectrum._phases
        phases = [np.full(phases0.shape, np.nan) for _ in range(len(floes))]
        phases[0] = phases0 + floes[0].left_edge * self.fsw.wavenumbers
        for i, floe in enumerate(floes[1:], 1):
            wui = self._compute_wui(floe)
            prev = floes[i - 1]
            phases[i] = (
                phases[i - 1]
                + floe.length * wui.wavenumbers
                + (prev.right_edge - floe.left_edge) * self.fsw.wavenumbers
            )
            phases[i] %= PI_2

        return phases

    def _init_amplitudes(self, floes):
        amplitudes0 = self.spectrum._amps

    def _init_floes(self, floes: Sequence[Floe]) -> list[FloeCoupled]:
        self._check_overlap(floes)
        floes = self.__class__._promote_floe(floes)
        wuis = (self.domain._compute_wui(floe) for floe in floes)
        phases = self._init_phases(floes)

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
        complex_shifts = np.exp(-1j * phase_shifts)
        # TODO: can be optimised by iterating a first time to extract the
        # edges, coerce them to a np.array, apply the product with
        # complex_shifts, and then iterate a second time to build the objects.
        # See Propagation_tests.ipynb/DNE06-26
        new_wufs = [
            WavesUnderFloe(wuf.wui, wuf.floe, wuf.edge_amplitudes * complex_shifts)
            for wuf in self.subdomains
        ]
        self.subdomains = SortedList(new_wufs)
        if self.growth_params is not None:
            # Phases are only modulo'd in the setter
            self._shift_growth_means(phase_shifts)

    def _pop_c_floe(self, wuf: WavesUnderFloe):
        self.floes.remove(wuf)

    def _add_c_floes(self, wuf: tuple[WavesUnderFloe]):
        # It is assume no overlap will occur, and phases have been properly
        # set, as these method should only be called after a fracture event
        self.floes.update(wuf)

    def breakup(self, fracture_handler, an_sol=None, num_params=None):
        dct = {}
        for i, wuf in enumerate(self.subdomains):
            xf = fracture_handler.search(wuf, self.growth_params, an_sol, num_params)
            if xf is not None:
                old = wuf
                new = fracture_handler.split(wuf, xf)
                dct[i] = old, new
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


@attrs.define
class Experiment:
    time: float
    domain: Domain
    history: dict = attrs.field(factory=dict, init=False)

    # def __init__(self, domain: Domain, floes: Floe | Sequence[Floe]):
    #     self.__time = 0
    #     self.__domain = domain
    #     match floes:
    #         case Floe():
    #             floes = (floes,)
    #         case Sequence():
    #             pass
    #         case _:
    #             ValueError(
    #                 "`floes` should be a `Floe` object or a sequence of such objects"
    #             )
    #     self.domain.floes.update(self._init_floes(floes))
    #     self.__history = {}
    #     self.save_step()

    def __attrs_post_init__(self):
        self.save_step()

    @classmethod
    def from_domain(cls, domain: Domain):
        wuf_list = cls._init_floes(cls._promote_floe(floes))
        domain.floes.update(wuf_list)
        return cls(0, domain)

    @classmethod
    def from_discrete(
        cls,
        gravity: float,
        spectrum: DiscreteSpectrum,
        ocean: Ocean,
        growth_params: tuple,
    ):
        return cls.from_domain(
            0, Domain.from_discrete(gravity, spectrum, ocean, growth_params)
        )

    def add_floes(self, floes: Floe | Sequence[Floe]):
        self.domain.add_floes(floes)
        # No need to save_step: self.history only holds a reference to
        # self.domain.subdomains

    # @property
    # def domain(self):
    #     return self.__domain

    # @property
    # def history(self):
    #     return self.__history

    # @property
    # def time(self):
    #     return self.__time

    # @time.setter
    # def time(self, time: float):
    #     self.__time = time

    def last_step(self):
        return self.history[next(reversed(self.history))]

    def save_step(self):
        self.history[self.time] = (
            self.domain.subdomains,
            (self.domain.growth_params[0].copy(), self.domain.growth_params[1]),
        )

    def step(self, delta_time: float, an_sol=None, num_params=None):
        self.domain.breakup(an_sol, num_params)
        self.domain.iterate(delta_time)
        self.time += delta_time
        self.save_step()
