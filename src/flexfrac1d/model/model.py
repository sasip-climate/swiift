#!/usr/bin/env python3

from __future__ import annotations

import attrs
from collections.abc import Sequence
import functools
import itertools
from numbers import Real
import numpy as np
from sortedcontainers import SortedList

from ..lib.constants import PI_2, SQR2
from ..lib import dr
from ..lib.graphics import plot_displacement


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
    cached_wuis: dict[Ice, WavesUnderIce] = attrs.field(init=False, factory=dict)

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

    def _compute_wui(self, ice: Ice):
        if ice not in self.cached_wuis:
            self.cached_wuis[ice] = WavesUnderIce.from_ocean(
                ice, self.fsw.ocean, self.spectrum, self.gravity
            )
        return self.cached_wuis[ice]

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

    def add_floes(self, floes: Floe | Sequence[Floe]):
        subdomains = self._init_subdomains(floes)
        self._add_c_floes(subdomains)

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
        l_edges, r_edges = map(
            np.array, zip(*((floe.left_edge, floe.right_edge) for floe in floes))
        )
        if not (r_edges[:-1] <= l_edges[1:]).all():
            raise ValueError("Floe overlap")  # TODO: dedicated exception

    def _init_phases(self, floes: Sequence[Floe]) -> np.ndarray:
        phases = np.full((len(floes), self.spectrum.nf), np.nan)
        phases[0] = self.spectrum._phases + floes[0].left_edge * self.fsw.wavenumbers
        for i, floe in enumerate(floes[1:], 1):
            wui = self._compute_wui(floe.ice)
            prev = floes[i - 1]
            phases[i:,] = (
                phases[i - 1]
                + floe.length * wui.wavenumbers
                + (prev.right_edge - floe.left_edge) * self.fsw.wavenumbers
            )
        return phases % PI_2

    def _init_amplitudes(self, floes: Sequence[Floe]) -> np.ndarray:
        amplitudes = np.full((len(floes), self.spectrum.nf), np.nan)
        amplitudes[0, :] = self.spectrum._amps
        for i, floe in enumerate(floes[1:], 1):
            amplitudes[i, :] = amplitudes[i - 1, :] * np.exp(
                -self._compute_wui(floe.ice).attenuations * floe.length
            )
        return amplitudes

    def _init_subdomains(self, floes: Sequence[Floe]) -> list[WavesUnderFloe]:
        # TODO: look for already existing floes. In the present state, only
        # valid for starting from scratch, not for adding floes to a domain
        # that already has some.
        floes = self.__class__._promote_floe(floes)
        self._check_overlap(floes)
        complex_amplitudes = self._init_amplitudes(floes) * np.exp(
            1j * self._init_phases(floes)
        )

        return [
            WavesUnderFloe(self._compute_wui(floe.ice), floe, edge_amplitudes)
            for floe, edge_amplitudes in zip(floes, complex_amplitudes)
        ]

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
        for i in range(len(self.subdomains)):
            self.subdomains[i].edge_amplitudes *= complex_shifts
        if self.growth_params is not None:
            # Phases are only modulo'd in the setter
            self._shift_growth_means(phase_shifts)

    def _pop_c_floe(self, wuf: WavesUnderFloe):
        self.subdomains.remove(wuf)

    def _add_c_floes(self, wuf: Sequence[WavesUnderFloe]):
        # It is assume no overlap will occur, and phases have been properly
        # set, as these method should only be called after a fracture event
        self.subdomains.update(wuf)

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
        plot_displacement(
            resolution, self, left_bound, ax, an_sol, add_surface, base, kw_dis, kw_sur
        )
