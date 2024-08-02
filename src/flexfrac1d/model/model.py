from __future__ import annotations

import attrs
from collections.abc import Sequence
import functools
import itertools
from numbers import Real
import numpy as np
import operator
from sortedcontainers import SortedList
import typing

from ..lib.constants import PI_2, SQR2
from ..lib import att
from ..lib import dr
from ..lib import physics as ph
from ..lib.graphics import plot_displacement

if typing.TYPE_CHECKING:
    # Guard against circular imports
    from . import frac_handlers as fh


@attrs.define(frozen=True)
class Wave:
    """A monochromatic wave.

    Parameters
    ----------
    amplitude : float
        Wave amplitude in m
    period : float
        Wave period in s
    phase :
        Wave phase in rad

    Attributes
    ----------
    frequency : float
        Wave frequency in Hz
    angular_frequency: float
        Wave angular frequency in rad s^-1

    """

    amplitude: float
    period: float
    phase: float = attrs.field(default=0, converter=lambda raw: raw % PI_2)

    @classmethod
    def from_frequency(cls, amplitude, frequency, phase=0):
        """Build a wave from frequency instead of period.

        Parameters
        ----------
        amplitude : float
           The wave amplitude in m
        frequency : float
           The wave frequency in Hz
        phase :
           The wave phase in rad

        Returns
        -------
        Wave

        """
        return cls(amplitude, 1 / frequency, phase)

    @functools.cached_property
    def frequency(self) -> float:
        return 1 / self.period

    @functools.cached_property
    def angular_frequency(self) -> float:
        return PI_2 / self.period

    @functools.cached_property
    def _angular_frequency_pow2(self) -> float:
        """Return the squared angular frequency.

        This is a convenience method, this quantity appearing frequently in
        computations.

        Returns
        -------
        float


        """
        return self.angular_frequency**2


@attrs.define(frozen=True)
class Ocean:
    """The fluid bearing ice floes.

    This class encapsulates the properties of an incompressible ocean of
    constant depth and given density.

    Parameters
    ----------
    depth : float
        Ocean depth in m
    density : float
        Ocean density in kg m^-3

    """

    depth: float = np.inf
    density: float = 1025


@attrs.define(frozen=True, eq=False)
@functools.total_ordering
class _Subdomain:
    """A segment localised in space.

    Parameters
    ----------
    left_edge : float
       Coordinate of the left edge of the domain in m
    length : float
        Length of the domain in m

    Attributes
    ----------
    right_edge : float
        Coordinate of the right edge of the domain in m

    Notes
    -----
    To be used within sorted collections, instances of `_Subdomain` and its
    subclasses need to be sortable. The order is defined with respect to the
    left edge. Therefore, an equality test between two instances of
    `_Subdomain` with the same `left_edge` attribute, and differing `length`
    attributes, would hold. This unfortunately differs from the behaviour of
    all other `attrs`-defined class, and can be surprising.

    """

    # TODO: total_ordering and SortedList are nice and all, but the surprise
    # element of `==` not behaving as expected is not. Maybe consider a
    # rewrite/perf comparison, doing without, and reinstatiating a list in
    # order after breakup events, à la swisib.

    left_edge: float
    length: float

    def __eq__(self, other: _Subdomain | Real) -> bool:
        match other:
            case _Subdomain():
                return self.left_edge == other.left_edge
            case Real():
                return self.left_edge == other
            case _:
                raise TypeError(
                    "Comparison not supported between instance of "
                    f"{type(self)} and {type(other)}"
                )

    def __lt__(self, other: _Subdomain | Real) -> bool:
        match other:
            case _Subdomain():
                return self.left_edge < other.left_edge
            case Real():
                return self.left_edge < other
            case _:
                raise TypeError(
                    "Comparison not supported between instance of "
                    f"{type(self)} and {type(other)}"
                )

    @functools.cached_property
    def right_edge(self):
        return self.left_edge + self.length


@attrs.define(frozen=True)
class Ice:
    """A container for ice mechanical properties.

    Ice is modelled as an elastic thin plate, with prescribed density,
    thickness, Poisson's ratio and Young's modulus. Its fracture under bending
    is considered either through the lens of Griffith's fracture mechanics, or
    through the framework of strain failure commonly used in the sea ice
    modelling community. The fracture toughness is relevant to the former,
    while the strain threshold is relevant to the latter. Ice is considered
    translationally invariant in one horizontal direction, so that its
    quadratic moment of area is given per unit length in that direction.

    Parameters
    ----------
    density : float
        Density in kg m^-3
    frac_toughness : float
        Fracture toughness in Pa m^-1/2
    poissons_ratio : float
        Poisson's ratio
    strain_threshold : float
        Critical flexural strain in m m^-1
    thickness : float
        Ice thickness in m
    youngs_modulus : float
        Scalar Young's modulus in Pa

    Attributes
    ----------
    quad_moment : float
        Quadratic moment of area in m^3
    flex_rigidity : float
        Flexural rigidity in
    frac_energy_rate : float
        Fracture energy release rate in J m^-2

    """

    density: float = 922.5
    frac_toughness: float = 1e5
    poissons_ratio: float = 0.3
    strain_threshold: float = 3e-5
    thickness: float = 1.0
    youngs_modulus: float = 6e9

    @functools.cached_property
    def quad_moment(self) -> float:
        return self.thickness**3 / (12 * (1 - self.poissons_ratio**2))

    @functools.cached_property
    def flex_rigidity(self) -> float:
        return self.quad_moment * self.youngs_modulus

    @functools.cached_property
    def frac_energy_rate(self) -> float:
        return (
            (1 - self.poissons_ratio**2) * self.frac_toughness**2 / self.youngs_modulus
        )


@attrs.define(kw_only=True, frozen=True)
class FloatingIce(Ice):
    """An extension of `Ice` to represent properties due to buyoancy.

    Parameters
    ----------
    draft : float
       Immersed ice thickness at rest in m
    dud : float
        Height of the water column underneath the ice at rest in m
    elastic_length_pow4 : float
        Characteristic elastic length scale, raised to the 4th power, in m^4

    Attributes
    ----------
    elastic_length : float
        Characteristic elastic length scale in m
    freeboard : float
        Emerged ice thickness at rest in m

    """

    draft: float
    dud: float
    elastic_length_pow4: float

    @classmethod
    def from_ice_ocean(cls, ice: Ice, ocean: Ocean, gravity: float) -> FloatingIce:
        """Build an instance by combining properties of existing objects.

        Parameters
        ----------
        ice : Ice
        ocean : Ocean
        gravity : float
           Strengh of the local gravitational field in m s^-2

        Returns
        -------
        FloatingIce

        """
        draft = ice.density / ocean.density * ice.thickness
        dud = ocean.depth - draft
        # NOTE: as the 4th power of the elastic length scale arises naturally,
        # we prefer using it to instantiate the class and computing the length
        # scale when needed, over using the length scale for instantiation and
        # recomputing the fourth power from it, as the latter approach can lead
        # to substantial numerical imprecision.
        el_lgth_pow4 = ice.flex_rigidity / (ocean.density * gravity)
        return cls(
            density=ice.density,
            frac_toughness=ice.frac_toughness,
            poissons_ratio=ice.poissons_ratio,
            strain_threshold=ice.strain_threshold,
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
    def _elastic_number(self) -> float:
        """Reciprocal of the Characteristic elastic length scale.

        Returns
        -------
        float
            Elastic number in m^-1

        """
        return 1 / self.elastic_length

    @functools.cached_property
    def _red_elastic_number(self) -> float:
        """Characteristic elastic number scaled by 1/sqrt(2).

        Returns
        -------
        float
            Reduced elastic number in m^-1

        """
        return 1 / (SQR2 * self.elastic_length)


@attrs.define(frozen=True)
class WavesUnderElasticPlate:
    """A non-localised zone characterised by wave action.

    The spatial behaviour of waves (wavelength) is linked to their temporal
    behaviour (period) through a dispersion relation. In the case of waves
    propagating underneath floating ice, considered as an elastic plate, this
    dispersion relation depends on the properties of the ice as encapsulated by
    the `FloatingIce` class.

    Parameters
    ----------
    ice : FloatingIce
    wavenumbers : 1d array_like of float
        Propagating wavenumbers, in rad m^-1

    """

    ice: FloatingIce
    wavenumbers: np.ndarray = attrs.field(repr=False)

    @classmethod
    def from_floating(
        cls,
        ice: FloatingIce,
        spectrum: DiscreteSpectrum,
        gravity: float,
    ) -> typing.Self:
        """Build an instance by combining properties of existing objects.

        Parameters
        ----------
        ice : FloatingIce
        spectrum : DiscreteSpectrum
        gravity : float
           Strengh of the local gravitational field in m s^-2

        Returns
        -------
        WavesUnderElasticPlate

        """
        alphas = spectrum._ang_freqs_pow2 / gravity
        deg1 = 1 - alphas * ice.draft
        deg0 = -alphas * ice.elastic_length
        scaled_ratio = ice.dud / ice.elastic_length

        solver = dr.ElasticMassLoadingSolver(alphas, deg1, deg0, scaled_ratio)
        # NOTE: `solver` returns non-dimensional wavenumbers
        wavenumbers = solver.compute_wavenumbers() / ice.elastic_length

        return cls(ice, wavenumbers)

    @classmethod
    def from_ocean(
        cls,
        ice: Ice,
        ocean: Ocean,
        spectrum: DiscreteSpectrum,
        gravity: float,
    ) -> typing.Self:
        """Build an instance by combining properties of existing objects.

        Parameters
        ----------
        ice : Ice
        ocean : Ocean
        spectrum : DiscreteSpectrum
        gravity : float
           Strengh of the local gravitational field in m s^-2

        Returns
        -------
        WavesUnderElasticPlate

        """
        floating_ice = FloatingIce.from_ice_ocean(ice, ocean, gravity)
        return cls.from_floating(floating_ice, spectrum, gravity)


# TODO: check docstring
@attrs.define(frozen=True)
class WavesUnderIce:
    """A non-localised zone characetrised by wave action under floating ice.

    This class extends the behaviour of `WavesUnderElasticPlate` by adding an
    `attenuations` attribute, that parametrises the observed exponential decay
    of waves underneath floating ice.

    Parameters
    ----------
    ice : FloatingIce
    wavenumbers : 1d array_like of float
        Propagating wavenumbers, in rad m^-1
    attenuations : 1d array_like of float
        Parametrised wave amplitude attenuation rate, in m^-1

    """

    ice: FloatingIce
    wavenumbers: np.ndarray = attrs.field(repr=False)
    attenuations: np.ndarray | Real = attrs.field(repr=False)

    @classmethod
    def without_attenuation(cls, waves_under_ep: WavesUnderElasticPlate) -> typing.Self:
        """Build an instance by combining properties of existing objects.

        Parameters
        ----------
        waves_under_ep : WavesUnderElasticPlate
            An object instance

        Returns
        -------
        WavesUnderIce

        See Also
        -----
        lib.att.no_attenuation

        """
        return cls(
            waves_under_ep.ice,
            waves_under_ep.wavenumbers,
            att.no_attenuation(),
        )

    @classmethod
    def with_attenuation_01(cls, waves_under_ep: WavesUnderElasticPlate) -> typing.Self:
        """Build an instance by combining properties of existing objects.

        Parameters
        ----------
        waves_under_ep : WavesUnderElasticPlate
            An object instance

        Returns
        -------
        WavesUnderIce

        See Also
        -----
        lib.att.parameterisation_01

        """
        return cls(
            waves_under_ep.ice,
            waves_under_ep.wavenumbers,
            att.parameterisation_01(
                waves_under_ep.ice.thickness, waves_under_ep.wavenumbers
            ),
        )

    @classmethod
    def with_generic_attenuation(
        cls,
        waves_under_ep: WavesUnderElasticPlate,
        parameterisation: typing.Callable,
        args: str | None = None,
        **kwargs,
    ) -> typing.Self:
        """Instantiate a `WavesUnderFloe` with custom attenuation.

        Parameters
        ----------
        waves_under_ep : WavesUnderElasticPlate
            An object instance.
        parameterisation : typing.Callable
            Function defining attenuation.
            Must return a type broadcastable to `waves_under_ep.wavenumbers`.
        args : str | None
            A string of attributes of `waves_under_ep`, separated by
            whitespace, to be passed as parameters to `parameterisation`.
            All parameters will be passed as a mapping between the stem of the
            attribute and its value.
        **kwargs : dict
            Additional parameters to pass to `parameterisation`.

        Returns
        -------
        WavesUnderFloe

        Examples
        --------
        Assuming an existing `wue` instance of `WavesUnderElasticPlate`, the
        three following objects are identical, setting the attenuation egal to
        the ice density for all wave modes.

        >>> WavesUnderIce.with_generic_attenuation_param(
            wue,
            lambda density: density,
            "ice.density"
        )
        >>> WavesUnderIce.with_generic_attenuation_param(
            wue,
            lambda density: density,
            {"density": wue.ice.density},
        )
        >>> WavesUnderIce(wue.ice, wue.wavenumbers, wue.ice.density)

        """
        if args is not None:
            kwargs |= {
                arg.split(".")[-1]: operator.attrgetter(arg)(waves_under_ep)
                for arg in args.split()
            }
        return cls(
            waves_under_ep.ice, waves_under_ep.wavenumbers, parameterisation(**kwargs)
        )

    @functools.cached_property
    def _c_wavenumbers(self) -> np.ndarray:
        """Complex wavenumbers.

        Their real part correspond to the propagating wavenumber, while their
        imaginary part correspond to the attenuation rate.

        Returns
        -------
        1d np.ndarray of complex
            The complex wavenumbers in m^-1

        """
        return self.wavenumbers + 1j * self.attenuations


@attrs.define(frozen=True)
class FreeSurfaceWaves:
    """The wave state in the absence of ice.

    The spatial behaviour of waves (wavelength) is linked to their temporal
    behaviour (period) through a dispersion relation. In the case of free
    surface waves, propagating underneath floating ice, this dispersion
    relation depends on the properties of the ocean as encapsulated in the
    `Ocean` class.

    Parameters
    ----------
    ocean : Ocean
    wavenumbers : array_like
        Propagating wavenumbers in rad m^-1

    Attributes
    ----------
    wavelengths : 1d np.ndarray of float
        Propagating wavelengths in m

    """

    ocean: Ocean
    wavenumbers: np.ndarray

    @classmethod
    def from_ocean(cls, ocean: Ocean, spectrum: DiscreteSpectrum, gravity: float):
        """Build an instance by combining properties of existing objects."""
        alphas = spectrum._ang_freqs_pow2 / gravity
        solver = dr.FreeSurfaceSolver(alphas, ocean.depth)
        wavenumbers = solver.compute_wavenumbers()
        return cls(ocean, wavenumbers)

    @functools.cached_property
    def wavelengths(self) -> np.ndarray:
        return PI_2 / self.wavenumbers


# TODO: docstring inheritance
@attrs.define(kw_only=True, eq=False)
class Floe(_Subdomain):
    """An ice floe localised in space.

    Parameters
    ----------
    ice : Ice
        The mechanical properties of the floe

    """

    ice: Ice


@attrs.define(kw_only=True, eq=False)
class WavesUnderFloe(_Subdomain):
    """A localised zone characetrised by wave action under floating ice.

    Parameters
    ----------
    wui : WavesUnderIce
    edge_amplitudes : 1d np.ndarray of complex
        The wave complex amplitude at the left edge of the floe in m
    generation : int
        The number of fractures that led to the existence of this floe

    """

    wui: WavesUnderIce
    edge_amplitudes: np.ndarray
    generation: int = 0

    @functools.cached_property
    def _adim(self) -> float:
        """A non-dimentional number characetrising the floe.

        Returns
        -------
        float

        """
        return self.length * self.wui.ice._red_elastic_number

    # TODO: typing.Self?
    def make_copy(self) -> WavesUnderFloe:
        return WavesUnderFloe(
            left_edge=self.left_edge,
            length=self.length,
            wui=self.wui,
            edge_amplitudes=self.edge_amplitudes.copy(),
            generation=self.generation,
        )

    @typing.overload
    def shift_waves(self, phase_shifts: np.ndarray, inplace: typing.Literal[True]): ...

    # TODO: typing.Self
    @typing.overload
    def shift_waves(
        self, phase_shifts: np.ndarray, inplace: typing.Literal[False]
    ) -> WavesUnderFloe: ...

    # TODO: docstring
    def shift_waves(self, phase_shifts: np.ndarray, inplace: bool = True):
        shifted_amplitudes = self.edge_amplitudes * np.exp(-1j * phase_shifts)
        if not inplace:
            return WavesUnderFloe(
                left_edge=self.left_edge,
                length=self.length,
                wui=self.wui,
                edge_amplitudes=shifted_amplitudes,
                generation=self.generation,
            )
        # HACK: instantiating an new object is cheap, resorting a whole list of
        # subdomains is not, so we mutate the amplitude/phase instead
        object.__setattr__(self, "edge_amplitudes", shifted_amplitudes)

    def energy(self, growth_params=None, an_sol: bool = False, num_params=None):
        return ph.EnergyHandler.from_wuf(self, growth_params).compute(
            an_sol, num_params
        )


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
    def _ang_freqs_pow2(self):
        return self._ang_freqs**2

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


# TODO: docstrings
@attrs.define
class Domain:
    """A spatial domain forced by waves.

    This represents the state of a MIZ at a given time.


    Attributes
    ----------
    gravity : float
    spectrum : DiscreteSpectrum
    fsw : FreeSurfaceWaves
    attenuation: flexrac1d.lib.att.Attenuation
    growth_params : list
    subdomains : SortedList of WavesUnderFloe
    cached_wuis :
    cached_phases :

    """

    gravity: float
    spectrum: DiscreteSpectrum
    fsw: FreeSurfaceWaves
    attenuation: att.Attenuation = attrs.field(repr=False)
    growth_params: list[np.array, float] | None = None
    subdomains: SortedList = attrs.field(repr=False, init=False, factory=SortedList)
    cached_wuis: dict[Ice, WavesUnderIce] = attrs.field(
        repr=False, init=False, factory=dict
    )
    cached_phases: dict[float, np.ndarray] = attrs.field(
        repr=False, init=False, factory=dict
    )

    @classmethod
    def from_discrete(
        cls,
        gravity,
        spectrum,
        ocean,
        attenuation: att.Attenuation | None = None,
        growth_params: tuple | None = None,
    ):
        fsw = FreeSurfaceWaves.from_ocean(ocean, spectrum, gravity)
        if attenuation is None:
            attenuation = att.AttenuationParameterisation(1)
        return cls(gravity, spectrum, fsw, attenuation, growth_params)

    @classmethod
    def with_growth_means(
        cls,
        gravity: float,
        spectrum: DiscreteSpectrum,
        ocean: Ocean,
        growth_means: np.ndarray | Sequence[Real] | Real,
        attenuation: att.Attenuation | None = None,
    ) -> typing.Self:
        return cls.from_discrete(
            gravity, spectrum, ocean, attenuation, (growth_means, None)
        )

    @classmethod
    def with_growth_std(
        cls,
        gravity: float,
        spectrum: DiscreteSpectrum,
        ocean: Ocean,
        growth_std: Real,
        attenuation: att.Attenuation | None = None,
    ) -> typing.Self:
        return cls.from_discrete(gravity, spectrum, ocean, attenuation, (0, growth_std))

    def __attrs_post_init__(self):
        if self.growth_params is not None:
            if len(self.growth_params) != 2:
                raise ValueError
            growth_means, growth_std = (
                np.asarray(self.growth_params[0]),
                self.growth_params[1],
            )
            # TODO: simplify all this. Ideally, do not test for anything or
            # babysit the user. Why was upping growth_mean to a column
            # necessary in case its of size 1?
            if growth_means.size == 1:
                # As `broadcast_to` returns a view,
                # copying is necessary to obtain a mutable array. It is easier
                # than dealing with 0-length and 1-length arrays seperately.
                growth_means = np.broadcast_to(
                    growth_means, (self.spectrum.nf, 1)
                ).copy()
            else:
                if growth_means.size != self.spectrum.nf:
                    raise ValueError(
                        f"Means (size {growth_means.size}) could not be"
                        "broadcast with the shape of the spectrum"
                        f"({self.spectrum.nf} components)"
                    )
            if growth_std is None:
                growth_std = self.fsw.wavelengths[self.spectrum._amps.argmax()]
            self.growth_params = [growth_means, growth_std]

    def _compute_phase_shifts(self, delta_time: float):
        if delta_time not in self.cached_phases:
            self.cached_phases[delta_time] = delta_time * self.spectrum._ang_freqs
        return self.cached_phases[delta_time]

    def _compute_wui(self, ice: Ice):
        if ice not in self.cached_wuis:
            wup = WavesUnderElasticPlate.from_ocean(
                ice, self.fsw.ocean, self.spectrum, self.gravity
            )
            if isinstance(self.attenuation, att.AttenuationParameterisation):
                if self.attenuation == att.AttenuationParameterisation.NO:
                    wui = WavesUnderIce.without_attenuation(wup)
                elif self.attenuation == att.AttenuationParameterisation.PARAM_01:
                    wui = WavesUnderIce.with_attenuation_01(wup)
            else:
                wui = WavesUnderIce.with_generic_attenuation(
                    wup,
                    self.attenuation.function,
                    self.attenuation.args,
                    **self.attenuation.kwargs,
                )
            self.cached_wuis[ice] = wui
        return self.cached_wuis[ice]

    def _shift_phases(self, phases: np.ndarray):
        for i in range(len(self.floes)):
            self.floes[i].phases -= phases

    def _shift_growth_means(self, phases: np.ndarray):
        # TODO: refine to take into account subdomain transitions
        # and floes with variying properties
        mask = self.growth_params[0] < self.subdomains[0].left_edge
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
    def _promote_floe(floes: Floe | Sequence[Floe]) -> Sequence[Floe]:
        match floes:
            case Floe():
                return (floes,)
            case Sequence():
                return floes
            case _:
                raise ValueError(
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
            WavesUnderFloe(
                left_edge=floe.left_edge,
                length=floe.length,
                wui=self._compute_wui(floe.ice),
                edge_amplitudes=edge_amplitudes,
            )
            for floe, edge_amplitudes in zip(floes, complex_amplitudes)
        ]

    def iterate(self, delta_time: float):
        phase_shifts = self._compute_phase_shifts(delta_time)
        # TODO: can be optimised by iterating a first time to extract the
        # edges, coerce them to a np.array, apply the product with
        # complex_shifts, and then iterate a second time to build the objects.
        # See Propagation_tests.ipynb/DNE06-26
        for i in range(len(self.subdomains)):
            self.subdomains[i].shift_waves(phase_shifts)
        if self.growth_params is not None:
            # Phases are only modulo'd in the setter
            self._shift_growth_means(phase_shifts)

    def _pop_c_floe(self, wuf: WavesUnderFloe):
        self.subdomains.remove(wuf)

    def _add_c_floes(self, wuf: Sequence[WavesUnderFloe]):
        # It is assume no overlap will occur, and phases have been properly
        # set, as these method should only be called after a fracture event
        self.subdomains.update(wuf)

    def breakup(
        self, fracture_handler: fh._FractureHandler, an_sol=None, num_params=None
    ):
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
