#!/usr/bin/env python3

from __future__ import annotations

import attrs

# from collections import namedtuple
from collections.abc import Sequence

# from .lib.displacement import displacement
# from .lib.curvature import curvature

# from .lib.energy import energy
# from .lib.numerical import free_surface

from ..model import model as md
from ..model import frac_handlers as fh

# class FloeCoupled(Floe):
#     def __init__(
#         self,
#         floe: Floe,
#         ice: IceCoupled,
#         phases: np.ndarray | list[float] | float,
#         amp_coefficients: np.ndarray | list[float] | float,
#         gen: int = 0,
#         dispersion=None,
#     ):
#         super().__init__(floe.left_edge, floe.length, ice, dispersion)
#         self.phases = np.asarray(phases)  # no dunder: uses the setter method
#         self.__amp_coefficients = amp_coefficients
#         self.__gen = gen

#     @property
#     def phases(self) -> np.ndarray:
#         return self.__phases

#     @phases.setter
#     def phases(self, value):
#         self.__phases = np.asarray(value) % PI_2

#     @property
#     def amp_coefficients(self):
#         return self.__amp_coefficients

#     @property
#     def gen(self) -> int:
#         return self.__gen

#     @property
#     def ice(self) -> IceCoupled:
#         return self._Floe__ice

#     @functools.cached_property
#     def _adim(self):
#         return self.length * self.ice._red_elastic_number

#     def _pack(
#         self, spectrum: DiscreteSpectrum
#     ) -> tuple[tuple[float], tuple[np.ndarray]]:
#         return (self.ice._red_elastic_number, self.length), (
#             self.amp_coefficients * spectrum._amps,
#             self.ice._c_wavenumbers,
#             self.phases,
#         )

#     def forcing(self, x, spectrum, growth_params):
#         return free_surface(x, self._pack(spectrum)[1], growth_params)

#     def displacement(self, x, spectrum, growth_params, an_sol, num_params):
#         """Complete solution of the displacement ODE

#         `x` is expected to be relative to the floe, i.e. to be bounded by 0, L
#         """
#         return displacement(x, *self._pack(spectrum), growth_params, an_sol, num_params)

#     def curvature(self, x, spectrum, growth_params, an_sol, num_params):
#         """Curvature of the floe, i.e. second derivative of the vertical displacement"""
#         return curvature(x, *self._pack(spectrum), growth_params, an_sol, num_params)

#     def energy(self, spectrum: DiscreteSpectrum, growth_params, an_sol, num_params):
#         factor = self.ice.flex_rigidity / (2 * self.ice.thickness)
#         unit_energy = energy(*self._pack(spectrum), growth_params, an_sol, num_params)
#         return factor * unit_energy
#         # In case of a numerical solution, the result is the output of
#         # integrate.quad, that is a (solution, bound on error) tuple.
#         # We do not do anything with the latter at the moment.
#         # return factor * unit_energy[0]

#     def search_fracture(
#         self, spectrum: DiscreteSpectrum, growth_params, an_sol, num_params
#     ):
#         return self.binary_fracture(spectrum, growth_params, an_sol, num_params)

#     def binary_fracture(
#         self, spectrum: DiscreteSpectrum, growth_params, an_sol, num_params
#     ) -> float | None:
#         coef_nd = 4
#         base_energy = self.energy(spectrum, growth_params, an_sol, num_params)
#         # No fracture if the elastic energy is below the threshold
#         if base_energy < self.ice.frac_energy_rate:
#             return None
#         else:
#             nd = (
#                 np.ceil(
#                     4 * self.length * self.ice.wavenumbers.max() / (2 * np.pi)
#                 ).astype(int)
#                 + 2
#             )
#             lengths = np.linspace(0, self.length, nd * coef_nd)[1:-1]
#             ener = np.full(lengths.shape, np.nan)
#             for i, length in enumerate(lengths):
#                 ener[i] = self._ener_min(
#                     length, spectrum, growth_params, an_sol, num_params
#                 )

#             peak_idxs = np.hstack(
#                 (0, signal.find_peaks(np.log(ener), distance=2)[0], ener.size - 1)
#             )

#             local_ener_cost = functools.partial(
#                 self._ener_min,
#                 spectrum=spectrum,
#                 growth_params=growth_params,
#                 an_sol=an_sol,
#                 num_params=num_params,
#             )
#             opts = [
#                 optimize.minimize_scalar(
#                     local_ener_cost,
#                     bounds=lengths[peak_idxs[[i, i + 1]]],
#                 )
#                 for i in range(len(peak_idxs) - 1)
#             ]
#             opt = min(filter(lambda opt: opt.success, opts), key=lambda opt: opt.fun)
#             # Minimisation is done on the log of energy
#             if np.exp(opt.fun) + self.ice.frac_energy_rate < base_energy:
#                 return opt.x
#             else:
#                 return None

#     def _fracture_diagnostic(
#         self, spectrum, res=0.1, growth_params=None, an_sol=False, num_params=None
#     ):
#         lengths = np.linspace(
#             0, self.length, np.ceil(self.length / res).astype(int) + 1
#         )[1:-1]
#         energies = np.full((lengths.size, 2), np.nan)
#         for i, length in enumerate(lengths):
#             energies[i, :] = [
#                 _f.energy(spectrum, growth_params, an_sol, num_params)
#                 for _f in self._binary_split(length)
#             ]
#         frac_diag = namedtuple("FractureDiagnostic", ("length", "energy"))
#         return frac_diag(lengths, energies)

#     def _binary_split(self, length) -> tuple[FloeCoupled]:
#         floe_l = Floe(left_edge=self.left_edge, length=length)
#         cf_l = FloeCoupled(floe_l, self.ice, self.phases, self.amp_coefficients)

#         floe_r = Floe(left_edge=self.left_edge + length, length=self.length - length)
#         phases_r = cf_l.phases + self.ice.wavenumbers * floe_l.length
#         cf_r = FloeCoupled(
#             floe_r,
#             self.ice,
#             phases_r,
#             self.amp_coefficients * np.exp(-self.ice.attenuations * cf_l.length),
#         )
#         return cf_l, cf_r

#     def _ener_min(self, length, spectrum, growth_params, an_sol, num_params) -> float:
#         """Objective function to minimise for energy-based fracture"""
#         cf_l, cf_r = self._binary_split(length)
#         growth_params_r = (
#             (growth_params[0] - length, growth_params[1])
#             if growth_params is not None
#             else None
#         )

#         en_l, en_r = (
#             _f.energy(spectrum, _gp, an_sol, num_params)
#             for _f, _gp in zip((cf_l, cf_r), (growth_params, growth_params_r))
#         )
#         return np.log(en_l + en_r)

#     def fracture(
#         self, xfs: np.ndarray | float
#     ) -> tuple[FloeCoupled, list[FloeCoupled]]:
#         xfs = np.asarray(xfs) + self.left_edge  # domain reference frame
#         left_edges = np.hstack((self.left_edge, xfs))
#         right_edges = np.hstack((xfs, self.right_edge))
#         lengths = right_edges - left_edges
#         phases = np.vstack(
#             (self.phases, self.ice.wavenumbers * lengths[:-1, None])
#         ).cumsum(axis=0)
#         amp_coefficients = np.exp(
#             np.vstack(
#                 (
#                     np.log(self.amp_coefficients),
#                     -self.ice.attenuations * lengths[:-1, None],
#                 )
#             ).cumsum(axis=0)
#         )

#         # TODO instead of instantiating FloeCoupled objects, return iterators
#         # on the parameters, so that the phases can be altered before
#         # instantiation and the need for a setter can be removed
#         return self, [
#             FloeCoupled(Floe(left_edge, length), self.ice, phases_, coefs_, gen_)
#             for left_edge, length, phases_, coefs_, gen_ in zip(
#                 left_edges, lengths, phases,
#                 amp_coefficients, (self.gen + 1, self.gen)
#             )
#         ]


@attrs.define
class Experiment:
    time: float
    domain: md.Domain
    history: dict = attrs.field(init=False, factory=dict)
    fracture_handler: fh.FractureHandler = attrs.field(default=fh.BinaryFracture())

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
    def from_discrete(
        cls,
        gravity: float,
        spectrum: md.DiscreteSpectrum,
        ocean: md.Ocean,
        growth_params: tuple,
        fracture_handler: fh.BinaryFracture = None,
    ):

        domain = md.Domain.from_discrete(gravity, spectrum, ocean, growth_params)
        if fracture_handler is None:
            return cls(0, domain)
        return cls(0, domain, fracture_handler)

    def add_floes(self, floes: md.Floe | Sequence[md.Floe]):
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
            tuple(
                md.WavesUnderFloe(wuf.wui, wuf.floe, wuf.edge_amplitudes)
                for wuf in self.domain.subdomains
            ),
            (
                (self.domain.growth_params[0].copy(), self.domain.growth_params[1])
                if self.domain.growth_params is not None
                else None
            ),
        )

    def step(self, delta_time: float, an_sol=None, num_params=None):
        self.domain.breakup(self.fracture_handler, an_sol, num_params)
        self.domain.iterate(delta_time)
        self.time += delta_time
        self.save_step()
