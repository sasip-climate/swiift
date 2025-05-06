from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence

import attrs
import numpy as np

from ..lib import att
from ..model import frac_handlers as fh, model as md

# TODO: make into an attrs class for more flexibility (repr of subdomains)
Step = namedtuple("Step", ["subdomains", "growth_params"])


@attrs.define
class Experiment:
    time: float
    domain: md.Domain
    history: dict[float, Step] = attrs.field(init=False, factory=dict, repr=False)
    fracture_handler: fh._FractureHandler = attrs.field(factory=fh.BinaryFracture)

    @classmethod
    def from_discrete(
        cls,
        gravity: float,
        spectrum: md.DiscreteSpectrum,
        ocean: md.Ocean,
        growth_params: tuple | None = None,
        fracture_handler: fh._FractureHandler | None = None,
        attenuation_spec: att.Attenuation | None = None,
    ):
        if attenuation_spec is None:
            attenuation_spec = att.AttenuationParameterisation(1)
        domain = md.Domain.from_discrete(
            gravity, spectrum, ocean, attenuation_spec, growth_params
        )

        if fracture_handler is None:
            return cls(0, domain)
        return cls(0, domain, fracture_handler)

    def add_floes(self, floes: md.Floe | Sequence[md.Floe]):
        self.domain.add_floes(floes)
        self.save_step()

    def last_step(self):
        return self.history[next(reversed(self.history))]

    def save_step(self):
        self.history[self.time] = Step(
            tuple(wuf.make_copy() for wuf in self.domain.subdomains),
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

    def get_steps(self, times: np.ndarray | float) -> dict[float, Step]:
        """Return a subset of the history matching the given times.

        Parameters
        ----------
        times : 1D array_like, float
            Time, or sequence of times.

        Returns
        -------
        dict[float, Step]
            A dictionary containing the `Step`s closest to the input `times`.

        """
        times = np.ravel(times)
        timestep_keys = np.array(list(self.history.keys()))
        indexes = (np.abs(times - timestep_keys[:, None])).argmin(axis=0)
        return {k: self.history[k] for k in timestep_keys[indexes]}

    def serialize(self, fname):
        pass
