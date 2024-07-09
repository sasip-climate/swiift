#!/usr/bin/env python3

from __future__ import annotations

import attrs
from collections import namedtuple
from collections.abc import Sequence

from ..model import model as md
from ..model import frac_handlers as fh

Step = namedtuple("Step", ["subdomains", "growth_params"])


@attrs.define
class Experiment:
    time: float
    domain: md.Domain
    history: dict[float, Step] = attrs.field(init=False, factory=dict, repr=False)
    fracture_handler: fh._FractureHandler = attrs.field(default=fh.BinaryFracture())

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
