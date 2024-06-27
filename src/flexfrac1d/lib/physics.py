#!/usr/bin/env python3

import attrs
import numpy as np

from .curvature import curvature
from .displacement import displacement
from .energy import energy
from ..model import model


def _package_wuf(wuf: model.WavesUnderFloe, growth_params):
    floe_params = wuf.wui.ice._red_elastic_number, wuf.floe.length
    wave_params = wuf.edge_amplitudes, wuf.wui._c_wavenumbers
    if growth_params is not None:
        growth_params = growth_params[0] - wuf.floe.left_edge, growth_params[1]
    return floe_params, wave_params, growth_params


@attrs.define
class DisplacementHandler:
    floe_params: tuple[float]
    wave_params: tuple[np.ndarray]
    growth_params: list[np.ndarray, float] | None

    @classmethod
    def from_wuf(cls, wuf: model.WavesUnderFloe, growth_params=None):
        return cls(*_package_wuf(wuf, growth_params))

    def compute(self, x, an_sol, num_params):
        return displacement(
            x,
            self.floe_params,
            self.wave_params,
            self.growth_params,
            an_sol,
            num_params,
        )


@attrs.define
class CurvatureHandler:
    floe_params: tuple[float]
    wave_params: tuple[np.ndarray]
    growth_params: list[np.ndarray, float] | None

    @classmethod
    def from_wuf(cls, wuf: model.WavesUnderFloe, growth_params=None):
        return cls(*_package_wuf(wuf, growth_params))

    def compute(self, x, an_sol, num_params):
        return curvature(
            x,
            self.floe_params,
            self.wave_params,
            self.growth_params,
            an_sol,
            num_params,
        )


@attrs.define
class EnergyHandler:
    floe_params: tuple[float]
    wave_params: tuple[np.ndarray]
    growth_params: list[np.ndarray, float] | None
    factor: float

    @classmethod
    def from_wuf(cls, wuf: model.WavesUnderFloe, growth_params=None):
        factor = wuf.wui.ice.flex_rigidity / (2 * wuf.wui.ice.thickness)
        return cls(*_package_wuf(wuf, growth_params), factor)

    def compute(self, an_sol, num_params):
        unit_energy = energy(
            self.floe_params, self.wave_params, self.growth_params, an_sol, num_params
        )
        return self.factor * unit_energy
