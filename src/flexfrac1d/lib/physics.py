#!/usr/bin/env python3

import attrs
import functools
import numpy as np

from .curvature import curvature
from .displacement import displacement
from .energy import energy
from ..model import model

# TODO: add a handler for that former FloeCoupled method
#     def forcing(self, x, spectrum, growth_params):
#         return free_surface(x, self._pack(spectrum)[1], growth_params)


def _package_wuf(wuf: model.WavesUnderFloe, growth_params):
    floe_params = wuf.wui.ice._red_elastic_number, wuf.length
    wave_params = wuf.edge_amplitudes, wuf.wui._c_wavenumbers
    if growth_params is not None:
        growth_params = growth_params[0] - wuf.left_edge, growth_params[1]
    return floe_params, wave_params, growth_params


def _demote_to_scalar(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        res = f(self, *args, **kwargs)
        if len(res) == 1:
            return res.item()
        return res

    return wrapper


@attrs.define
class DisplacementHandler:
    floe_params: tuple[float]
    wave_params: tuple[np.ndarray]
    growth_params: list[np.ndarray, float] | None

    @classmethod
    def from_wuf(cls, wuf: model.WavesUnderFloe, growth_params=None):
        return cls(*_package_wuf(wuf, growth_params))

    @_demote_to_scalar
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

    @_demote_to_scalar
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
class StrainHandler:
    curv_handler: CurvatureHandler
    thickness: float

    @classmethod
    def from_wuf(cls, wuf: model.WavesUnderFloe, growth_params=None):
        return cls(CurvatureHandler.from_wuf(wuf, growth_params), wuf.wui.ice.thickness)

    # Doesn't need to be decorated, as relies on CurvatureHandler.compute,
    # which already is
    def compute(self, x, an_sol, num_params):
        return -self.thickness / 2 * self.curv_handler.compute(x, an_sol, num_params)


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
