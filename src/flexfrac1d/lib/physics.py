#!/usr/bin/env python3

import attrs
from collections import namedtuple
from collections.abc import Iterator
import functools
import numpy as np
import scipy.optimize as optimize
import scipy.signal as signal

from ..flexfrac1d import Floe, WavesUnderFloe
from .curvature import curvature
from .displacement import displacement
from .energy import energy
from .constants import PI_2


def _package_wuf(wuf: WavesUnderFloe, growth_params):
    floe_params = wuf.wui.ice._red_elastic_number, wuf.floe.length
    wave_params = wuf.edge_amplitudes, wuf.wui._c_wavenumbers
    if growth_params is not None:
        l_growth_params = growth_params[0] - wuf.floe.left_edge, growth_params[1]
    return floe_params, wave_params, l_growth_params


@attrs.define
class DisplacementHandler:
    floe_params: tuple[float]
    wave_params: tuple[np.ndarray]
    growth_params: list[np.ndarray, float] | None

    @classmethod
    def from_wuf(cls, wuf: WavesUnderFloe, growth_params=None):
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
    def from_wuf(cls, wuf: WavesUnderFloe, growth_params=None):
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
    def from_wuf(cls, wuf: WavesUnderFloe, growth_params=None):
        factor = wuf.wui.ice.flex_rigidity / (2 * wuf.wui.ice.thickness)
        return cls(*_package_wuf(wuf, growth_params), factor)

    def compute(self, an_sol, num_params):
        unit_energy = energy(
            self.floe_params, self.wave_params, self.growth_params, an_sol, num_params
        )
        return self.factor * unit_energy


@attrs.define
class BinaryFracture:
    coef_nd: int = 4

    def split(self, wuf, length) -> tuple[WavesUnderFloe]:
        sub_left = WavesUnderFloe(
            wuf.wui,
            Floe(left_edge=wuf.floe.left_edge, length=length),
            wuf.edge_amplitudes,
        )
        sub_right = WavesUnderFloe(
            wuf.wui,
            Floe(
                left_edge=wuf.floe.left_edge + length, length=wuf.floe.length - length
            ),
            wuf.edge_amplitudes * np.exp(1j * wuf.wui._c_wavenumbers * length),
        )
        return sub_left, sub_right

    def compute_energies(
        self, wuf_collection, growth_params, an_sol, num_params
    ) -> tuple[float]:
        energies = np.full(len(wuf_collection), np.nan)
        for i, wuf in enumerate(wuf_collection):
            handler = EnergyHandler.from_wuf(wuf, growth_params)
            energies[i] = handler.compute(an_sol, num_params)
        return energies

    def _ener_min(self, wuf, length, growth_params, an_sol, num_params) -> float:
        """Objective function to minimise for energy-based fracture"""
        sub_left, sub_right = self.split(length)
        energy_left, energy_right = self.compute_energies(
            self.split(wuf, length), growth_params, an_sol, num_params
        )
        return np.log(energy_left + energy_right)

    def diagnose(
        self,
        wuf: WavesUnderFloe,
        res: float = 0.5,
        growth_params=None,
        an_sol=False,
        num_params=None,
    ):
        floe_length = wuf.floe.length
        lengths = np.linspace(
            0, floe_length, np.ceil(floe_length / res).astype(int) + 1
        )[1:-1]
        energies = np.full((lengths.size, 2), np.nan)
        for i, length in enumerate(lengths):
            energies[i, :] = self.compute_energies(
                self.split(wuf, length), growth_params, an_sol, num_params
            )
        frac_diag = namedtuple("FractureDiagnostic", ("length", "energy"))
        return frac_diag(lengths, energies)

    def discrete_sweep(
        self, wuf, an_sol, growth_params, num_params
    ) -> Iterator[tuple[float]]:
        nd = (
            np.ceil(4 * self.length * self.ice.wavenumbers.max() / PI_2).astype(int) + 2
        )
        lengths = np.linspace(0, self.length, nd * self.coef_nd)[1:-1]
        ener = np.full(lengths.shape, np.nan)
        for i, length in enumerate(lengths):
            ener[i] = self._ener_min(length, growth_params, an_sol, num_params)

        peak_idxs = np.hstack(
            (0, signal.find_peaks(np.log(ener), distance=2)[0], ener.size - 1)
        )
        return zip(lengths[peak_idxs[:-1]], lengths[peak_idxs[1:]])

    def search(
        self, wuf: WavesUnderFloe, growth_params, an_sol, num_params
    ) -> float | None:
        base_handler = EnergyHandler.from_wuf(wuf)
        base_energy = base_handler.compute(growth_params, an_sol, num_params)

        # No fracture if the elastic energy is below the threshold
        if base_energy < wuf.wui.ice.frac_energy_rate:
            return None
        else:
            bounds_iterator = self.discrete_sweep(
                wuf, an_sol, growth_params, num_params
            )
            local_ener_cost = functools.partial(
                self._ener_min,
                growth_params=growth_params,
                an_sol=an_sol,
                num_params=num_params,
            )
            opts = [
                optimize.minimize_scalar(local_ener_cost, bounds=bounds)
                for bounds in bounds_iterator
            ]
            opt = min(filter(lambda opt: opt.success, opts), key=lambda opt: opt.fun)
            # Minimisation is done on the log of energy
            if np.exp(opt.fun) + self.ice.frac_energy_rate < base_energy:
                return opt.x
            else:
                return None
