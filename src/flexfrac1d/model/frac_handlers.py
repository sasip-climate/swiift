#!/usr/bin/env python3

import abc
import attrs
from collections import namedtuple
from collections.abc import Iterator
import functools
import numpy as np
import scipy.optimize as optimize
import scipy.signal as signal

from ..lib.constants import PI_2
from . import model
from ..lib import physics as ph


class FractureHandler(abc.ABC):
    pass


@attrs.define
class BinaryFracture:
    coef_nd: int = 4

    def split(self, wuf, length) -> tuple[model.WavesUnderFloe]:
        sub_left = model.WavesUnderFloe(
            left_edge=wuf.left_edge,
            length=wuf.length,
            wui=wuf.wui,
            edge_amplitudes=wuf.edge_amplitudes,
            generation=wuf.generation + 1,
        )
        sub_right = model.WavesUnderFloe(
            wui=wuf.wui,
            left_edge=wuf.left_edge + length,
            length=wuf.length - length,
            edge_amplitudes=(
                wuf.edge_amplitudes * np.exp(1j * wuf.wui._c_wavenumbers * length)
            ),
            generation=wuf.generation,
        )
        return sub_left, sub_right

    def compute_energies(
        self, wuf_collection, growth_params, an_sol, num_params
    ) -> tuple[float]:
        energies = np.full(len(wuf_collection), np.nan)
        for i, wuf in enumerate(wuf_collection):
            handler = ph.EnergyHandler.from_wuf(wuf, growth_params)
            energies[i] = handler.compute(an_sol, num_params)
        return energies

    def _ener_min(self, length, wuf, growth_params, an_sol, num_params) -> float:
        """Objective function to minimise for energy-based fracture"""
        sub_left, sub_right = self.split(wuf, length)
        energy_left, energy_right = self.compute_energies(
            self.split(wuf, length), growth_params, an_sol, num_params
        )
        return np.log(energy_left + energy_right)

    def diagnose(
        self,
        wuf: model.WavesUnderFloe,
        res: float = 0.5,
        growth_params=None,
        an_sol=False,
        num_params=None,
    ):
        floe_length = wuf.length
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
        nd = np.ceil(4 * wuf.length * wuf.wui.wavenumbers.max() / PI_2).astype(int) + 2
        lengths = np.linspace(0, wuf.length, nd * self.coef_nd)[1:-1]
        ener = np.full(lengths.shape, np.nan)
        for i, length in enumerate(lengths):
            ener[i] = self._ener_min(length, wuf, growth_params, an_sol, num_params)

        peak_idxs = np.hstack(
            (0, signal.find_peaks(np.log(ener), distance=2)[0], ener.size - 1)
        )
        return zip(lengths[peak_idxs[:-1]], lengths[peak_idxs[1:]])

    def search(
        self, wuf: model.WavesUnderFloe, growth_params, an_sol, num_params
    ) -> float | None:
        base_handler = ph.EnergyHandler.from_wuf(wuf)
        base_energy = base_handler.compute(an_sol, num_params)

        # No fracture if the elastic energy is below the threshold
        if base_energy < wuf.wui.ice.frac_energy_rate:
            return None
        else:
            bounds_iterator = self.discrete_sweep(
                wuf, an_sol, growth_params, num_params
            )
            local_ener_cost = functools.partial(
                self._ener_min,
                wuf=wuf,
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
            if np.exp(opt.fun) + wuf.wui.ice.frac_energy_rate < base_energy:
                return opt.x
            else:
                return None
