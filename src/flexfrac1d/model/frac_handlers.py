#!/usr/bin/env python3

import abc
import attrs
from collections import namedtuple
from collections.abc import Iterator, Sequence
import functools
from numbers import Real
import numpy as np
import scipy.optimize as optimize
import scipy.signal as signal

from ..lib.constants import PI_2
from . import model
from ..lib import physics as ph


def _make_search_array(wuf: model.WavesUnderFloe, coef: int):
    nd = np.ceil(4 * wuf.length * wuf.wui.wavenumbers.max() / PI_2).astype(int) + 2
    return np.linspace(0, wuf.length, nd * coef)[1:-1]


def _make_diagnose_array(wuf: model.WavesUnderFloe, res: float):
    return np.linspace(0, wuf.length, np.ceil(wuf.length / res).astype(int) + 1)


@attrs.define(frozen=True)
class _StrainDiag:
    x: np.ndarray
    strain: np.ndarray
    peaks: np.ndarray
    strain_extrema: np.ndarray


@attrs.define(frozen=True)
class _FractureDiag:
    x: np.ndarray
    energy: np.ndarray


@attrs.define
class FractureHandler(abc.ABC):
    coef_nd: int = 4

    def split(
        self, wuf: model.WavesUnderFloe, xf: Real | np.ndarray
    ) -> list[model.WavesUnderFloe]:
        xf = np.hstack((0, xf))
        lengths = np.ediff1d(np.hstack((xf, wuf.length)))
        edges = wuf.left_edge + xf
        edge_amplitudes = np.exp(1j * wuf.wui._c_wavenumbers * xf[:, None])
        gens = wuf.generation * np.ones(xf.size)
        gens[:-1] += 1
        return [
            model.WavesUnderFloe(
                left_edge=edge,
                length=lgth,
                wui=wuf.wui,
                edge_amplitudes=amplitudes,
                generation=gen,
            )
            for edge, lgth, amplitudes, gen in zip(
                edges, lengths, edge_amplitudes, gens
            )
        ]

    @abc.abstractmethod
    def search(self, wuf: model.WavesUnderFloe, growth_params, an_sol, num_params):
        raise NotImplementedError


@attrs.define
class BinaryFracture(FractureHandler):
    def compute_energies(
        self,
        wuf_collection: Sequence[model.WavesUnderFloe],
        growth_params,
        an_sol: bool,
        num_params,
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
        lengths = _make_diagnose_array(wuf, res)[1:-1]
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
        lengths = _make_search_array(wuf, self.coef_nd)
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
        base_handler = ph.EnergyHandler.from_wuf(wuf, growth_params)
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
            return None


@attrs.define
class StrainFracture(FractureHandler):
    def discrete_sweep(
        self, strain_handler, wuf, growth_params, an_sol, num_params
    ) -> Iterator[tuple[float]]:
        x = _make_search_array(wuf, self.coef_nd)
        strain = strain_handler.compute(x, an_sol, num_params)
        peak_idxs = np.hstack((0, signal.find_peaks(-(strain**2))[0], x.size - 1))
        return zip(x[peak_idxs[:-1]], x[peak_idxs[1:]])

    def search_peaks(
        self,
        wuf: model.WavesUnderFloe,
        growth_params: tuple | None,
        an_sol: bool,
        num_params: dict | None,
    ) -> Iterator[optimize.OptimizeResult]:
        strain_handler = ph.StrainHandler.from_wuf(wuf, growth_params)
        bounds_iterator = self.discrete_sweep(
            strain_handler, wuf, growth_params, an_sol, num_params
        )
        opts = (
            optimize.minimize_scalar(
                lambda x: -strain_handler.compute(x, an_sol, num_params) ** 2,
                bounds=bounds,
            )
            for bounds in bounds_iterator
        )
        return filter(lambda opt: opt.success, opts)

    def diagnose(
        self,
        wuf: model.WavesUnderFloe,
        res: float = 0.5,
        growth_params: tuple | None = None,
        an_sol: bool = True,
        num_params: dict | None = None,
    ) -> _StrainDiag:
        x = _make_diagnose_array(wuf, res)
        strain_handler = ph.StrainHandler.from_wuf(wuf, growth_params)
        opts = self.search_peaks(wuf, growth_params, an_sol, num_params)
        peaks = np.array([opt.x for opt in opts])
        return _StrainDiag(
            x,
            strain_handler.compute(x, an_sol, num_params),
            peaks,
            strain_handler.compute(peaks, an_sol, num_params),
        )


@attrs.define
class BinaryStrainFracture(StrainFracture):
    def search(
        self,
        wuf: model.WavesUnderFloe,
        growth_params,
        an_sol,
        num_params,
    ) -> float | None:
        opts = self.search_peaks(wuf, growth_params, an_sol, num_params)
        opt = min(opts, key=lambda opt: opt.fun)
        if (-opt.fun) ** 0.5 >= wuf.wui.ice.strain_threshold:
            return opt.x
        return None


@attrs.define
class MultipleStrainFracture(StrainFracture):
    def search(
        self,
        wuf: model.WavesUnderFloe,
        growth_params,
        an_sol,
        num_params,
    ) -> list[float] | None:
        opts = self.search_peaks(wuf, growth_params, an_sol, num_params)
        xfs = [
            opt.x
            for opt in filter(
                lambda opt: (-opt.fun) ** 0.5 >= wuf.wui.ice.strain_threshold, opts
            )
        ]
        if len(xfs) > 0:
            return xfs
        return None

    def diagnose(
        self,
        wuf: model.WavesUnderFloe,
        res: float = 0.5,
        growth_params=None,
        an_sol: bool = False,
        num_params=None,
    ): ...
