import abc
from collections.abc import Iterator, Sequence
import functools

import attrs
import numpy as np
import scipy.optimize as optimize
import scipy.signal as signal

from . import model
from ..lib import phase_shift as ps, physics as ph
from ..lib.constants import PI_2


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
    initial_energy: float
    frac_energy_rate: float


@attrs.define
class _FractureHandler(abc.ABC):
    coef_nd: int = 4
    scattering_handler: ps._ScatteringHandler = attrs.field(
        factory=ps.ContinuousScatteringHandler
    )

    def split(
        self,
        wuf: model.WavesUnderFloe,
        xf: float | Sequence[float],
        is_searching: bool = False,
    ) -> list[model.WavesUnderFloe]:
        new_relative_edges = np.hstack((0, xf))
        new_absolute_edges = wuf.left_edge + new_relative_edges
        new_lengths = np.ediff1d(np.hstack((new_absolute_edges, wuf.right_edge)))

        if is_searching:
            post_breakup_amplitudes = (
                ps.ContinuousScatteringHandler().compute_edge_amplitudes(
                    wuf.edge_amplitudes, wuf.wui._c_wavenumbers, new_relative_edges
                )
            )
        else:
            post_breakup_amplitudes = np.full(
                (new_relative_edges.size, wuf.edge_amplitudes.size),
                np.nan,
                dtype=complex,
            )
            post_breakup_amplitudes[0] = wuf.edge_amplitudes.copy()
            post_breakup_amplitudes[1:] = (
                self.scattering_handler.compute_edge_amplitudes(
                    wuf.edge_amplitudes, wuf.wui._c_wavenumbers, new_relative_edges[1:]
                )
            )
        # edge_amplitudes = wuf.edge_amplitudes * np.exp(
        #     1j * wuf.wui._c_wavenumbers * xf[:, None]
        # )
        gens = wuf.generation * np.ones(new_relative_edges.size, dtype=int)
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
                new_absolute_edges, new_lengths, post_breakup_amplitudes, gens
            )
        ]

    @abc.abstractmethod
    def search(self, wuf: model.WavesUnderFloe, growth_params, an_sol, num_params):
        raise NotImplementedError


@attrs.define(frozen=True)
class BinaryFracture(_FractureHandler):
    def compute_energies(
        self,
        wuf_collection: Sequence[model.WavesUnderFloe],
        growth_params,
        an_sol: bool,
        num_params,
        linear_curvature: bool | None = None,
    ) -> np.ndarray:
        energies = np.full(len(wuf_collection), np.nan)
        for i, wuf in enumerate(wuf_collection):
            handler = ph.EnergyHandler.from_wuf(wuf, growth_params)
            energies[i] = handler.compute(an_sol, num_params, linear_curvature)
        return energies

    def _ener_min(
        self,
        length,
        wuf,
        growth_params,
        an_sol,
        num_params,
        linear_curvature: bool | None = None,
    ) -> float:
        """Objective function to minimise for energy-based fracture"""
        sub_left, sub_right = self.split(wuf, length)
        energy_left, energy_right = self.compute_energies(
            self.split(wuf, length, True),
            growth_params,
            an_sol,
            num_params,
            linear_curvature,
        )
        return np.log(energy_left + energy_right)

    def diagnose(
        self,
        wuf: model.WavesUnderFloe,
        res: float = 0.5,
        growth_params=None,
        an_sol=None,
        num_params=None,
        linear_curvature: bool | None = None,
    ):
        lengths = _make_diagnose_array(wuf, res)[1:-1]
        energies = np.full((lengths.size, 2), np.nan)
        initial_energy = ph.EnergyHandler.from_wuf(wuf, growth_params).compute(
            an_sol, num_params, linear_curvature
        )
        for i, length in enumerate(lengths):
            energies[i, :] = self.compute_energies(
                self.split(wuf, length),
                growth_params,
                an_sol,
                num_params,
                linear_curvature,
            )
        return _FractureDiag(
            lengths,
            energies,
            initial_energy,
            wuf.wui.ice.frac_energy_rate,
        )

    def discrete_sweep(
        self,
        wuf,
        an_sol,
        growth_params,
        num_params,
        linear_curvature: bool | None = None,
    ) -> Iterator[tuple[float]]:
        lengths = _make_search_array(wuf, self.coef_nd)
        ener = np.full(lengths.shape, np.nan)
        for i, length in enumerate(lengths):
            ener[i] = self._ener_min(
                length, wuf, growth_params, an_sol, num_params, linear_curvature
            )

        peak_idxs = np.hstack(
            (0, signal.find_peaks(ener, distance=2)[0], ener.size - 1)
        )
        return zip(lengths[peak_idxs[:-1]], lengths[peak_idxs[1:]])

    def search(
        self,
        wuf: model.WavesUnderFloe,
        growth_params,
        an_sol,
        num_params,
        linear_curvature: bool | None = None,
    ) -> float | None:
        base_handler = ph.EnergyHandler.from_wuf(wuf, growth_params)
        base_energy = base_handler.compute(an_sol, num_params, linear_curvature)

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
class _StrainFracture(_FractureHandler):
    def discrete_sweep(
        self, strain_handler, wuf, growth_params, an_sol, num_params
    ) -> Iterator[tuple[float]]:
        # NOTE: caveat: some small peaks close to the edges can be missed. The
        # multi-fracture handler is more here as a demo anyway, so this is very
        # low priority.
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


@attrs.define(frozen=True)
class BinaryStrainFracture(_StrainFracture):
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


@attrs.define(frozen=True)
class MultipleStrainFracture(_StrainFracture):
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
