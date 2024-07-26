"""Pseudo-scattering parameterisations."""

import abc
import attrs
from numbers import Real
import typing
import numpy as np

from .constants import PI_2


def _seed_rng(seed: int):
    return np.random.default_rng(seed)


class ScatteringHandler(abc.ABC):
    @abc.abstractmethod
    def compute_edge_amplitudes(
        self,
        edge_amplitudes: np.ndarray,
        c_wavenumbers: np.ndarray,
        xf: np.ndarray,
    ) -> np.ndarray:
        pass


class ContinuousScatteringHandler(ScatteringHandler):
    @staticmethod
    def compute_edge_amplitudes(
        edge_amplitudes,
        c_wavenumbers: np.ndarray,
        xf: np.ndarray,
    ) -> np.ndarray:
        return edge_amplitudes * np.exp(1j * c_wavenumbers * xf[:, None])


@attrs.frozen
class UniformScatteringHandler(ScatteringHandler):
    rng: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int) -> typing.Self:
        return cls(_seed_rng(seed))

    def compute_edge_amplitudes(
        self,
        edge_amplitudes: np.ndarray,
        c_wavenumbers: np.ndarray,
        xf: np.ndarray,
    ) -> np.ndarray:
        phases = self.rng.uniform(0, PI_2, size=edge_amplitudes.shape)
        return (
            np.abs(edge_amplitudes)
            * np.exp(-np.imag(c_wavenumbers) * xf[:, None])
            * np.exp(1j * phases)
        )


@attrs.frozen
class PerturbationScatteringHandler(ScatteringHandler):
    rng: np.random.Generator
    scale: Real = 1

    @classmethod
    def from_seed(cls, seed: int, scale: Real | None = None) -> typing.Self:
        rng = _seed_rng(seed)
        if scale is None:
            return cls(rng)
        else:
            return cls(rng, scale)

    def compute_edge_amplitudes(
        self,
        edge_amplitudes: np.ndarray,
        c_wavenumbers: np.ndarray,
        xf: np.ndarray,
    ) -> np.ndarray:
        uni_handler = UniformScatteringHandler(self.rng)
        edge_amplitudes = uni_handler.compute_edge_amplitudes(
            edge_amplitudes, c_wavenumbers, xf
        )
        phases = np.angle(edge_amplitudes)
        perturbations = self.rng.normal(0, self.scale, size=edge_amplitudes.shape)
        phases += perturbations
        return np.abs(edge_amplitudes) * np.exp(1j * phases)
