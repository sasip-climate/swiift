"""Pseudo-scattering parameterisations."""

from __future__ import annotations

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
        """Determine post-breakup wave amplitudes at the edge of new floes.

        The wave propagates and is attenuated underneath the floe. For the
        current timestep, for each wave component, the complex wave amplitude
        is fully determined at all the coordinates where fracture is about to
        occur. After fractures have occured, the complex amplitudes at these
        coordinates become the complex amplitudes at the left edges of new
        fragments. Further pseudo-scattering rules can be used if it is not
        desirable to keep the wave surface in phase on both sides of a floe
        edge.


        Parameters
        ----------
        edge_amplitudes : np.ndarray of complex
            The complex wave amplitudes at the edge of a breaking floe, in m
        c_wavenumbers : np.ndarray of complex
            The complex wavenumbers stressing the floe, in m^-1
        xf : np.ndarray of float
            The coordinates of fractures, in m

        Returns
        -------
        np.ndarray of complex

        """
        pass


class ContinuousScatteringHandler(ScatteringHandler):
    """No scattering.

    The surface stays continuous across floes edges.

    """

    @staticmethod
    def compute_edge_amplitudes(
        edge_amplitudes,
        c_wavenumbers: np.ndarray,
        xf: np.ndarray,
    ) -> np.ndarray:
        return edge_amplitudes * np.exp(1j * c_wavenumbers * xf[:, None])


@attrs.frozen
class UniformScatteringHandler(ScatteringHandler):
    r"""Scattering with uniformly sampled new phases.

    The wave phase at the edge of a new floe is sampled from the uniform
    distribution on :math:`[0; 2\pi)`.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator used to sample phases


    ------
                            )]

    """

    rng: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int) -> typing.Self:
        """Instantiate self with an RNG seeded by an integer.

        Parameters
        ----------
        seed : int
            A seed passed to `numpy.random.default_rng`

        Returns
        -------
        UniformScatteringHandler

        """
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
    """Scattering with phases perturbated around the continuous solution.

    The wave phase at the left edge of a new floe is computed to maintain
    continuity of the surface across the edge. Then, a random perturbation
    sampled from  a normal distribution, is added to the phase.

    Attributes
    ----------
    rng : numpy.random.Generator
        Random generator used to sample perturbations
    loc : Real
        The mean of the normal distribution used to sample perturbations,
        in rad
    scale : Real
        The standard deviation of the normal distribution used to sample
        perturbations, in rad

    Notes
    -----
    Perturbations are always added to an existing phase. The expectation of the
    resulting phase is thus the sum of `loc` and the phase of the continuous
    solution.

    """

    rng: np.random.Generator
    loc: Real
    scale: Real

    @classmethod
    def from_seed(cls, seed: int, loc: Real = 0, scale: Real = 1) -> typing.Self:
        """Instantiate with an RNG seeded with an integer.

        Parameters
        ----------
        seed : int
            A seed passed to `numpy.random.default_rng`
        loc : Real
            Mean of a normal distribution, in rad
        scale : Real
            Standard deviation of a normal distribution, in rad

        Returns
        -------
        typing.Self
            [TODO:description]

        """
        rng = _seed_rng(seed)
        return cls(rng, loc, scale)

    def compute_edge_amplitudes(
        self,
        edge_amplitudes: np.ndarray,
        c_wavenumbers: np.ndarray,
        xf: np.ndarray,
    ) -> np.ndarray:
        edge_amplitudes = ContinuousScatteringHandler.compute_edge_amplitudes(
            edge_amplitudes, c_wavenumbers, xf
        )
        perturbations = self.rng.normal(
            self.loc, self.scale, size=edge_amplitudes.shape
        )
        edge_amplitudes *= np.exp(1j * perturbations)
        return edge_amplitudes
