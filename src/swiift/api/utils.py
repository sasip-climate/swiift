from __future__ import annotations

import typing

import numpy as np

from ..lib import dr

if typing.TYPE_CHECKING:
    from ..model.model import DiscreteSpectrum, FloatingIce, Ocean


def compute_free_surface_wavenumbers(
    ocean: Ocean, spectrum: DiscreteSpectrum, gravity: float
) -> np.ndarray:
    solver = dr.FreeSurfaceSolver.from_ocean(ocean, spectrum, gravity)
    wavenumbers = solver.compute_wavenumbers(real=True)
    return wavenumbers


def compute_elastic_mass_loading_wavenumbers(
    ice: FloatingIce, spectrum: DiscreteSpectrum, gravity: float
) -> np.ndarray:
    solver = dr.ElasticMassLoadingSolver.from_floating(ice, spectrum, gravity)
    wavenumbers = solver.compute_wavenumbers(real=True)
    return wavenumbers
