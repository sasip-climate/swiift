import numpy as np

from ..model.model import Ice, Ocean


def free_surface(wavenumber, depth):
    return wavenumber * np.tanh(wavenumber * depth)


def elas_mass_surface(
    wavenumbers: np.ndarray, ice: Ice, ocean: Ocean, gravity: float
) -> np.ndarray:
    l4 = ice.flex_rigidity / (ocean.density * gravity)
    draft = ice.density / ocean.density * ice.thickness
    dud = ocean.depth - draft
    k_tanh_kdud = wavenumbers * np.tanh(wavenumbers * dud)

    return (l4 * wavenumbers**4 + 1) / (1 + draft * k_tanh_kdud) * k_tanh_kdud
