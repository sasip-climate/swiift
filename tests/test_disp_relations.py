from hypothesis import given, settings
import numpy as np

from swiift.model.model import (
    DiscreteSpectrum,
    FreeSurfaceWaves,
    Ice,
    Ocean,
    WavesUnderElasticPlate,
)

from tests.model_strategies import coupled_ocean_ice, ocean_and_mono_spectrum, spec_mono


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


# Use a monochromatic spectrum, which sould not be limiting as
# polychromatic DiscreteSpectrum objects are just collections
# of independent Wave objects
@given(**ocean_and_mono_spectrum)
def test_free_surface(ocean, spectrum, gravity):
    angfreqs2 = spectrum._ang_freqs_pow2
    fsw = FreeSurfaceWaves.from_ocean(ocean, spectrum, gravity)
    x = free_surface(fsw.wavenumbers, ocean.depth)
    y = angfreqs2 / gravity
    assert np.allclose(x * ocean.depth, y * ocean.depth)


@given(**(coupled_ocean_ice | {"spec": spec_mono()}))
@settings(max_examples=500)
def test_elas_mass_loading(
    ocean: Ocean, spec: DiscreteSpectrum, ice: Ice, gravity: float
):
    assert ocean.density > ice.density
    assert ocean.depth - ice.density / ocean.density * ice.thickness > 0
    angfreqs2 = spec._ang_freqs_pow2
    wui = WavesUnderElasticPlate.from_ocean(ice, ocean, spec, gravity)
    x = elas_mass_surface(wui.wavenumbers, ice, ocean, gravity)
    y = angfreqs2 / gravity
    assert np.allclose(x, y)
