#!/usr/bin/env python3

from hypothesis import given, settings, strategies as st
import numpy as np

from flexfrac1d.flexfrac1d import Ice, Ocean, DiscreteSpectrum
from flexfrac1d.flexfrac1d import FreeSurfaceWave, WaveUnderIce
from flexfrac1d.lib.disprel import free_surface, elas_mass_surface

from .conftest import physical_strategies
from .conftest import coupled_ocean_ice, spec_mono


# Use a monochromatic spectrum, which sould not be limiting as
# polychromatic DiscreteSpectrum objects are just collections
# of independent Wave objects
@given(
    ocean=st.builds(
        Ocean,
        depth=physical_strategies["ocean"]["depth"],
        density=physical_strategies["ocean"]["density"],
    ),
    spec=st.builds(
        DiscreteSpectrum,
        st.just(1),
        physical_strategies["wave"]["wave_frequency"],
    ),
    gravity=physical_strategies["gravity"],
)
def test_free_surface(ocean, spec, gravity):
    angfreqs2 = np.array([wave.angular_frequency2 for wave in spec.waves])
    fsw = FreeSurfaceWave.from_ocean(ocean, spec, gravity)
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
    angfreqs2 = spec._ang_freq2
    # co = OceanCoupled(ocean, spec, gravity)
    # ci = IceCoupled(ice, co, spec, None, gravity)
    wui = WaveUnderIce.from_ocean(ice, ocean, spec, gravity)
    x = elas_mass_surface(wui.wavenumbers, ice, ocean, gravity)
    # x = elas_mass_surface(ci.wavenumbers, ice, ocean, gravity)
    y = angfreqs2 / gravity
    assert np.allclose(x, y)
