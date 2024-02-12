#!/usr/bin/env python3

from hypothesis import assume, given, strategies as st
import numpy as np

from flexfrac1d.flexfrac1d import Ocean, OceanCoupled, DiscreteSpectrum
from flexfrac1d.flexfrac1d import Ice, IceCoupled
from flexfrac1d.libraries.WaveUtils import free_surface, elas_mass_surface


# Bounds set to be sure to avoid non-sensical overflows and underflows,
# when multiplying and dividing various variables together:
# we want m < (2 pi)^2 x^4 < M,
# where m := float.tiny, M := float.max, x the sampled value
float_kw = {
    "min_value": 3e-77,
    "max_value": 4e76,
    "allow_nan": False,
    "exclude_min": True,
    "allow_subnormal": False,
}


# Use a monochromatic spectrum, which sould not be limiting as
# polychromatic DiscreteSpectrum objects are just collections
# of independent Wave objects
@given(
    ocean=st.builds(
        Ocean,
        depth=st.floats(**float_kw) | st.just(np.inf),
        density=st.floats(**float_kw),
    ),
    spec=st.builds(
        DiscreteSpectrum,
        st.just(1),
        st.floats(**float_kw),
    ),
    gravity=st.floats(**float_kw),
)
def test_free_surface(ocean, spec, gravity):
    angfreqs2 = np.array([wave.angular_frequency2 for wave in spec.waves])
    co = OceanCoupled(ocean, spec, gravity)
    x = free_surface(co.wavenumbers, co.depth)
    y = angfreqs2 / gravity
    assert np.allclose(x * co.depth, y * co.depth)


float_kw = {
    "min_value": 5e-39,
    "max_value": 4e51,
    "allow_nan": False,
    "exclude_min": True,
    "allow_subnormal": False,
}


@given(
    ocean=st.builds(
        Ocean,
        depth=st.floats(**float_kw) | st.just(np.inf),
        density=st.floats(**float_kw),
    ),
    spec=st.builds(
        DiscreteSpectrum,
        st.just(1),
        st.floats(**float_kw),
    ),
    ice=st.builds(
        Ice,
        density=st.floats(**float_kw),
        frac_energy=st.floats(**float_kw),
        poissons_ratio=st.floats(
            min_value=-1,
            max_value=0.5,
            allow_nan=False,
            allow_subnormal=False,
            exclude_min=True,
        ),
        thickness=st.floats(**float_kw),
        youngs_modulus=st.floats(**float_kw),
    ),
    gravity=st.floats(**float_kw),
)
def test_elas_mass_surface(
    ocean: Ocean, spec: DiscreteSpectrum, ice: Ice, gravity: float
):
    assume(ocean.density > ice.density)
    # Has to be done manually as instantiation of the IceCoupled object can
    # fail if not respected
    assume(ocean.depth - ice.thickness * ice.density / ocean.density > 0)
    angfreqs2 = spec._ang_freq2
    co = OceanCoupled(ocean, spec, gravity)
    ci = IceCoupled(ice, co, spec, None, gravity)
    x = elas_mass_surface(ci.wavenumbers, ice, ocean, gravity)
    y = angfreqs2 / gravity
    # elastic_length = (ice.flex_rigidity / (ocean.density * gravity)) ** 0.25
    # assert np.allclose(x * elastic_length, y * elastic_length)
    assert np.allclose(x, y)
