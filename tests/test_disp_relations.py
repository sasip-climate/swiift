#!/usr/bin/env python3

from hypothesis import given, strategies as st
import numpy as np

from flexfrac1d.flexfrac1d import Ocean, OceanCoupled, DiscreteSpectrum
from flexfrac1d.libraries.WaveUtils import free_surface


# Bounds set to be sure to avoid non-sensical overflows and underflows,
# when multiplying and dividing various variables together
float_kw = {
    "min_value": 4e-78,
    "max_value": 4e76,
    "allow_nan": False,
    "exclude_min": True,
    "allow_subnormal": False,
}


# Use a monochromatic spectrum, which sould not be limiting as
# polychromatic DiscreteSpectrum objects are just collections
# of independent Wave objects
@given(
    ocean=st.builds(Ocean, st.floats(**float_kw), st.floats(**float_kw)),
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
