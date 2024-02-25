#!/usr/bin/env python3

from hypothesis import given, settings, strategies as st
import numpy as np

from flexfrac1d.flexfrac1d import Ocean, OceanCoupled, DiscreteSpectrum
from flexfrac1d.flexfrac1d import Ice, IceCoupled
from flexfrac1d.libraries.WaveUtils import free_surface, elas_mass_surface

from .conftest import physical_strategies
from .conftest import coupled_ocean_ice, spec_mono

# Bounds set to be sure to avoid non-sensical overflows and underflows,
# when multiplying and dividing various variables together:
# we want m < (2 pi)^2 x^4 < M,
# where m := float.tiny, M := float.max, x the sampled value
# float_kw = {
#     "min_value": 3e-77,
#     "max_value": 4e76,
#     "allow_nan": False,
#     "exclude_min": True,
#     "allow_subnormal": False,
# }


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
    co = OceanCoupled(ocean, spec, gravity)
    x = free_surface(co.wavenumbers, co.depth)
    y = angfreqs2 / gravity
    assert np.allclose(x * co.depth, y * co.depth)


# float_kw = {
#     "min_value": 5e-39,
#     "max_value": 4e51,
#     "allow_nan": False,
#     "exclude_min": True,
#     "allow_subnormal": False,
# }


@given(
    **(coupled_ocean_ice | {"spec": spec_mono()})
    # ocean=st.builds(
    #     Ocean,
    #     depth=st.shared(physical_strategies["ocean"]["depth"], key="H"),
    #     density=st.shared(physical_strategies["ocean"]["density"], key="rhow"),
    # ),
    # spec=st.builds(
    #     DiscreteSpectrum,
    #     st.just(1),
    #     physical_strategies["wave"]["wave_frequency"],
    # ),
    # ice=st.builds(
    #     Ice,
    #     density=st.shared(
    #         physical_strategies["ice"]["density"](
    #             st.shared(physical_strategies["ocean"]["density"], key="rhow")
    #         ),
    #         key="rhoi",
    #     ),
    #     frac_energy=physical_strategies["ice"]["frac_energy"],
    #     poissons_ratio=physical_strategies["ice"]["poissons_ratio"],
    #     thickness=physical_strategies["ice"]["thickness"](
    #         ocean_depth=st.shared(physical_strategies["ocean"]["depth"], key="H"),
    #         ocean_density=st.shared(
    #             physical_strategies["ocean"]["density"], key="rhow"
    #         ),
    #         ice_density=st.shared(
    #             physical_strategies["ice"]["density"](
    #                 st.shared(physical_strategies["ocean"]["density"], key="rhow")
    #             ),
    #             key="rhoi",
    #         ),
    #     ),
    #     youngs_modulus=physical_strategies["ice"]["youngs_modulus"],
    # ),
    # gravity=physical_strategies["gravity"],
)
@settings(max_examples=500)
def test_elas_mass_surface(
    ocean: Ocean, spec: DiscreteSpectrum, ice: Ice, gravity: float
):
    assert ocean.density > ice.density
    assert ocean.depth - ice.density / ocean.density * ice.thickness > 0
    angfreqs2 = spec._ang_freq2
    co = OceanCoupled(ocean, spec, gravity)
    ci = IceCoupled(ice, co, spec, None, gravity)
    x = elas_mass_surface(ci.wavenumbers, ice, ocean, gravity)
    y = angfreqs2 / gravity
    assert np.allclose(x, y)
