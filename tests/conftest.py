#!/usr/bin/env python3

import numpy as np
from hypothesis import strategies as st
from flexfrac1d.flexfrac1d import Ocean, DiscreteSpectrum, Ice, Floe
from flexfrac1d.lib.constants import PI_2

# Generic float options
float_kw = {
    "allow_nan": False,
    "allow_subnormal": False,
}


# For the composite strategies, set somewhat stricter upper bounds than plain
# inequality to avoid head-scratching floating point innacuracies
@st.composite
def ice_density(draw, ocean_density):
    ex = draw(ocean_density)
    return draw(st.floats(10, 0.9999 * ex, **float_kw))


@st.composite
def ice_thickness(draw, ocean_density, ocean_depth, ice_density):
    kwgs = {"rhow": ocean_density, "rhoi": ice_density, "H": ocean_depth}
    ex = {k: draw(v) for k, v in kwgs.items()}
    upper_bound = 0.9999 * ex["rhow"] / ex["rhoi"] * ex["H"]
    return draw(
        st.floats(0.1e-3, min(1000e3, upper_bound), exclude_max=True, **float_kw)
    )


@st.composite
def floe_length(draw, ice: Ice) -> float:
    return draw(st.floats(2 * draw(ice).thickness, 1000e3, **float_kw))


# All SI units
physical_strategies = {
    "floe": {
        "left_edge": st.floats(0, 5e3, **float_kw),
    },
    "ocean": {
        "depth": (
            st.floats(min_value=1e-3, max_value=1000e3, **float_kw) | st.just(np.inf)
        ),
        "density": st.floats(500, 5e3, **float_kw),
    },
    "ice": {
        "frac_toughness": st.floats(1e3, 1e7, **float_kw),
        "poissons_ratio": st.floats(-1, 0.5, exclude_min=True, **float_kw),
        "youngs_modulus": st.floats(1e6, 100e9, **float_kw),
    },
    "wave": {
        "wave_amplitude": st.floats(1e-6, 1e3, **float_kw),
        "wave_period": st.floats(min_value=1e-1, max_value=1e4, **float_kw),
        "wave_frequency": st.floats(min_value=1e-4, max_value=10, **float_kw),
        "wave_phase": st.floats(0, PI_2, exclude_max=True, **float_kw),
    },
    "gravity": st.floats(0.1, 30, **float_kw),
}


physical_strategies["floe"]["length"] = floe_length
physical_strategies["ice"]["density"] = ice_density
physical_strategies["ice"]["thickness"] = ice_thickness


@st.composite
def spec_mono(draw):
    return DiscreteSpectrum(
        draw(st.just(0.5)), draw(physical_strategies["wave"]["wave_frequency"])
    )


@st.composite
def spec_poly(draw):
    n = draw(st.integers(min_value=1, max_value=100))
    amplitudes = draw(
        st.lists(
            physical_strategies["wave"]["wave_amplitude"],
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    frequencies = draw(
        st.lists(
            physical_strategies["wave"]["wave_frequency"],
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    # phases = draw(
    #     st.lists(
    #         physical_strategies["wave"]["wave_phase"],
    #         min_size=n,
    #         max_size=n,
    #         unique=True,
    #     )
    # )
    return DiscreteSpectrum(amplitudes, frequencies)


coupled_ocean_ice = {
    "ocean": st.builds(
        Ocean,
        depth=st.shared(physical_strategies["ocean"]["depth"], key="H"),
        density=st.shared(physical_strategies["ocean"]["density"], key="rhow"),
    ),
    # "spec": st.builds(
    #     DiscreteSpectrum,
    #     st.just(1),
    #     physical_strategies["wave"]["wave_frequency"],
    # ),
    "ice": st.shared(
        st.builds(
            Ice,
            density=st.shared(
                physical_strategies["ice"]["density"](
                    st.shared(physical_strategies["ocean"]["density"], key="rhow")
                ),
                key="rhoi",
            ),
            frac_toughness=physical_strategies["ice"]["frac_toughness"],
            poissons_ratio=physical_strategies["ice"]["poissons_ratio"],
            thickness=physical_strategies["ice"]["thickness"](
                ocean_depth=st.shared(physical_strategies["ocean"]["depth"], key="H"),
                ocean_density=st.shared(
                    physical_strategies["ocean"]["density"], key="rhow"
                ),
                ice_density=st.shared(
                    physical_strategies["ice"]["density"](
                        st.shared(physical_strategies["ocean"]["density"], key="rhow")
                    ),
                    key="rhoi",
                ),
            ),
            youngs_modulus=physical_strategies["ice"]["youngs_modulus"],
        ),
        key="ice",
    ),
    "gravity": physical_strategies["gravity"],
}


coupled_floe = {
    "floe": st.builds(
        Floe,
        left_edge=physical_strategies["floe"]["left_edge"],
        length=physical_strategies["floe"]["length"](
            st.shared(coupled_ocean_ice["ice"], key="ice")
        ),
        ice=st.shared(coupled_ocean_ice["ice"], key="ice"),
    )
} | coupled_ocean_ice

# coupled_ocean_ice2 = {
#     "ocean": st.builds(
#         Ocean,
#         depth=st.shared(physical_strategies["ocean"]["depth"], key="H"),
#         density=st.shared(physical_strategies["ocean"]["density"], key="rhow"),
#     ),
#     "ice": st.builds(
#         Ice,
#         density=st.shared(
#             physical_strategies["ice"]["density"](
#                 st.shared(physical_strategies["ocean"]["density"], key="rhow")
#             ),
#             key="rhoi",
#         ),
#         frac_energy=physical_strategies["ice"]["frac_energy"],
#         poissons_ratio=physical_strategies["ice"]["poissons_ratio"],
#         thickness=physical_strategies["ice"]["thickness"](
#             ocean_depth=st.shared(physical_strategies["ocean"]["depth"], key="H"),
#             ocean_density=st.shared(
#                 physical_strategies["ocean"]["density"], key="rhow"
#             ),
#             ice_density=st.shared(
#                 physical_strategies["ice"]["density"](
#                     st.shared(physical_strategies["ocean"]["density"], key="rhow")
#                 ),
#                 key="rhoi",
#             ),
#         ),
#         youngs_modulus=physical_strategies["ice"]["youngs_modulus"],
#     ),
#     "gravity": physical_strategies["gravity"],
# }
