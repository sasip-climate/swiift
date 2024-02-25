#!/usr/bin/env python3

import numpy as np
from hypothesis import strategies as st

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


# All SI units
physical_strategies = {
    "ocean": {
        "depth": (
            st.floats(min_value=1e-3, max_value=1000e3, **float_kw) | st.just(np.inf)
        ),
        "density": st.floats(500, 5e3, **float_kw),
    },
    "ice": {
        "frac_energy": st.floats(**float_kw),
        "poissons_ratio": st.floats(-1, 0.5, exclude_min=True, **float_kw),
        "youngs_modulus": st.floats(1e6, 100e9, **float_kw),
    },
    "wave": {
        "wave_frequency": st.floats(min_value=1e-4, max_value=10, **float_kw),
        "wave_phase": st.floats(-np.pi, np.pi, exclude_min=True, **float_kw),
        "wave_amplitude": st.floats(1e-6, 1e3, **float_kw),
    },
    "gravity": st.floats(0.1, 30, **float_kw),
}


physical_strategies["ice"]["density"] = ice_density
physical_strategies["ice"]["thickness"] = ice_thickness
