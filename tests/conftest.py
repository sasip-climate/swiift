import numpy as np
from hypothesis import strategies as st
from flexfrac1d.model.model import Ocean, DiscreteSpectrum, Ice, Floe
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
    return draw(st.floats(0.1e-3, min(1000, upper_bound), exclude_max=True, **float_kw))


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
        "poissons_ratio": st.floats(-0.999, 0.5, **float_kw),
        "strain_threshold": st.floats(1e-7, 1e-3, **float_kw),
        "thickness": st.floats(1e-3, 1e3, **float_kw),
        "youngs_modulus": st.floats(1e6, 100e9, **float_kw),
        "elastic_length": st.floats(
            5e-4, 3e4, **float_kw
        ),  # derived from the bounds on its constituents
    },
    "wave": {
        "amplitude": st.floats(1e-6, 1e3, **float_kw),
        "period": st.floats(min_value=1e-1, max_value=1e4, **float_kw),
        "frequency": st.floats(min_value=1e-4, max_value=10, **float_kw),
        "phase": st.floats(0, PI_2, exclude_max=True, **float_kw),
        "wavenumber": st.floats(
            7e-4, 600, **float_kw
        ),  # covers waves from about 10 cm to 10 km
    },
    "gravity": st.floats(0.1, 30, **float_kw),
}


physical_strategies["floe"]["length"] = floe_length
physical_strategies["ice"]["density_coupled"] = ice_density
physical_strategies["ice"]["thickness_coupled"] = ice_thickness


@st.composite
def spec_mono(draw):
    return DiscreteSpectrum(
        draw(st.just(0.5)), draw(physical_strategies["wave"]["frequency"])
    )


@st.composite
def spec_poly(draw):
    n = draw(st.integers(min_value=2, max_value=10))
    amplitudes = draw(
        st.lists(
            physical_strategies["wave"]["amplitude"],
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    frequencies = draw(
        st.lists(
            physical_strategies["wave"]["frequency"],
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    # phases = draw(
    #     st.lists(
    #         physical_strategies["wave"]["phase"],
    #         min_size=n,
    #         max_size=n,
    #         unique=True,
    #     )
    # )
    return DiscreteSpectrum(amplitudes, frequencies)


ocean_and_mono_spectrum = {
    "ocean": st.builds(
        Ocean,
        depth=physical_strategies["ocean"]["depth"],
        density=physical_strategies["ocean"]["density"],
    ),
    "spectrum": spec_mono(),
    "gravity": physical_strategies["gravity"],
}

ocean_and_poly_spectrum = {
    "ocean": st.builds(
        Ocean,
        depth=physical_strategies["ocean"]["depth"],
        density=physical_strategies["ocean"]["density"],
    ),
    "spectrum": spec_poly(),
    "gravity": physical_strategies["gravity"],
}

ocean_and_spectrum = {
    "ocean": st.builds(
        Ocean,
        depth=physical_strategies["ocean"]["depth"],
        density=physical_strategies["ocean"]["density"],
    ),
    "spectrum": spec_mono() | spec_poly(),
    "gravity": physical_strategies["gravity"],
}

coupled_ocean_ice = {
    "ocean": st.builds(
        Ocean,
        depth=st.shared(physical_strategies["ocean"]["depth"], key="H"),
        density=st.shared(physical_strategies["ocean"]["density"], key="rhow"),
    ),
    "ice": st.shared(
        st.builds(
            Ice,
            density=st.shared(
                physical_strategies["ice"]["density_coupled"](
                    st.shared(physical_strategies["ocean"]["density"], key="rhow")
                ),
                key="rhoi",
            ),
            frac_toughness=physical_strategies["ice"]["frac_toughness"],
            poissons_ratio=physical_strategies["ice"]["poissons_ratio"],
            thickness=physical_strategies["ice"]["thickness_coupled"](
                ocean_depth=st.shared(physical_strategies["ocean"]["depth"], key="H"),
                ocean_density=st.shared(
                    physical_strategies["ocean"]["density"], key="rhow"
                ),
                ice_density=st.shared(
                    physical_strategies["ice"]["density_coupled"](
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

simple_objects = {
    "ocean": Ocean(),
    "ice": Ice(),
    "spec_mono": DiscreteSpectrum(0.5, 1 / 7),
    "spec_poly": DiscreteSpectrum((0.1, 0.2), (1 / 7, 1 / 5)),
    "gravity": 9.8,
    "length": 101.5,
    "left_edge": 13.8,
}
