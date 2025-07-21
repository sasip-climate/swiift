"""Define strategies to construct model objects."""

from hypothesis import strategies as st

from swiift.model.model import DiscreteSpectrum, Floe, Ice, Ocean
from tests.physical_strategies import (
    PHYSICAL_STRATEGIES,
    PHYSICAL_STRATEGIES_COMPOSITE,
)


@st.composite
def spec_mono(draw: st.DrawFn) -> DiscreteSpectrum:
    return DiscreteSpectrum(
        draw(st.just(0.5)), draw(PHYSICAL_STRATEGIES[("wave", "frequency")])
    )


@st.composite
def spec_poly(draw: st.DrawFn) -> DiscreteSpectrum:
    n = draw(st.integers(min_value=2, max_value=10))
    amplitudes = draw(
        st.lists(
            PHYSICAL_STRATEGIES[("wave", "amplitude")],
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    frequencies = draw(
        st.lists(
            PHYSICAL_STRATEGIES[("wave", "frequency")],
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    return DiscreteSpectrum(amplitudes, frequencies)


ocean_and_mono_spectrum = {
    "ocean": st.builds(
        Ocean,
        depth=PHYSICAL_STRATEGIES[("ocean", "depth")],
        density=PHYSICAL_STRATEGIES[("ocean", "density")],
    ),
    "spectrum": spec_mono(),
    "gravity": PHYSICAL_STRATEGIES[("gravity",)],
}

ocean_and_poly_spectrum = {
    "ocean": st.builds(
        Ocean,
        depth=PHYSICAL_STRATEGIES[("ocean", "depth")],
        density=PHYSICAL_STRATEGIES[("ocean", "density")],
    ),
    "spectrum": spec_poly(),
    "gravity": PHYSICAL_STRATEGIES[("gravity",)],
}

ocean_and_spectrum = {
    "ocean": st.builds(
        Ocean,
        depth=PHYSICAL_STRATEGIES[("ocean", "depth")],
        density=PHYSICAL_STRATEGIES[("ocean", "density")],
    ),
    "spectrum": spec_mono() | spec_poly(),
    "gravity": PHYSICAL_STRATEGIES[("gravity",)],
}

coupled_ocean_ice = {
    "ocean": st.builds(
        Ocean,
        depth=st.shared(PHYSICAL_STRATEGIES[("ocean", "depth")], key="H"),
        density=st.shared(PHYSICAL_STRATEGIES[("ocean", "density")], key="rhow"),
    ),
    "ice": st.shared(
        st.builds(
            Ice,
            density=st.shared(
                PHYSICAL_STRATEGIES_COMPOSITE["ice_density"](
                    st.shared(PHYSICAL_STRATEGIES[("ocean", "density")], key="rhow")
                ),
                key="rhoi",
            ),
            frac_toughness=PHYSICAL_STRATEGIES[("ice", "frac_toughness")],
            poissons_ratio=PHYSICAL_STRATEGIES[("ice", "poissons_ratio")],
            thickness=PHYSICAL_STRATEGIES_COMPOSITE["ice_thickness"](
                ocean_density=st.shared(
                    PHYSICAL_STRATEGIES[("ocean", "density")], key="rhow"
                ),
                ocean_depth=st.shared(PHYSICAL_STRATEGIES[("ocean", "depth")], key="H"),
                ice_density=st.shared(
                    PHYSICAL_STRATEGIES_COMPOSITE["ice_density"](
                        st.shared(PHYSICAL_STRATEGIES[("ocean", "density")], key="rhow")
                    ),
                    key="rhoi",
                ),
            ),
            youngs_modulus=PHYSICAL_STRATEGIES[("ice", "youngs_modulus")],
        ),
        key="ice",
    ),
    "gravity": PHYSICAL_STRATEGIES[("gravity",)],
}

coupled_floe = {
    "floe": st.builds(
        Floe,
        left_edge=PHYSICAL_STRATEGIES[("floe", "left_edge")],
        length=PHYSICAL_STRATEGIES_COMPOSITE["floe_length"](
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

