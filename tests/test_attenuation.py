from hypothesis import given
import hypothesis.extra.numpy as npst
import numpy as np

import swiift.lib.att as att
from swiift.model.model import FloatingIce, WavesUnderElasticPlate, WavesUnderIce
from tests.model_strategies import coupled_ocean_ice

wavenumbers_strategy = npst.arrays(
    float,
    npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=10),
    elements=PHYSICAL_STRATEGIES[("wave", "wavenumber")],
)


class TestNoAttenuation:
    @staticmethod
    def test_unit():
        assert att.no_attenuation() == 0

    @staticmethod
    @given(**coupled_ocean_ice, wavenumbers=wavenumbers_strategy)
    def test_integrated(ice, ocean, gravity, wavenumbers):
        fi = FloatingIce.from_ice_ocean(ice, ocean, gravity)
        wup = WavesUnderElasticPlate(fi, wavenumbers)
        wui = WavesUnderIce.without_attenuation(wup)
        assert wui.attenuations == 0


class TestParam01:
    @staticmethod
    @given(
        thickness=PHYSICAL_STRATEGIES[("ice", "thickness")],
        wavenumbers=wavenumbers_strategy,
    )
    def test_unit(thickness, wavenumbers):
        attenuations = wavenumbers**2 * thickness / 4
        assert np.allclose(
            attenuations - att.parameterisation_01(thickness, wavenumbers), 0
        )

    @staticmethod
    @given(**coupled_ocean_ice, wavenumbers=wavenumbers_strategy)
    def test_integrated(ice, ocean, gravity, wavenumbers):
        fi = FloatingIce.from_ice_ocean(ice, ocean, gravity)
        wup = WavesUnderElasticPlate(fi, wavenumbers)
        wui = WavesUnderIce.with_attenuation_01(wup)
        assert np.allclose(
            wui.attenuations - att.parameterisation_01(ice.thickness, wavenumbers), 0
        )


class TestGeneric:
    # Test passing a generic attenuation function against the existing
    # parameterisations.

    @staticmethod
    @given(**coupled_ocean_ice, wavenumbers=wavenumbers_strategy)
    def test_no_attenuation(ice, ocean, gravity, wavenumbers):
        fi = FloatingIce.from_ice_ocean(ice, ocean, gravity)
        wup = WavesUnderElasticPlate(fi, wavenumbers)
        wui = WavesUnderIce.with_generic_attenuation(wup, lambda: 0)
        wui_ref = WavesUnderIce.without_attenuation(wup)
        assert wui.attenuations == wui_ref.attenuations

    @staticmethod
    @given(**coupled_ocean_ice, wavenumbers=wavenumbers_strategy)
    def test_param01_args(ice, ocean, gravity, wavenumbers):
        # Test Param01 providing arguments as a string
        fi = FloatingIce.from_ice_ocean(ice, ocean, gravity)
        wup = WavesUnderElasticPlate(fi, wavenumbers)
        wui = WavesUnderIce.with_generic_attenuation(
            wup,
            lambda thickness, wavenumbers: wavenumbers**2 * thickness / 4,
            "ice.thickness wavenumbers",
        )
        wui_ref = WavesUnderIce.with_attenuation_01(wup)
        assert np.allclose(wui.attenuations - wui_ref.attenuations, 0)

    @staticmethod
    @given(**coupled_ocean_ice, wavenumbers=wavenumbers_strategy)
    def test_param01_kwargs(ice, ocean, gravity, wavenumbers):
        # Test Param01 providing arguments as a dict
        fi = FloatingIce.from_ice_ocean(ice, ocean, gravity)
        wup = WavesUnderElasticPlate(fi, wavenumbers)
        wui = WavesUnderIce.with_generic_attenuation(
            wup,
            lambda thickness, wavenumbers: wavenumbers**2 * thickness / 4,
            **{"thickness": ice.thickness, "wavenumbers": wavenumbers},
        )
        wui_ref = WavesUnderIce.with_attenuation_01(wup)
        assert np.allclose(wui.attenuations - wui_ref.attenuations, 0)
