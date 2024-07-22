from hypothesis import given
import hypothesis.extra.numpy as npst
import numpy as np

from .conftest import physical_strategies, coupled_ocean_ice
from flexfrac1d.model.model import FloatingIce
from flexfrac1d.model.model import WavesUnderIce, WavesUnderElasticPlate
import flexfrac1d.lib.att as att

wavenumbers_strategy = npst.arrays(
    float,
    npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=10),
    elements=physical_strategies["wave"]["wavenumber"],
)


class TestNoAttenuation:
    def test_unit(self):
        assert att.no_attenuation() == 0

    @given(**coupled_ocean_ice, wavenumbers=wavenumbers_strategy)
    def test_integrated(self, ice, ocean, gravity, wavenumbers):
        fi = FloatingIce.from_ice_ocean(ice, ocean, gravity)
        wup = WavesUnderElasticPlate(fi, wavenumbers)
        wui = WavesUnderIce.from_ep_no_attenuation(wup)
        assert wui.attenuations == 0


class TestParam01:
    @given(
        thickness=physical_strategies["ice"]["thickness"],
        wavenumbers=wavenumbers_strategy,
    )
    def test_unit(self, thickness, wavenumbers):
        attenuations = wavenumbers**2 * thickness / 4
        assert np.allclose(
            attenuations - att.parameterisation_01(thickness, wavenumbers), 0
        )

    @given(**coupled_ocean_ice, wavenumbers=wavenumbers_strategy)
    def test_integrated(self, ice, ocean, gravity, wavenumbers):
        fi = FloatingIce.from_ice_ocean(ice, ocean, gravity)
        wup = WavesUnderElasticPlate(fi, wavenumbers)
        wui = WavesUnderIce.from_ep_attenuation_param_01(wup)
        assert np.allclose(
            wui.attenuations - att.parameterisation_01(ice.thickness, wavenumbers), 0
        )


class TestGeneric:
    # Test passing a generic attenuation function against the existing
    # parameterisations.

    @given(**coupled_ocean_ice, wavenumbers=wavenumbers_strategy)
    def test_no_attenuation(self, ice, ocean, gravity, wavenumbers):
        fi = FloatingIce.from_ice_ocean(ice, ocean, gravity)
        wup = WavesUnderElasticPlate(fi, wavenumbers)
        wui = WavesUnderIce.from_ep_generic_attenuation_param(wup, lambda: 0)
        wui_ref = WavesUnderIce.from_ep_no_attenuation(wup)
        assert wui.attenuations == wui_ref.attenuations

    @given(**coupled_ocean_ice, wavenumbers=wavenumbers_strategy)
    def test_param01_args(self, ice, ocean, gravity, wavenumbers):
        # Test Param01 providing arguments as a string
        fi = FloatingIce.from_ice_ocean(ice, ocean, gravity)
        wup = WavesUnderElasticPlate(fi, wavenumbers)
        wui = WavesUnderIce.from_ep_generic_attenuation_param(
            wup,
            lambda thickness, wavenumbers: wavenumbers**2 * thickness / 4,
            "ice.thickness wavenumbers",
        )
        wui_ref = WavesUnderIce.from_ep_attenuation_param_01(wup)
        assert np.allclose(wui.attenuations - wui_ref.attenuations, 0)

    @given(**coupled_ocean_ice, wavenumbers=wavenumbers_strategy)
    def test_param01_kwargs(self, ice, ocean, gravity, wavenumbers):
        # Test Param01 providing arguments as a dict
        fi = FloatingIce.from_ice_ocean(ice, ocean, gravity)
        wup = WavesUnderElasticPlate(fi, wavenumbers)
        wui = WavesUnderIce.from_ep_generic_attenuation_param(
            wup,
            lambda thickness, wavenumbers: wavenumbers**2 * thickness / 4,
            **{"thickness": ice.thickness, "wavenumbers": wavenumbers},
        )
        wui_ref = WavesUnderIce.from_ep_attenuation_param_01(wup)
        assert np.allclose(wui.attenuations - wui_ref.attenuations, 0)
