from hypothesis import given
import numpy as np
import pytest

from .conftest import (
    ocean_and_spectrum,
    simple_objects,
)
import flexfrac1d.lib.att as att
from flexfrac1d.model.model import (
    Domain,
    Floe,
    Ocean,
    DiscreteSpectrum,
    FreeSurfaceWaves,
)

growth_params = (None, (-13, None), (-28, 75), (np.array([-45]), None))


def instantiate_domain(att_spec, is_mono) -> None:
    ocean = simple_objects["ocean"]
    if is_mono:
        spectrum = simple_objects["spec_mono"]
    else:
        spectrum = simple_objects["spec_poly"]
    gravity = simple_objects["gravity"]
    return Domain.from_discrete(gravity, spectrum, ocean, attenuation=att_spec)


@given(**ocean_and_spectrum)
def test_initialisation(gravity: float, spectrum: DiscreteSpectrum, ocean: Ocean):
    domain = Domain.from_discrete(gravity, spectrum, ocean)
    fsw = FreeSurfaceWaves.from_ocean(ocean, spectrum, gravity)

    assert domain.gravity == gravity

    assert domain.fsw.ocean == ocean
    assert np.all(domain.fsw.wavenumbers == fsw.wavenumbers)

    # TODO: to reenable when DiscreteSpectrum has been attrs'd
    # assert np.all(domain.spectrum.amplitudes == spectrum.amplitudes)
    # assert np.all(domain.spectrum.frequencies == spectrum.frequencies)
    # assert np.all(domain.spectrum.phases == spectrum.phases)

    assert domain.growth_params is None
    assert domain.attenuation == att.AttenuationParameterisation.PARAM_01

    assert len(domain.cached_wuis) == 0
    assert len(domain.cached_phases) == 0


@given(**ocean_and_spectrum)
def test_failing(gravity: float, spectrum: DiscreteSpectrum, ocean: Ocean):
    with pytest.raises(TypeError):
        Domain.from_discrete(gravity, spectrum, ocean, growth_params=1)

    with pytest.raises(ValueError):
        Domain.from_discrete(gravity, spectrum, ocean, growth_params=(1, 1, 1))

    nf = spectrum.nf
    if nf in (1, 2):
        nf = 3
    else:
        nf -= 1
    means = np.zeros(nf)
    with pytest.raises(ValueError):
        Domain.with_growth_means(gravity, spectrum, ocean, growth_means=means)


def instantiate_floe() -> None:
    ice = simple_objects["ice"]
    return Floe(
        left_edge=simple_objects["left_edge"],
        length=simple_objects["length"],
        ice=ice,
    )


@pytest.mark.parametrize("is_mono", (True, False))
@pytest.mark.parametrize("att_spec", att.AttenuationParameterisation)
def test_att_parameterisations(att_spec, is_mono):
    floe = instantiate_floe()
    domain = instantiate_domain(att_spec, is_mono)
    domain.add_floes(floe)
    assert len(domain.cached_wuis) == 1
    assert floe.ice in domain.cached_wuis


def test_promote():
    floe = instantiate_floe()
    res = Domain._promote_floe(floe)
    assert isinstance(res, tuple)
    assert len(res) == 1
    assert res[0] == floe

    floes = [floe]
    res = Domain._promote_floe(floes)
    assert res == floes

    with pytest.raises(ValueError):
        Domain._promote_floe(1)
