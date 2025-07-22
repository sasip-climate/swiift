from hypothesis import given
import numpy as np
import pytest

import swiift.lib.att as att
import swiift.model.frac_handlers as fh
from swiift.model.model import (
    DiscreteSpectrum,
    Domain,
    Floe,
    FreeSurfaceWaves,
    Ocean,
)
from tests.model_strategies import ocean_and_spectrum, simple_objects
from tests.utils import fracture_handler_types

growth_params = (None, (-13, None), (-28, 75), (np.array([-45]), None))


def instantiate_domain(att_spec: att.Attenuation, is_mono: bool) -> Domain:
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


def instantiate_floe() -> Floe:
    return Floe(
        left_edge=simple_objects["left_edge"],
        length=simple_objects["length"],
        ice=simple_objects["ice"],
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


@pytest.mark.parametrize("is_mono", (True, False))
@pytest.mark.parametrize("att_spec", att.AttenuationParameterisation)
@pytest.mark.parametrize("fracture_handler_type", fracture_handler_types)
def test_breakup(
    att_spec: att.AttenuationParameterisation,
    is_mono: bool,
    fracture_handler_type: type[fh._FractureHandler],
):
    fracture_handler = fracture_handler_type()
    domain = instantiate_domain(att_spec, is_mono)
    floe = instantiate_floe()
    domain.add_floes(floe)
    wuf0 = domain.subdomains[0]
    assert len(domain.subdomains) == 1

    domain.breakup(fracture_handler, an_sol=True)

    # Check we did have some breakup
    match fracture_handler:
        case fh.BinaryFracture() | fh.BinaryStrainFracture():
            assert len(domain.subdomains) == 2
        case fh.MultipleStrainFracture():
            assert len(domain.subdomains) >= 2
        case _:  # pragma: no cover
            raise ValueError("Unknown fracture handler")
    # Check the edge has not moved
    assert domain.subdomains[0].left_edge == wuf0.left_edge

    # Check all new floes except the last have had their generation counter incremented. The last one should have the same generation counter.
    for _wuf in domain.subdomains[:-1]:
        assert _wuf.generation == wuf0.generation + 1
    assert domain.subdomains[-1].generation == wuf0.generation

    # Check individual fragment lengths are less than that of the original floe.
    lengths = np.array([_wuf.length for _wuf in domain.subdomains])
    assert np.all(lengths < wuf0.length)

    # Check floes are in order of their left edges.
    left_edges = np.array([_wuf.left_edge for _wuf in domain.subdomains])
    assert np.all(np.ediff1d(left_edges) > 0)

    # Check the two definitions are equivalent
    relative_new_edges = left_edges[1:] - left_edges[0]
    assert np.allclose(relative_new_edges, lengths[:-1].cumsum())

    # Checked the complex amplitude at the edge is identical, as it should be
    # modified at a later step
    assert np.all(domain.subdomains[0].edge_amplitudes == wuf0.edge_amplitudes)
    # Check new fragments have the expected complex amplitudes at their left
    # edges. As there is no random scattering here, these amplitudes are
    # obtained by "propagating" spatially the original edge amplitudes over the
    # fragment lengths.
    phase_diffs = relative_new_edges[:, None] * (
        wuf0.wui.wavenumbers + 1j * wuf0.wui.attenuations
    )
    assert np.all(
        np.isclose(
            np.vstack([_wuf.edge_amplitudes for _wuf in domain.subdomains[1:]]),
            wuf0.edge_amplitudes * np.exp(1j * phase_diffs),
        )
    )
