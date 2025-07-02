import io
import pathlib
import pickle

from hypothesis import given
import numpy as np
import pytest
from pytest_mock import MockerFixture
from sortedcontainers import SortedList

import swiift.api.api as api
from swiift.api.api import Experiment
import swiift.lib.att as att
import swiift.lib.phase_shift as ps
import swiift.model.frac_handlers as fh
from swiift.model.model import DiscreteSpectrum, Domain, Floe, Ice, Ocean

from .conftest import coupled_ocean_ice, ocean_and_mono_spectrum, spec_mono

fracture_handlers = (
    fh.BinaryFracture,
    fh.BinaryStrainFracture,
    fh.MultipleStrainFracture,
)
attenuation_parameterisations = att.AttenuationParameterisation
growth_params = (None, (-13, None), (-28, 75), (np.array([-45]), None))


def make_dummy_experiment():
    gravity = 9.8
    spectrum = DiscreteSpectrum(1, 1)
    ocean = Ocean()
    return Experiment.from_discrete(gravity, spectrum, ocean)


def test_create_path(tmp_path: pathlib.Path):
    path = api._create_path(tmp_path)
    assert path.exists()


@pytest.mark.parametrize("step", (False, True))
def test_simple_read(mocker: MockerFixture, step):
    experiment = make_dummy_experiment()
    step_size = 10  # simply to test we do recover different instance properties
    # HACK: remove this line once DiscreteSpectrum has been attrs'd
    experiment.domain = None
    if step:
        experiment.time = 10
    file_content = io.BytesIO(pickle.dumps(experiment))
    mocker.patch("builtins.open", return_value=file_content)
    loaded_result = api._read_pickle("dummy.pickle")
    assert experiment == loaded_result
    if step:
        assert loaded_result.time == step_size




@given(**ocean_and_mono_spectrum)
def test_initialisation(gravity, spectrum, ocean):
    experiment = Experiment.from_discrete(gravity, spectrum, ocean)

    assert experiment.time == 0
    assert isinstance(experiment.domain, Domain)
    assert experiment.domain.growth_params is None
    assert (
        isinstance(experiment.domain.subdomains, SortedList)
        and len(experiment.domain.subdomains) == 0
    )
    assert (
        isinstance(experiment.domain.attenuation, att.AttenuationParameterisation)
        and experiment.domain.attenuation == att.AttenuationParameterisation.PARAM_01
    )
    assert isinstance(
        experiment.fracture_handler.scattering_handler, ps.ContinuousScatteringHandler
    )
    assert isinstance(experiment.history, dict) and len(experiment.history) == 0
    assert isinstance(experiment.fracture_handler, fh.BinaryFracture)


@given(**ocean_and_mono_spectrum)
@pytest.mark.parametrize("growth_params", growth_params)
@pytest.mark.parametrize("fracture_handler_type", fracture_handlers)
@pytest.mark.parametrize("att_spec", att.AttenuationParameterisation)
def test_initialisation_with_opt_params(
    gravity,
    spectrum,
    ocean,
    growth_params,
    fracture_handler_type,
    att_spec,
):
    fracture_handler = fracture_handler_type()
    experiment = Experiment.from_discrete(
        gravity,
        spectrum,
        ocean,
        growth_params=growth_params,
        fracture_handler=fracture_handler,
        attenuation_spec=att_spec,
    )

    if growth_params is None:
        assert experiment.domain.growth_params is None
    else:
        assert len(experiment.domain.growth_params) == 2
        assert experiment.domain.growth_params[0] == growth_params[0]
        assert experiment.domain.growth_params[1] is not None
    assert isinstance(experiment.fracture_handler, fracture_handler_type)
    assert isinstance(experiment.domain.attenuation, att.AttenuationParameterisation)
    assert experiment.domain.attenuation == att_spec


@given(spectrum=spec_mono(), **coupled_ocean_ice)
def test_add_floes_single(gravity, spectrum, ocean, ice):
    floe = Floe(left_edge=0, length=100, ice=ice)
    experiment = Experiment.from_discrete(gravity, spectrum, ocean)
    assert len(experiment.history) == 0
    assert len(experiment.domain.subdomains) == 0
    assert len(experiment.domain.cached_wuis) == 0
    experiment.add_floes(floe)
    assert len(experiment.history) == 1
    assert len(experiment.domain.subdomains) == 1
    assert experiment.domain.subdomains[0].left_edge == floe.left_edge
    assert experiment.domain.subdomains[0].length == floe.length
    assert ice in experiment.domain.cached_wuis
    assert experiment.history[0].subdomains[0] == experiment.domain.subdomains[0]


@given(spectrum=spec_mono(), **coupled_ocean_ice)
def test_add_floes_collection(gravity, spectrum, ocean, ice):
    floe1 = Floe(left_edge=0, length=100, ice=ice)
    floe2 = Floe(left_edge=100, length=100, ice=ice)
    experiment = Experiment.from_discrete(gravity, spectrum, ocean)
    experiment.add_floes((floe1, floe2))
    assert len(experiment.history) == 1
    assert len(experiment.history[0].subdomains) == 2
    assert len(experiment.domain.subdomains) == 2


@given(spectrum=spec_mono(), **coupled_ocean_ice)
def test_add_floes_overlap(gravity, spectrum, ocean, ice):
    floe1 = Floe(left_edge=0, length=100, ice=ice)
    floe2 = Floe(left_edge=80, length=100, ice=ice)
    experiment = Experiment.from_discrete(gravity, spectrum, ocean)
    with pytest.raises(ValueError):
        experiment.add_floes((floe1, floe2))


def total_length_comparison(subdomains, floe: Floe):
    total_length = sum(wuf.length for wuf in subdomains)
    return np.allclose(total_length - floe.length, 0)


def test_step():
    amplitude = 2
    period = 7
    spectrum = DiscreteSpectrum(amplitude, 1 / period)
    thickness = 0.5
    ice = Ice(thickness=thickness)
    depth = np.inf
    ocean = Ocean(depth=depth)
    gravity = 9.8

    experiment = Experiment.from_discrete(gravity, spectrum, ocean)
    floe = Floe(left_edge=0, length=200, ice=ice)
    experiment.add_floes(floe)
    assert len(experiment.history) == 1
    assert len(experiment.domain.subdomains) == 1

    # NOTE: use an integer here to avoid floating point precision issues down the line
    delta_t = 1
    experiment.step(delta_t, True)
    assert np.allclose(experiment.time - delta_t, 0)
    assert len(experiment.history) == 2
    assert (
        len(experiment.domain.subdomains) == 2
    )  # this floe should definitely have fractured in these conditions
    assert total_length_comparison(experiment.domain.subdomains, floe)
    assert delta_t in experiment.domain.cached_phases

    number_of_additional_steps = 5
    for _ in range(number_of_additional_steps):
        experiment.step(delta_t)

    assert np.allclose(experiment.time - (number_of_additional_steps + 1) * delta_t, 0)
    assert len(experiment.history) == number_of_additional_steps + 2
    assert total_length_comparison(experiment.domain.subdomains, floe)
    last_step = experiment.get_final_state()
    assert experiment.history[(number_of_additional_steps + 1) * delta_t] == last_step
