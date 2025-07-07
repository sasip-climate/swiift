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


def setup_experiment() -> api.Experiment:
    amplitude = 2
    period = 7
    spectrum = DiscreteSpectrum(amplitude, 1 / period)
    depth = np.inf
    ocean = Ocean(depth=depth)
    gravity = 9.8
    return Experiment.from_discrete(gravity, spectrum, ocean)


def setup_experiment_with_floe() -> tuple[api.Experiment, Floe]:
    experiment = setup_experiment()
    thickness = 0.5
    ice = Ice(thickness=thickness)
    floe = Floe(left_edge=0, length=200, ice=ice)
    experiment.add_floes(floe)
    return experiment, floe


def step_experiment(experiment: api.Experiment, delta_t: float) -> api.Experiment:
    experiment.step(delta_t)
    return experiment


@pytest.mark.parametrize("dir_to_create", ("tmp_dir", pathlib.Path("tmp_dir2")))
def test_create_directory(tmp_path: pathlib.Path, dir_to_create: str | pathlib.Path):
    target_path = tmp_path.joinpath(dir_to_create)
    path = api._create_path(target_path)
    assert path.exists()
    path2 = api._create_path(target_path)
    assert path == path2


@pytest.mark.parametrize("step", (False, True))
def test_simple_read(mocker: MockerFixture, step):
    experiment = setup_experiment()
    step_size = 10  # simply to test we do recover different instance properties
    # HACK: remove this line once DiscreteSpectrum has been attrs'd.
    # For now, needed for equality test.
    experiment.domain = None
    if step:
        experiment.time = 10
    file_content = io.BytesIO(pickle.dumps(experiment))
    mocker.patch("builtins.open", return_value=file_content)
    loaded_result = api._load_pickle("dummy.pickle")
    assert experiment == loaded_result
    if step:
        assert loaded_result.time == step_size


def test_read_wrong_type(mocker: MockerFixture):
    experiment = 1.12
    file_content = io.BytesIO(pickle.dumps(experiment))
    mocker.patch("builtins.open", return_value=file_content)
    with pytest.raises(TypeError):
        _ = api._load_pickle("dummy.pickle")


@pytest.mark.parametrize("use_glob", (True, False))
def test_file_error(use_glob: bool):
    fname = "exper_test.pickle"
    with pytest.raises(FileNotFoundError):
        if not use_glob:
            api.load_pickle(fname)
        else:
            api.load_pickles(fname)


@pytest.mark.parametrize("loading_option", loading_options)
def test_load_pickles(loading_option: str, monkeypatch):
    path_as_str = epxeriment_targets_path
    path = pathlib.Path(path_as_str)
    experiments = [api._load_pickle(_p) for _p in sorted(path.glob(fname_pattern))]
    if loading_option == "str":
        experiment = api.load_pickles(fname_pattern, path_as_str)
    elif loading_options == "path":
        experiment = api.load_pickles(fname_pattern, path)
    else:
        # Reading from cwd. To be able to read, we chdir to the path we want
        # first.
        monkeypatch.chdir(epxeriment_targets_path)
        experiment = api.load_pickles(fname_pattern)

    # Check the expected length. The read length should match the sum of the
    # individually loaded length, minus (total of experiment minus 1), as the
    # last key of a saved file should match the first key of the next one.
    assert len(experiment.history) == (
        sum(len(_exper.history) for _exper in experiments) - (len(experiments) - 1)
    )
    # Check the first history entry matches the first entry of the first history saved
    assert next(iter(experiment.history)) == next(iter(experiments[0].history))
    # Check the last history entry matches the last entry of the last history saved
    assert experiment.time == experiments[-1].time


@pytest.mark.parametrize("do_recursive", (True, False))
def test_recursive_load(do_recursive: bool):
    if do_recursive:
        path = pathlib.Path("/".join(epxeriment_targets_path.split("/")[:-1]))
    else:
        path = pathlib.Path(epxeriment_targets_path)
    _ = api.load_pickles(fname_pattern, path, do_recursive)


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
    experiment, floe = setup_experiment_with_floe()

    assert len(experiment.history) == 1
    assert len(experiment.domain.subdomains) == 1

    # NOTE: use an integer here to avoid floating point precision issues down the line
    delta_t = 1
    experiment = step_experiment(experiment, delta_t)
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


@pytest.mark.parametrize("delta_t", (0.1, 0.5, 1, 1.5))
def test_get_timesteps(delta_t):
    experiment, _ = setup_experiment_with_floe()
    n_steps = 4
    target_times = np.linspace(0, n_steps, n_steps + 1) * delta_t
    for i in range(n_steps):
        experiment = step_experiment(experiment, delta_t)
    times = experiment.timesteps
    assert np.allclose(target_times, times)
