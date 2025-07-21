import io
import logging
import pathlib
import pickle

from hypothesis import HealthCheck, given, settings, strategies as st
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
from tests.model_strategies import coupled_ocean_ice, ocean_and_mono_spectrum, spec_mono
from tests.utils import float_kw, fracture_handlers

epxeriment_targets_path = "tests/target/experiments"
fname_pattern = "exper_test*"

attenuation_parameterisations = att.AttenuationParameterisation
growth_params = (None, (-13, None), (-28, 75), (np.array([-45]), None))


loading_options = ("str", "path", "cwd")


class DummyPbar:
    def __init__(self):
        self.updates = 0
        self.closed = False

    def update(self, n):
        self.updates += n

    def close(self):
        self.closed = True

    @classmethod
    def write(cls, msg):
        pass


def mock_breakup(*args):
    return


@st.composite
def run_time_chunks_composite(draw: st.DrawFn) -> tuple[int, float, int]:
    n_step = draw(st.integers(min_value=1, max_value=15))
    delta_time = draw(st.floats(min_value=0.01, max_value=5.0, **float_kw))
    chunk_size = draw(st.integers(min_value=1, max_value=n_step))

    return n_step, delta_time, chunk_size


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


@pytest.fixture(scope="function")
def experiment_with_history() -> api.Experiment:
    return api.load_pickles(fname_pattern, epxeriment_targets_path)


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
    elif loading_option == "path":
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


def test_pre_post_factures(experiment_with_history):
    timesteps = experiment_with_history.timesteps
    pre_times = experiment_with_history.get_pre_fracture_times()
    post_times = experiment_with_history.get_post_fracture_times()

    # Diff between pre- and post-times should be the timestep.
    assert np.allclose(post_times - pre_times, timesteps[1])

    # Diff between number of post- and pre-fracture number of floes should be exactly 1.
    assert np.all(
        np.subtract(
            *[
                np.array(
                    [
                        len(experiment_with_history.history[_t].subdomains)
                        for _t in _times
                    ]
                )
                for _times in (post_times, pre_times)
            ]
        )
        == 1
    )


@given(data=st.data())
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_get_states_strict(data, experiment_with_history: api.Experiment):
    # Cast to list for hypothesis type correctness
    timesteps = experiment_with_history.timesteps.tolist()

    # Draw a single time from timesteps
    single_time = data.draw(st.sampled_from(timesteps), label="single_time")
    result_single = experiment_with_history.get_states(single_time)
    assert isinstance(result_single, dict)
    assert single_time in result_single
    result_single_strict = experiment_with_history.get_states_strict(single_time)
    assert isinstance(result_single_strict, dict)
    assert result_single == result_single_strict

    # Draw a random subset of timesteps (could be empty, single, or multiple)
    subset = data.draw(
        st.lists(st.sampled_from(timesteps), min_size=1, max_size=len(timesteps)),
        label="subset",
    )
    result_list = experiment_with_history.get_states(subset)
    assert isinstance(result_list, dict)
    assert np.all([t in result_list for t in subset])
    result_list_strict = experiment_with_history.get_states(subset)
    assert isinstance(result_list_strict, dict)
    assert result_list == result_list_strict

    # Test with a numpy array of floats
    subset_as_array = np.array(subset)
    result_array = experiment_with_history.get_states(subset_as_array)
    assert isinstance(result_array, dict)
    assert np.all([t in result_array for t in subset])
    result_array_strict = experiment_with_history.get_states_strict(subset)
    assert isinstance(result_array_strict, dict)
    assert result_array == result_array_strict


@given(data=st.data())
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_get_states_perturbated(data, experiment_with_history: api.Experiment):
    perturbation = 1e-3  # delta_time := 5/6 ~ 0.833
    # Cast to list for hypothesis type correctness
    timesteps = experiment_with_history.timesteps.tolist()

    # Draw a single time from timesteps
    single_time = data.draw(st.sampled_from(timesteps), label="single_time")
    perturbated_time = single_time + perturbation
    result_single = experiment_with_history.get_states(perturbated_time)
    assert isinstance(result_single, dict)
    assert single_time in result_single
    result_single = experiment_with_history.get_states_strict(perturbated_time)
    assert isinstance(result_single, dict)
    assert len(result_single) == 0

    # Draw a random subset of timesteps (could be empty, single, or multiple)
    subset = data.draw(
        st.lists(st.sampled_from(timesteps), min_size=1, max_size=len(timesteps)),
        label="subset",
    )
    perturbated_subset = [_v + perturbation for _v in subset]
    result_list = experiment_with_history.get_states(perturbated_subset)  # type: ignore
    assert isinstance(result_list, dict)
    assert np.all([t in result_list for t in subset])
    result_list = experiment_with_history.get_states_strict(perturbated_subset)  # type: ignore
    assert isinstance(result_list, dict)
    assert len(result_list) == 0

    # Test with a numpy array of floats
    perturbated_array = np.array(perturbated_subset)
    result_array = experiment_with_history.get_states(perturbated_array)
    assert isinstance(result_array, dict)
    assert np.all([t in result_array for t in subset])
    result_array = experiment_with_history.get_states_strict(perturbated_array)
    assert isinstance(result_array, dict)
    assert len(result_array) == 0


@pytest.mark.parametrize("with_prefix", (True, False))
def test_history_dump(
    tmp_path: pathlib.Path,
    experiment_with_history: api.Experiment,
    with_prefix: bool,
):
    prefix = "test_prefix" if with_prefix else None
    last_timestep = experiment_with_history.timesteps[-1]
    assert len(experiment_with_history.history) > 1
    experiment_with_history.dump_history(prefix, dir_path=tmp_path)
    assert len(experiment_with_history.history) == 1
    assert last_timestep in experiment_with_history.history
    if with_prefix:
        assert len(list(tmp_path.glob(f"{prefix}*.pickle"))) == 1


@given(
    n_steps=st.integers(1, 5),
    delta_time=st.floats(min_value=0.01, max_value=5.0, **float_kw),  # type: ignore
)
def test_run_basic(n_steps, delta_time):
    time = n_steps * delta_time
    expected_n_steps = np.ceil(time / delta_time).astype(int)
    # Rounding errors can lead to the actual number of steps exceeding the
    # expected number of steps.
    assert expected_n_steps in (n_steps, n_steps + 1)

    def step_spy(*args, **kwargs):
        # Function attribute! Magic!
        step_spy.calls += 1

    step_spy.calls = 0

    with pytest.MonkeyPatch().context() as mp:
        # Patching the class, not the instance, because methods are read-only.
        mp.setattr(api.Experiment, "step", step_spy)
        experiment, _ = setup_experiment_with_floe()
        experiment.run(time=time, delta_time=delta_time, dump_final=False)
        assert step_spy.calls == expected_n_steps


def test_run_with_pbar(monkeypatch):
    experiment, _ = setup_experiment_with_floe()

    pbar = DummyPbar()
    experiment.run(time=2.0, delta_time=1.0, pbar=pbar, dump_final=False)
    assert pbar.updates == 2
    assert pbar.closed


@given(args=run_time_chunks_composite())
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@pytest.mark.parametrize("dump_final", (True, False))
def test_run_with_chunk_size(
    args: tuple[int, float, int], tmp_path: pathlib.Path, dump_final: bool
):
    n_steps, delta_time, chunk_size = args
    time = n_steps * delta_time
    # extra division to account for float errors
    actual_n_steps = np.ceil(time / delta_time).astype(int)
    if chunk_size == 1:
        expected_chunks = actual_n_steps
    else:
        expected_chunks = actual_n_steps // chunk_size
        # 1 removed from n_steps, because arithemtic done on iterator index,
        # starting at 0 and ending at n_steps - 1
        if dump_final and (((actual_n_steps - 1) % chunk_size) != (chunk_size - 1)):
            expected_chunks += 1

    # Give unique names depending on given + parametrize, as tmp_path has
    # function scope and is not reinitialised for different @given cases.
    prefix = f"test_{hash(args + (dump_final,)):x}"

    with pytest.MonkeyPatch().context() as mp:
        # Patching the class, not the instance, because methods are read-only.
        mp.setattr(Domain, "breakup", mock_breakup)
        experiment, _ = setup_experiment_with_floe()
        experiment.run(
            time=time,
            delta_time=delta_time,
            chunk_size=chunk_size,
            path=tmp_path,
            dump_final=dump_final,
            dump_prefix=prefix,
        )
    saved_chunks = len(list(tmp_path.glob(f"{prefix}*pickle")))
    assert saved_chunks == expected_chunks


@pytest.mark.parametrize("verbose", (None, 1, 2))
def test_verbose_run(
    verbose: int | None,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO)
    experiment, _ = setup_experiment_with_floe()

    n_steps = 1
    delta_time = 1
    experiment.run(
        time=n_steps * delta_time,
        delta_time=delta_time,
        chunk_size=1,
        verbose=verbose,
        path=tmp_path,
        dump_final=True,
    )
    post_fracture_n_floes = len(experiment.get_final_state().subdomains)
    assert post_fracture_n_floes == 2

    if verbose is None:
        assert len(caplog.text) == 0
    else:
        if verbose == 1:
            assert len(caplog.messages) == 1
        assert "history dumped" in caplog.text

    if verbose == 2:
        assert len(caplog.messages) == 2
        assert f"N_f = {post_fracture_n_floes}" in caplog.text


@pytest.mark.parametrize("verbose", (None, 1, 2))
@pytest.mark.parametrize("chunk_size", (None, 2))
@pytest.mark.parametrize("dump_final", (False, True))
def test_verbose_run_with_pbar(
    verbose: int | None,
    chunk_size: int | None,
    dump_final: bool,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
):
    time = 3
    delta_time = 1
    pbar = DummyPbar()
    spy = mocker.spy(pbar, "write")
    with pytest.MonkeyPatch().context() as mp:
        # Patching the class, not the instance, because methods are read-only.
        mp.setattr(Domain, "breakup", mock_breakup)
        experiment, _ = setup_experiment_with_floe()
        experiment.run(
            time=time,
            delta_time=delta_time,
            chunk_size=chunk_size,
            verbose=verbose,
            pbar=pbar,
            path=tmp_path,
            dump_final=dump_final,
        )
    if verbose is None:
        spy.assert_not_called()
    else:
        if chunk_size is None:
            spy.assert_not_called()
        else:
            spy.assert_called_once()


@given(data=st.data())
def test_run_early_termination(data):
    n_steps = 5
    with pytest.MonkeyPatch().context() as mp:
        # Patching the class, not the instance, because methods are read-only.
        orig_should_terminate = Experiment._should_terminate

        def mock_should_terminate(self, *args):
            return orig_should_terminate(self, 0, *args[1:])

        mp.setattr(Experiment, "_should_terminate", mock_should_terminate)
        mp.setattr(Domain, "breakup", mock_breakup)

        experiment, _ = setup_experiment_with_floe()
        time = 1 / experiment.domain.spectrum.frequencies[0]
        delta_time = time / n_steps
        break_time = data.draw(st.floats(delta_time, max_value=2 * time, **float_kw))

        expected_time = np.ceil(time / delta_time).astype(int) * delta_time
        expected_time_with_break = (
            np.ceil(np.nextafter(break_time / delta_time, np.inf)).astype(int)
            * delta_time
        )

        experiment.run(
            time=time,
            delta_time=delta_time,
            break_time=break_time,
            dump_final=False,
        )
        if break_time < time:
            assert experiment.time == expected_time_with_break
        else:
            assert experiment.time == expected_time
