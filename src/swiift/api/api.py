from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence
import functools
import logging
import operator
import pathlib
import pickle
import typing
from typing import Any

import attrs
import numpy as np

from .. import __about__
from ..lib import att
from ..model import frac_handlers as fh, model as md

# TODO: make into an attrs class for more flexibility (repr of subdomains)
Step = namedtuple("Step", ["subdomains", "growth_params"])

logger = logging.getLogger(__name__)


if typing.TYPE_CHECKING:

    class _ProgressBarProtocol(typing.Protocol):
        def update(self, n): ...

        def close(self): ...

    class _VerboseProgressBarProtocol(_ProgressBarProtocol):
        @classmethod
        def write(cls, *args): ...


def _create_path(path: str | pathlib.Path) -> pathlib.Path:
    _path = pathlib.Path(path)
    if not _path.exists():
        _path.mkdir(parents=True)
    return _path


def _load_pickle(fname: str | pathlib.Path) -> Experiment:
    with open(fname, "rb") as file:
        logger.info(f"Reading {fname}...")
        instance = pickle.load(file)
        if not isinstance(instance, Experiment):
            raise TypeError("The pickled object is not an instance of `Experiment`.")
    return instance


def _glob(
    pattern: str, dir_path: pathlib.Path, recursive: bool, **kwargs
) -> list[Experiment]:
    attribute = "rglob" if recursive else "glob"
    iterator = getattr(dir_path, attribute)
    return [_load_pickle(fname) for fname in iterator(pattern, **kwargs)]


def _dct_keys_to_array(dct, dtype=np.float64) -> np.ndarray:
    return np.fromiter(dct, dtype, count=len(dct))


def _assemble_experiments(experiments: list[Experiment]) -> Experiment:
    latest_experiment = max(experiments, key=lambda _exp: _exp.time)
    common_history = functools.reduce(
        operator.or_, (_exp.history for _exp in experiments)
    )
    sorted_keys = np.sort(_dct_keys_to_array(common_history))
    common_history = {k: common_history[k] for k in sorted_keys}
    return Experiment(
        latest_experiment.time,
        latest_experiment.domain,
        common_history,
        latest_experiment.fracture_handler,
    )


def _str_to_path(dir_path: str | pathlib.Path | None) -> pathlib.Path:
    match dir_path:
        case None:
            _dir_path = pathlib.Path.cwd()
        case _:
            _dir_path = pathlib.Path(dir_path)
    return _dir_path


def load_pickles(
    pattern: str,
    dir_path: str | pathlib.Path | None = None,
    recursive: bool = False,
    **kwargs,
) -> Experiment:
    """Load pickle objects and assemble them into a single `Experiment`.

    This function relies on `pathlib.Path`'s `glob` and `rglob` methods.
    Files found matching the pattern are assembled into a single `Experiment`
    object: thas is, histories are concatenated. Duplicated keys (timestep
    entries) are thus lost. This function is therefore intended to be used
    on files which the user knows have no overlap between their time axes.

    Parameters
    ----------
    pattern : str
        A pattern to glob upon.
    root : str | pathlib.Path | None
        The directory in which files will be looked for. If `None`, search from
        the current working directory.
    recursive : bool
        Whether to search for the pattern recursively.
    **kwargs
        Arguments passed to `pathlib.Path.[r]glob`.

    Returns
    -------
    Experiment

    Raises
    ------
    FileNotFoundError
        If no file matches the pattern.
    ValueError
        If a found file does not correspond to an instance of `Experiment`.

    """
    _dir_path = _str_to_path(dir_path)

    experiments = _glob(pattern, _dir_path, recursive, **kwargs)
    if len(experiments) == 0:
        raise FileNotFoundError(f"No file matching {pattern} was found.")
    return _assemble_experiments(experiments)


def load_pickle(
    fname: str | pathlib.Path,
) -> Experiment:
    """Read and return an `Experiment` object stored in a pickle file.

    Parameters
    ----------
    fname : str | pathlib.Path
        A file name or path object.

    Returns
    -------
    Experiment

    Raises
    ------
    FileNotFoundError
        If files matching `fname` cannot be found.

    """
    return _load_pickle(fname)


@attrs.define
class Experiment:
    time: float
    domain: md.Domain
    history: dict[float, Step] = attrs.field(factory=dict, repr=False)
    fracture_handler: fh._FractureHandler = attrs.field(factory=fh.BinaryFracture)

    @classmethod
    def from_discrete(
        cls,
        gravity: float,
        spectrum: md.DiscreteSpectrum,
        ocean: md.Ocean,
        growth_params: tuple | None = None,
        fracture_handler: fh._FractureHandler | None = None,
        attenuation_spec: att.Attenuation | None = None,
    ):
        if attenuation_spec is None:
            attenuation_spec = att.AttenuationParameterisation(1)
        domain = md.Domain.from_discrete(
            gravity, spectrum, ocean, attenuation_spec, growth_params
        )

        if fracture_handler is None:
            return cls(0, domain)
        return cls(0, domain, fracture_handler=fracture_handler)

    @property
    def timesteps(self) -> np.ndarray:
        """The experiment timesteps in s.

        These can be used to index `self.history`.

        Returns
        -------
        1D array
            The existing timesteps.

        """
        return np.array(list(self.history.keys()))

    def add_floes(self, floes: md.Floe | Sequence[md.Floe]):
        self.domain.add_floes(floes)
        self._save_step()

    def _find_fracture_indices(self) -> np.ndarray[tuple[Any, ...], np.dtype[np.int_]]:
        """Find the indices of states immediately before fracture.

        Returns
        -------
        1D array of int
            The indices of the current timesteps corresponding to the states
            that broke on the next iteration.

        """
        _t = [len(step.subdomains) for step in self.history.values()]
        return np.nonzero(np.ediff1d(_t))[0]

    def get_pre_fracture_times(self) -> np.ndarray:
        """Return the times corresponding to states immediately after fracture.

        These can be used to index `self.history`.

        Returns
        -------
        1D array
            Output times.

        """
        return self.timesteps[self._find_fracture_indices()]

    def get_post_fracture_times(self) -> np.ndarray:
        """Return the times corresponding to states immediately after fracture.

        These can be used to index `self.history`.
        Note: in these states, the waves have been advected, compared to the
        corresponding pre-fracture states.

        Returns
        -------
        1D array
            Output times.

        """
        return self.timesteps[self._find_fracture_indices() + 1]

    def get_final_state(self) -> Step:
        """Return the final state of the experiment.

        Returns
        -------
        Step
            The `Step` corresponding to the last timestep.

        """
        return self.history[next(reversed(self.history))]

    def _save_step(self):
        self.history[self.time] = Step(
            tuple(wuf.make_copy() for wuf in self.domain.subdomains),
            (
                (self.domain.growth_params[0].copy(), self.domain.growth_params[1])
                if self.domain.growth_params is not None
                else None
            ),
        )

    def step(
        self,
        delta_time: float,
        an_sol: bool | None = None,
        num_params: dict | None = None,
    ):
        """Move the experiment forward in time.

        On step is a succession of events. First, the current floes are scanned
        for fractures. The domain is eventually updated with the newly formed
        fragments replacing the fractured floes. Then, the actual time
        progression happens, by updating the wave phases at the edge of every
        individual floe. Finally, this new state is saved to the history, at
        the index corresponding to the updated time.

        Parameters
        ----------
        delta_time : float
            The time increment in second.
        an_sol : bool, optional
            Whether to force the use of a numerical or analytical solution for
            the deflection of the floes.
        num_params : dict, optional
            Optional parameters to pass to the numerical solver, if applicable.

        """
        self.domain.breakup(self.fracture_handler, an_sol, num_params)
        self.domain.iterate(delta_time)
        self.time += delta_time
        self._save_step()

    def get_states(self, times: np.ndarray | float) -> dict[float, Step]:
        """Return a subset of the history matching the given times.

        Parameters
        ----------
        times : 1D array_like, float
            Time, or sequence of times.

        Returns
        -------
        dict[float, Step]
            A dictionary containing the `Step`s closest to the input `times`.

        """
        times = np.ravel(times)  # ensure we have exactly a 1D array
        timestep_keys = _dct_keys_to_array(self.history)
        indexes = (np.abs(times - timestep_keys[:, None])).argmin(axis=0)
        return {k: self.history[k] for k in timestep_keys[indexes]}

    def get_states_strict(self, times: np.ndarray | float) -> dict[float, Step]:
        """Return a subset of the history matching the given times.

        Parameters
        ----------
        times : np.ndarray | float
            Time, or sequence of times.

        Returns
        -------
        dict[float, Step]
            A dictionary containing the `Step`s matching exactly the input.

        """
        times, sort_idx = np.unique(np.ravel(times), return_index=True)
        timestep_keys = _dct_keys_to_array(self.history)
        # We `unsort' the output of np.unique with the index `sort_idx`,
        # so that values are returned in the order they were passed.
        filtered_times = times[
            np.isin(times[np.argsort(sort_idx)], timestep_keys, assume_unique=True)
        ]
        return {_time: self.history[_time] for _time in filtered_times}

    def _time_interval_str(self):
        first_time = next(iter(self.history))
        return f"{first_time:.3f}--{self.time:.3f}"

    def _generate_name(self, prefix: str | None) -> str:
        if prefix is None:
            prefix = f"{id(self):x}"
        return prefix + f"_v{__about__.__version__}_" + self._time_interval_str()

    def _dump(self, prefix: str | None, dir_path: pathlib.Path):
        fname = f"{self._generate_name(prefix)}.pickle"
        dir_path = _create_path(dir_path)
        full_path = dir_path.joinpath(fname)
        with open(full_path, "bw") as file:
            pickle.dump(self, file)

    def _clean_history(self):
        current_state = self.get_final_state()
        self.history.clear()
        self.history[self.time] = current_state

    def dump_history(
        self,
        prefix: str | None = None,
        dir_path: str | pathlib.Path | None = None,
    ):
        """Write the results to disk and clear the history.

        The whole object is pickled, before emptying the current history from
        memory. The filename is constructed with the `prefix` passed as
        argument, the package version number, and the time interval covered by
        the history.

        Parameters
        ----------
        prefix : str | None
            Prefix for the file name. If none is provided, defaults to the `id`
            of the `Experiment` object.

        """
        _dir_path = _str_to_path(dir_path)
        self._dump(prefix, _dir_path)
        self._clean_history()

    def _should_terminate(
        self,
        initial_number_of_fragments: int,
        number_of_fragments: int,
        time_since_fracture: float,
        break_time: float | None,
    ):
        return (
            break_time is not None
            and number_of_fragments > initial_number_of_fragments
            and time_since_fracture > break_time
        )

    @typing.overload
    def run(
        self,
        time: float,
        delta_time: float,
        break_time: float | None = ...,
        chunk_size: int | None = ...,
        verbose: None = ...,
        pbar: _ProgressBarProtocol | None = ...,
        path: str | pathlib.Path | None = ...,
        dump_final: bool = ...,
        dump_prefix: str | None = ...,
    ): ...

    @typing.overload
    def run(
        self,
        time: float,
        delta_time: float,
        break_time: float | None = ...,
        chunk_size: int | None = ...,
        verbose: int = ...,
        pbar: _VerboseProgressBarProtocol | None = ...,
        path: str | pathlib.Path | None = ...,
        dump_final: bool = ...,
        dump_prefix: str | None = ...,
    ): ...

    def run(
        self,
        time: float,
        delta_time: float,
        break_time: float | None = None,
        chunk_size: int | None = None,
        verbose: int | None = None,
        pbar: _ProgressBarProtocol | None = None,
        path: str | pathlib.Path | None = None,
        dump_final: bool = True,
        dump_prefix: str | None = None,
    ):
        """Run the experiment for a specified duration.

        The experiment is run from its current time for a duration
        corresponding to `time`, with states regularly spaced with step
        `delta_time`. If `time` is not an integer multiple of `delta_time`, the
        number of steps will be rounded up. The experiment can optionally be
        stopped before `time`, if no fracture happens for `break_time`, and at
        least one fracture has occured.

        The current object can be saved at regularly spaced step intervals, as
        specified by `chunk_size`.

        Optional messages can be printed to stdout, with a verbosity level
        controlled by `verbose`.

        A progress bar can be passed as an optional parameter to monitor the
        experiment. The implementation expect an objects that behaves as a
        `tqdm` bar; in particular, it needs to expose `update` and close
        `method`. If used conjonctly with `verbose`, it also needs to expose a
        `write` method.

        Parameters
        ----------
        time : float
            Duration to run the experiment for, in seconds.
        delta_time : float
            Time step between iterations, in seconds.
        break_time : float | None
            Time before stopping the experiment if no fracture occurs, in seconds.
        chunk_size : int | None
            Number of steps before writing the results to a file.
        verbose : int | None
            Verbosity level. If 1, outputs for disk writes. If 2, additional
            outputs for fractures.
        pbar : progress bar | None
            Progress bar monitoring the experiment.
        path : str | pathlib.Path | None
            Directory where files will be saved. If none is provided, files
            will be saved in the current directory.
        dump_final : bool
            Whether the results should be saved to disk at the end of the run
            by calling `dump_history`, thus clearing the history from memory.
        dump_prefix : str | None
            Prefix for the file names used in the dumps. If none is provided,
            defaults to the `id` of the `Experiment` object.

        """

        def pbar_print(msg: str, pbar: _VerboseProgressBarProtocol | None):
            if pbar is not None:
                pbar.write(msg)
            else:
                logger.info(msg)

        @typing.overload
        def dump_and_print(
            dump_prefix: str | None,
            path: str | pathlib.Path | None,
            verbose: None,
            pbar: _ProgressBarProtocol | None,
        ): ...

        @typing.overload
        def dump_and_print(
            dump_prefix: str | None,
            path: str | pathlib.Path | None,
            verbose: int,
            pbar: _VerboseProgressBarProtocol | None,
        ): ...

        def dump_and_print(
            dump_prefix,
            path,
            verbose,
            pbar,
        ):
            self.dump_history(dump_prefix, path)
            if verbose is not None and verbose >= 1:
                msg = f"t = {self.time:.3f} s; history dumped"
                pbar_print(msg, pbar)

        initial_number_of_fragments = len(self.domain.subdomains)
        number_of_fragments = initial_number_of_fragments
        number_of_steps = np.ceil(time / delta_time).astype(int)
        time_since_fracture = 0.0
        if chunk_size is not None:
            modulo_target = chunk_size - 1

        for i in range(number_of_steps):
            self.step(delta_time)
            new_nof = len(self.domain.subdomains)
            if new_nof > number_of_fragments:
                time_since_fracture = 0
                number_of_fragments = new_nof
                if verbose is not None and verbose >= 2:
                    msg = f"t = {self.time:.3f} s; N_f = {number_of_fragments}"
                    pbar_print(msg, pbar)
            else:
                time_since_fracture += delta_time

            if chunk_size is not None:
                if i % chunk_size == modulo_target:
                    dump_and_print(dump_prefix, path, verbose, pbar)

            if self._should_terminate(
                initial_number_of_fragments,
                number_of_fragments,
                time_since_fracture,
                break_time,
            ):
                msg = f"No fracture in {break_time:.3f} s, stopping"
                pbar_print(msg, pbar)
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        # If single item in history, either we just dump, either we did not
        # step, and there is no need to dump again.
        if dump_final and len(self.history) > 1:
            # No `pbar` passed as it should have been closed
            dump_and_print(dump_prefix, path, verbose, None)
