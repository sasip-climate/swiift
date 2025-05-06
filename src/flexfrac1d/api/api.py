from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence
import pickle

import attrs
import numpy as np

from .. import __about__
from ..lib import att
from ..model import frac_handlers as fh, model as md

# TODO: make into an attrs class for more flexibility (repr of subdomains)
Step = namedtuple("Step", ["subdomains", "growth_params"])


def read_pickle(fname) -> Experiment:
    with open(fname, "rb") as file:
        instance = pickle.load(file)
        if not isinstance(instance, Experiment):
            raise TypeError("The pickled object is not an instance of `Experiment`.")
    return instance


@attrs.define
class Experiment:
    time: float
    domain: md.Domain
    history: dict[float, Step] = attrs.field(init=False, factory=dict, repr=False)
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
        return cls(0, domain, fracture_handler)

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

    def _find_fracture_indices(self) -> np.ndarray[int]:
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
        times = np.ravel(times)
        timestep_keys = np.array(list(self.history.keys()))
        indexes = (np.abs(times - timestep_keys[:, None])).argmin(axis=0)
        return {k: self.history[k] for k in timestep_keys[indexes]}

    def _generate_name(self):
        first_time = next(iter(self.history))
        return f"{id(self)}_v{__about__.__version__}_{first_time:.3f}-{self.time:.3f}"

    def _dump(self):
        fname = f"{self._generate_name()}.pickle"
        with open(fname, "bw") as file:
            pickle.dump(self, file)

    def _clean_history(self):
        current_state = self.get_final_state()
        self.history.clear()
        self.history[self.time] = current_state

    def dump_history(self):
        self._dump()
        self._clean_history()
