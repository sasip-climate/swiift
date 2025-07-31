from __future__ import annotations

import abc
import pathlib
import typing

import numpy as np
import pytest

import swiift.lib.physics as ph

if typing.TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

# Test configurations visually examined against solution from scipy.solve_bvp
TARGET_DIR_MONO = pathlib.Path("tests/target/physics_monochromatic")
TARGET_DIR_POLY = pathlib.Path("tests/target/physics_polychromatic")

T = typing.TypeVar("T", ph.DisplacementHandler, ph.CurvatureHandler, ph.EnergyHandler)

# Hard-coded for ease of use within parametrize decorators. Tests ensure these
# numbers match the dimensions of the targets.
N_CASES_MONO = 49
N_N_FREQS = 8  # number of different spectral lengths
N_TRIES = 5  # number of tries per spectral length
N_CASES_POLY = N_N_FREQS * N_TRIES


def _flatten_and_squeeze(array, size):
    return np.squeeze(np.reshape(array, (array.shape[0], size, -1)))


def _expand(array):
    return np.expand_dims(array, 0)


class _TestPhysics(abc.ABC):
    target_dir: pathlib.Path
    n_cases: int

    def pytest_generate_tests(self, metafunc):
        if "j" in metafunc.fixturenames:
            metafunc.parametrize("j", range(self.n_cases))

    def _flatten_and_squeeze(self, array):
        return _flatten_and_squeeze(array, self.n_cases)

    def _load(self, filename: str):
        return np.load(self.target_dir.joinpath(filename))

    @pytest.fixture(scope="class")
    def x_axes(self) -> np.ndarray:
        return self._flatten_and_squeeze(_expand(self._load("x.npy")))

    @pytest.fixture(scope="class")
    def floe_params_all(self) -> np.ndarray:
        return self._flatten_and_squeeze(_expand(self._load("floe_params.npy")))

    @abc.abstractmethod
    def wave_params_all(self): ...

    @pytest.fixture(scope="class")
    def displacements(self) -> np.ndarray:
        return self._flatten_and_squeeze(self._load("displacements.npy"))

    @pytest.fixture(scope="class")
    def curvatures(self) -> np.ndarray:
        return self._flatten_and_squeeze(self._load("curvatures.npy"))

    @pytest.fixture(scope="class")
    def energies(self) -> np.ndarray:
        return self._flatten_and_squeeze(self._load("energies.npy"))

    def test_dimensions(
        self,
        x_axes,
        floe_params_all,
        wave_params_all,
        displacements,
        curvatures,
        energies,
    ):
        for arr in (x_axes, floe_params_all, wave_params_all):
            assert len(arr) == self.n_cases
        for arr in (
            displacements,
            curvatures,
            energies,
        ):
            assert arr.shape[1] == self.n_cases
        for arr in (
            displacements,
            curvatures,
        ):
            assert x_axes.shape[-1] == arr.shape[-1]

    @pytest.mark.parametrize("an_sol", (True, False))
    @pytest.mark.parametrize(
        "handler_type, target_name",
        (
            (ph.DisplacementHandler, "displacements"),
            (ph.CurvatureHandler, "curvatures"),
        ),
    )
    @pytest.mark.benchmark(group="Local: ")
    def test_local(
        self,
        request: pytest.FixtureRequest,
        x_axes: np.ndarray,
        floe_params_all: np.ndarray,
        wave_params_all: list[tuple[np.ndarray, np.ndarray]],
        handler_type: type[ph.DisplacementHandler] | type[ph.CurvatureHandler],
        target_name: str,
        an_sol: bool,
        j: int,
        benchmark: BenchmarkFixture,
    ):
        i = 0 if an_sol else 1
        target = request.getfixturevalue(target_name)
        x = x_axes[j]
        floe_params = floe_params_all[j]
        wave_params = wave_params_all[j]
        handler = handler_type(floe_params, wave_params)
        computed = benchmark(handler.compute, x, an_sol=an_sol)
        assert np.allclose(computed, target[i, j])

    @pytest.mark.parametrize(
        "integration_method",
        (
            None,
            "pseudo_an",
            "tanhsinh",
            "quad",
        ),
    )
    @pytest.mark.benchmark(group="Energy: ")
    def test_energy(
        self,
        floe_params_all: np.ndarray,
        wave_params_all: list[tuple[np.ndarray, np.ndarray]],
        energies: np.ndarray,
        integration_method: str | None,
        j: int,
        benchmark,
    ):
        benchmark.group += f"case: {j:02d}"
        floe_params = floe_params_all[j]
        wave_params = wave_params_all[j]
        handler = ph.EnergyHandler(floe_params, wave_params)
        if integration_method is None:
            an_sol = True
            i = 0
        else:
            an_sol = False
            match integration_method:
                case "pseudo_an":
                    i = 1
                case "tanhsinh":
                    i = 2
                case "quad":
                    i = 3
                case _:
                    raise ValueError(
                        f"Invalid integration method: {integration_method}."
                    )

        computed = benchmark(
            handler.compute, an_sol=an_sol, integration_method=integration_method
        )
        assert np.allclose(computed, energies[i, j])


class TestPhysicsMono(_TestPhysics):
    target_dir = TARGET_DIR_MONO
    n_cases = N_CASES_MONO

    @pytest.fixture(scope="class")
    def wave_params_all(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return list(
            zip(
                np.load(self.target_dir.joinpath("c_amplitudes.npy")),
                np.load(self.target_dir.joinpath("c_wavenumbers.npy")),
            )
        )


class TestPhysicsPoly(_TestPhysics):
    target_dir = TARGET_DIR_POLY
    n_cases = N_CASES_POLY

    @pytest.fixture(scope="class")
    def wave_params_all(self) -> list[tuple[np.ndarray, np.ndarray]]:
        # Expects
        wave_params = np.load(self.target_dir.joinpath("wave_params.npz"))
        assert len(wave_params) == N_N_FREQS
        flat_list = [(v[:, 0], v[:, 1]) for vals in wave_params.values() for v in vals]
        for i, nfreqs in enumerate(map(int, wave_params.keys())):
            for j in range(N_TRIES):
                assert len(flat_list[i * N_TRIES : (i + 1) * N_TRIES][j][0]) == nfreqs
                assert len(flat_list[i * N_TRIES : (i + 1) * N_TRIES][j][1]) == nfreqs
        return flat_list
