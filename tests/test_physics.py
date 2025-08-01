from __future__ import annotations

import abc
import pathlib
import typing

import numpy as np
import pytest

import swiift.lib.physics as ph

if typing.TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore

# Test configurations visually examined against solution from scipy.solve_bvp.
TARGET_DIR_MONO = pathlib.Path("tests/target/physics_monochromatic")
TARGET_DIR_POLY = pathlib.Path("tests/target/physics_polychromatic")

# Hard-coded for ease of use within parametrize decorators.
# Tests ensure these numbers match the dimensions of the targets
# (specifically, _TestPhysics::test_dimensions).
N_CASES_MONO = 49
N_N_FREQS = 8  # number of different spectral lengths (2 to 100)
N_TRIES = 5  # number of tries per spectral length
N_CASES_POLY = N_N_FREQS * N_TRIES
INTEGRATION_METHODS = "pseudo_an", "tanhsinh", "quad"


def _flatten_and_squeeze(array: np.ndarray, size: int):
    """Reshape an array by contracting middle dimensions.

    This function is intented to be flexible enough to transform
    different-shaped targets of mono- and polychromatic parameters to a
    standard shape that can be interpreted (and looped over) by the test
    functions.

    The reshaping outputs an array of dimension 3. However, its axes of length
    1 are removed before returning it. Therefore, the returned array is at most
    of dimension 3.

    The first dimension of the array is preserved. If the reshaping is expected
    to happen over that first dimension, expand the array by adding a dimension
    before passing it to this function (see example).

    Parameters
    ----------
    array : np.ndarray
        ND-array to reshape.
    size : int
        Size several dimensions will be reshaped to.

    Returns
    -------
    np.ndarray
        The reshaped array, squeezed to remove remaining axes of size 1.

    Examples
    --------
    Situation corresponding to the polychromatic displacement and curvature
    targets. The two middle dimensions are contracted into one, preserving the
    first and last original dimensions.

    >>> arr1 = np.empty((2, 8, 5, 20))
    >>> arr1.shape
    (2, 8, 5, 20)
    >>> _flatten_and_squeeze(arr1, 40).shape
    (2, 40, 20)

    Situation corresponding to the polychromatic energy target. The two last
    dimensions are contracted into one. The initial reshaped array has shape
    (4, 40, 1). That last axis is squeezed out.

    >>> arr2 = np.empty((4, 8, 5))
    >>> arr2.shape
    (4, 8, 5)
    >>> _flatten_and_squeeze(arr2, 40).shape
    (4, 40)

    Situation corresponding to the monochromatic floe_params input. No
    contraction is necessary, but the function handles that case for
    genericity; provided the user takes care to add an extra dimension to the
    array before passing it. The argument received by the function therefore
    has shape (1, 49, 2). After the initial reshaping, the array has shape
    (1, 49, 2, 1). The first and last axes are squeezed out, returing exactly
    the initial array.

    >>> arr3 = np.empty((49, 2))
    >>> arr3.shape
    (49, 2)
    >>> _flatten_and_squeeze(np.expand_dims(arr3, 0), 49).shape
    (49, 2)

    """
    return np.squeeze(np.reshape(array, (array.shape[0], size, -1)))


def _expand(array):
    return np.expand_dims(array, 0)


class _TestPhysics(abc.ABC):
    """Base class exposing fixtures and logic for physics-related test.

    By physics, we mean the computing of vertical displacement and the
    associated curvature along a floe, and the computing of the energy of the
    deformed floe, as defined by the Handler classes under swiift.lib.physics.
    This class tests the stability of the `compute` method of these handlers,
    by comparing outputs parametrised with known inputs, to expected outputs
    (targets) verified visually.

    The inputs are two positive real numbers and two complex numbers
    (monochromatic cases) or two positive real numbers and two arrays of
    complex numbers of same size (polychromatic cases). There are thus six
    independent real numbers for the monochromatic cases. These are generated
    using Latin hypercube sampling. These samples are in turn sampled and
    combined to generate inputs for the polychromatic cases.

    Attributes
    ----------
    target_dir : pathlib.Path
        Path of the directory containing the inputs and targets.
    n_cases : int
        The number of expected test cases (combination of input and target).

    """

    target_dir: pathlib.Path
    n_cases: int

    def pytest_generate_tests(self, metafunc):
        # Pytest magic. Equivalent to adding pytest.mark.parametrize on j, but
        # allows for using a class/instance attribute as a parameter, which
        # would not be possible with a simple decorator.
        if "j" in metafunc.fixturenames:
            metafunc.parametrize("j", range(self.n_cases))

    def _flatten_and_squeeze(self, array: np.ndarray) -> np.ndarray:
        """Wrapper over the module-level function, setting the size parameters.

        Parameters
        ----------
        array : np.ndarray
            Array to be reshaped.

        Returns
        -------
        np.ndarray

        """
        return _flatten_and_squeeze(array, self.n_cases)

    def _load(self, filename: str) -> np.ndarray:
        """Helper function loading arrays with respect to class attribute.

        The argument is expected to be an NPY file.

        Parameters
        ----------
        filename : str
            File under `cls.target_dir`.

        Returns
        -------
        np.ndarray

        """
        return np.load(self.target_dir.joinpath(filename))

    @pytest.fixture(scope="class")
    def x_axes(self) -> np.ndarray:
        """X-axes over which to compute displacement and curvature.

        Returns
        -------
        np.ndarray

        """
        return self._flatten_and_squeeze(_expand(self._load("x.npy")))

    @pytest.fixture(scope="class")
    def floe_params_all(self) -> np.ndarray:
        """Floe parameters to instantiate physical handlers.

        Returns
        -------
        np.ndarray

        """
        return self._flatten_and_squeeze(_expand(self._load("floe_params.npy")))

    @abc.abstractmethod
    def wave_params_all(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Wave parameters to instantiate physical handlers.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]

        """
        ...

    @pytest.fixture(scope="class")
    def displacements(self) -> np.ndarray:
        """Vertical displacement targets.

        Returns
        -------
        np.ndarray

        """
        return self._flatten_and_squeeze(self._load("displacements.npy"))

    @pytest.fixture(scope="class")
    def curvatures(self) -> np.ndarray:
        """Curvature targets.

        Returns
        -------
        np.ndarray

        """
        return self._flatten_and_squeeze(self._load("curvatures.npy"))

    @pytest.fixture(scope="class")
    def energies(self) -> np.ndarray:
        """Energy targets.

        Returns
        -------
        np.ndarray

        """
        return self._flatten_and_squeeze(self._load("energies.npy"))

    def test_dimensions(
        self,
        x_axes: np.ndarray,
        floe_params_all: np.ndarray,
        wave_params_all: list[tuple[np.ndarray, np.ndarray]],
        displacements: np.ndarray,
        curvatures: np.ndarray,
        energies: np.ndarray,
    ):
        """Check that the dimensions match the expected number of cases.

        Parameters
        ----------
        x_axes : np.ndarray
        floe_params_all : np.ndarray
        wave_params_all : list[tuple[np.ndarray, np.ndarray]]
        displacements : np.ndarray
        curvatures : np.ndarray
        energies : np.ndarray

        """
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
    @pytest.mark.benchmark(group=": ")
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
        """Compare local quantities (displacement, curvature) to targets.

        Parameters
        ----------
        request : pytest.FixtureRequest
        x_axes : np.ndarray
        floe_params_all : np.ndarray
        wave_params_all : list[tuple[np.ndarray, np.ndarray]]
        handler_type : type[ph.DisplacementHandler] | type[ph.CurvatureHandler]
            The type of handler to use.
        target_name : str
            The name of the fixture providing the target.
        an_sol : bool
            Whether to use the analytical solution formulation.
        j : int
            Index of the test case.
        benchmark : BenchmarkFixture

        """
        benchmark.group = (
            f"{str(self.target_dir).split()[-1]}_{target_name}:case_{j:02d}"
        )
        # The first dimension of the target has size 2.
        # The first entry (i := 0) corresponds to the analytical solution, the
        # second entry (i := 1) to the numerical solution.
        i = 0 if an_sol else 1
        # Pytest magic, get fixture by name as fixtures cannot be used directly
        # in parametrize.
        target = request.getfixturevalue(target_name)
        x = x_axes[j]
        floe_params = floe_params_all[j]
        wave_params = wave_params_all[j]
        handler = handler_type(floe_params, wave_params)
        computed = benchmark(handler.compute, x, an_sol=an_sol)
        assert np.allclose(computed, target[i, j])

    @pytest.mark.parametrize(
        "integration_method", (None, "pseudo_an", "tanhsinh", "quad")
    )
    @pytest.mark.parametrize("integration_method", (None, *INTEGRATION_METHODS))
    @pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
    def test_energy(
        self,
        floe_params_all: np.ndarray,
        wave_params_all: list[tuple[np.ndarray, np.ndarray]],
        energies: np.ndarray,
        integration_method: str | None,
        j: int,
        benchmark: BenchmarkFixture,
    ):
        """Compare energy to target.

        Parameters
        ----------
        floe_params_all : np.ndarray
        wave_params_all : list[tuple[np.ndarray, np.ndarray]]
        energies : np.ndarray
        integration_method : str | None
            Which integration method to use. If none, compute the analytical
            solution.
        j : int
            Index of the test case.
        benchmark : BenchmarkFixture

        Warns
        -----
        Four cases (j in {7, 15, 26, 29}) raise an IntegrationWarning when
        using the quad method. This is expected and non consequantial; the
        issue ("[...] The error may be underestimated.") happens when
        generating the test cases, and the accuracy (when compared to other
        methods) is correct. We thus filter the warnings when running the test
        to avoid clutter.

        """
        benchmark.group = f"{str(self.target_dir).split()[-1]}_energy:case_{j:02d}"
        floe_params = floe_params_all[j]
        wave_params = wave_params_all[j]
        handler = ph.EnergyHandler(floe_params, wave_params)
        if integration_method is None:
            an_sol = True
            i = 0
        else:
            an_sol = False
            i = 1 + INTEGRATION_METHODS.index(integration_method)

        computed = benchmark(
            handler.compute, an_sol=an_sol, integration_method=integration_method
        )
        assert np.allclose(computed, energies[i, j])


class TestPhysicsMono(_TestPhysics):
    """Class dealing with monochromatic physics."""

    target_dir = TARGET_DIR_MONO
    n_cases = N_CASES_MONO

    @pytest.fixture(scope="class")
    def wave_params_all(self) -> list[tuple[np.ndarray, np.ndarray]]:
        # Turn iterator into list as it will be used three times, by three
        # handlers, and to allow indexing.
        return list(
            zip(
                np.load(self.target_dir.joinpath("c_amplitudes.npy")),
                np.load(self.target_dir.joinpath("c_wavenumbers.npy")),
            )
        )


class TestPhysicsPoly(_TestPhysics):
    """Class dealing with polychromatic physics."""

    target_dir = TARGET_DIR_POLY
    n_cases = N_CASES_POLY

    @pytest.fixture(scope="class")
    def wave_params_all(self) -> list[tuple[np.ndarray, np.ndarray]]:
        wave_params = np.load(self.target_dir.joinpath("wave_params.npz"))
        # Check we do have the expected number of different numbers of frequencies.
        assert len(wave_params) == N_N_FREQS
        # Turn the dict-like Npz object into a list of tuples.
        flat_list = [(v[:, 0], v[:, 1]) for vals in wave_params.values() for v in vals]
        for i, nfreqs in enumerate(map(int, wave_params.keys())):
            # Sanity check: we do have the expected number of frequencies
            # (as provided by the keys of the Npz object).
            for j in range(N_TRIES):
                slice = flat_list[i * N_TRIES : (i + 1) * N_TRIES][j]
                assert len(slice) == 2
                assert len(slice[0]) == nfreqs  # complex amplitudes
                assert len(slice[1]) == nfreqs  # complex wavenumbers
        return flat_list
