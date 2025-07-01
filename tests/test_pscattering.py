import typing

import numpy as np
import pytest

import swiift.lib.phase_shift as ps

random_handlers = (
    ps.UniformScatteringHandler,
    ps.PerturbationScatteringHandler,
)

edge_amplitudes = (
    np.array(1.5 + 1.8j),
    np.array([0.11833799 + 0.63103069j, 0.78585931 + 0.84839417j]),
)
c_wavenumbers = (
    np.array(0.78032224 + 0.85889267j),
    np.array([0.10716051 + 0.7798643j, 0.54648262 + 0.94570875j]),
)
xfs = (
    np.array([11.1]),
    np.array([4.7, 8.3]),
    np.array([2.2, 6.4, 9.3]),
)


@pytest.mark.parametrize(
    "edge_amplitudes, c_wavenumbers", zip(edge_amplitudes, c_wavenumbers)
)
@pytest.mark.parametrize("xf", xfs)
def test_continuity(
    edge_amplitudes: np.ndarray, c_wavenumbers: np.ndarray, xf: np.ndarray
):
    # Verify that when no scattering is used, the right edge of the left floe
    # is in the same state as the left edge of the right floe. That is, the
    # complex amplitude advected from the left edge of the left floe is equal to the
    # amplitude at the left edge of the right floe.
    post_amplitudes = ps.ContinuousScatteringHandler.compute_edge_amplitudes(
        edge_amplitudes, c_wavenumbers, xf
    )
    for xl, xr, pl, pr in zip(
        xf[:-1], xf[1:], post_amplitudes[:-1], post_amplitudes[1:]
    ):
        assert np.allclose(pl * np.exp(1j * c_wavenumbers * (xr - xl)) - pr, 0)


@pytest.mark.parametrize("handler_type", random_handlers)
@pytest.mark.parametrize(
    "edge_amplitudes, c_wavenumbers", zip(edge_amplitudes, c_wavenumbers)
)
@pytest.mark.parametrize("xf", xfs)
def test_absolute_value_continuity(
    handler_type: typing.Type[ps._RandomScatteringHandler],
    edge_amplitudes: np.ndarray,
    c_wavenumbers: np.ndarray,
    xf: np.ndarray,
):
    # Verify that when scattering is used, the amplitude at the right edge of
    # the left floe is equal, in amplitude, and different, in phase, from the
    # amplitude at the left edge of the right floe.
    seed = 3103
    random_handler = handler_type.from_seed(seed)
    pa_continuous = ps.ContinuousScatteringHandler.compute_edge_amplitudes(
        edge_amplitudes, c_wavenumbers, xf
    )
    pa_random = random_handler.compute_edge_amplitudes(
        edge_amplitudes, c_wavenumbers, xf
    )
    for pac, par in zip(pa_continuous[1:], pa_random[1:]):
        assert not np.allclose(pac - par, 0)
        assert np.allclose(np.abs(pac) - np.abs(par), 0)


@pytest.mark.parametrize("random_handler", random_handlers)
@pytest.mark.parametrize(
    "edge_amplitudes, c_wavenumbers", zip(edge_amplitudes, c_wavenumbers)
)
@pytest.mark.parametrize("xf", xfs)
def test_repeatability(
    random_handler: typing.Type[ps._RandomScatteringHandler],
    edge_amplitudes,
    c_wavenumbers,
    xf,
):
    # Verify that two random scattering handlers return the same result when
    # computing the amplitudes from a generator seeded with the same number.
    seed = 83

    handler1 = random_handler.from_seed(seed)
    test1 = handler1.compute_edge_amplitudes(edge_amplitudes, c_wavenumbers, xf)

    handler2 = random_handler.from_seed(seed)
    test2 = handler2.compute_edge_amplitudes(edge_amplitudes, c_wavenumbers, xf)

    assert np.allclose(test1 - test2, 0)
