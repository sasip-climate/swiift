import typing

import numpy as np

from swiift.model.model import FloatingIce, Ice, Ocean, WavesUnderFloe, WavesUnderIce

wave_params = (
    (
        np.array([0.00950574 + 0.13669057j]),
        np.array([0.02660581 + 0.02685434j]),
    ),  # monochromatic
    (
        (
            np.array([0.00950574 + 0.13669057j, 0.02660581 + 0.0265434j]),
            np.array([0.03552382 + 0.05215654j, 0.06214718 + 0.1250975j]),
        )  # polychromatic
    ),
)
growth_params_bool = (None, True)


def make_growth_params(
    growth_params_bool: typing.Literal[True] | None,
    wave_params: tuple[np.ndarray, np.ndarray],
) -> tuple | None:
    if growth_params_bool is not None:
        one_and_maybe_two = np.linspace(1, 2, len(wave_params[0]))
        # Set arbitrary growth kernel with correct shape. Setting the mean to a
        # negative number ensures a numerical solution is used.
        return (-3 * one_and_maybe_two[:, None], 20)
    return growth_params_bool


def setup_wuf(wave_params: tuple[np.ndarray, np.ndarray]) -> WavesUnderFloe:
    gravity = 9.8
    left_edge = -13.2
    length = 98.3

    amplitudes, c_wavenumbers = wave_params
    wavenumbers, attenuations = (func(c_wavenumbers) for func in (np.real, np.imag))
    wui = WavesUnderIce(
        FloatingIce.from_ice_ocean(Ice(), Ocean(), gravity),
        wavenumbers,
        attenuations,
    )
    return WavesUnderFloe(
        left_edge=left_edge, length=length, wui=wui, edge_amplitudes=amplitudes
    )
