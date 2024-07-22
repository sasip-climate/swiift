#!/usr/bin/env python3

import numpy as np


def _unit_wavefield(x: np.ndarray, c_wavenumbers: np.ndarray) -> np.ndarray:
    return np.exp((1j * c_wavenumbers[:, None]) * x)


def _wavefield(
    x: np.ndarray, c_amps: np.ndarray, c_wavenumbers: np.ndarray
) -> np.ndarray:
    return np.imag(c_amps @ _unit_wavefield(x, c_wavenumbers))
