#!/usr/bin/env python3

import numpy as np


def _unit_wavefield(x, c_wavenumbers):
    return np.exp((1j * c_wavenumbers[:, None]) * x)


def _wavefield(x, c_amps, c_wavenumbers):
    return np.imag(c_amps @ _unit_wavefield(x, c_wavenumbers))
