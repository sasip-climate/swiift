#!/usr/bin/env python3

import numpy as np
import warnings


def sanitize_input(arg, name, allow_neg=False):
    real_part = np.real(arg)
    absolute_real_part = np.abs(real_part)
    complex_modulus = np.abs(arg)
    if absolute_real_part != complex_modulus:
        warnings.warn(
            f"Parameter {name} is expected to be real and "
            "the provided imaginary part will be discarded",
            np.ComplexWarning,
            stacklevel=2,
        )
    if allow_neg:
        return real_part
    return absolute_real_part
