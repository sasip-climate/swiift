"""Attenuation parameterisations."""

import enum
import typing

import attrs
import numpy as np


def no_attenuation():
    """No attenuation.

    Waves propagate indifinitely, as if the ice cover is perfectly elastic and
    the fluid perfectly inviscid.

    Returns
    -------
    typing.Literal[0]
        Amplitude attenuation, in m^-1

    Notes
    -----
    The attenuation is defined as:

    .. math::

        \alpha_j = 0 \forall j.

    """
    return 0


def parameterisation_01(thickness: float, wavenumbers: np.ndarray) -> np.ndarray:
    """Parameterised attenuation for individual wave modes.

    Parameters
    ----------
    thickness : float
        Ice thickness, in m
    wavenumbers : np.ndarray
        Propagating wavenumbers, in rad m^-1

    Returns
    -------
    np.ndarray
        Amplitude attenuation, in m^-1

    Notes
    -----
    The attenuation is defined as:

    .. math::

        \alpha_j = \frac{1}{4} {k_j}^2 h.

    """
    return wavenumbers**2 * thickness / 4


class AttenuationParameterisation(enum.Enum):
    NO = 0
    PARAM_01 = 1


@attrs.frozen
class AttenuationSpecification:
    function: typing.Callable
    args: str | None = None
    kwargs: dict[str, typing.Any] = attrs.field(factory=dict)


Attenuation: typing.TypeAlias = AttenuationParameterisation | AttenuationSpecification
