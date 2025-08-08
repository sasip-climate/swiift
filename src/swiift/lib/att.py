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


def parameterisation_yu2022(
    thickness: float, gravity: float, angular_frequencies: np.ndarray
) -> np.ndarray:
    r"""Parameterised attenuation for individual wave modes.

    This parameterisation is issued from Yu et al. (2022) [1]_.

    .. versionadded:: 0.16.0

    Parameters
    ----------
    thickness : float
        Ice thickness, in m.
    gravity : float
        Acceleration of gravity, in m s**-2.
    angular_frequencies : np.ndarray
        Angular frequencies, in rad s**-1.

    Returns
    -------
    np.ndarray
        Amplitude attenuation rates, in m**-1.

    Notes
    -----
    The attenuation is defined as:

    .. math::

        \alpha_j h = 0.108 {(\omega\sqrt{\frac{h}{g}})}^4.46

    where the prefactor and exponents were obtained by a best fit to
    available data [1]_.

    References
    ----------
    .. [1] Yu, J., W. E. Rogers, and D. W. Wang (2022). A new method for
    parameterization of wave dissipation by sea ice. Cold Regions Science and
    Technology 199, p. 103582.
    DOI: https://doi.org/10.1016/j.coldregions.2022.103582.

    """
    prefactor, exponent = 0.108, 4.46
    return (
        prefactor
        * angular_frequencies**exponent
        * thickness ** (exponent / 2 - 1)
        / gravity ** (exponent / 2)
    )


class AttenuationParameterisation(enum.Enum):
    NO = 0
    PARAM_01 = 1
    PARAM_YU_2022 = 20


@attrs.frozen
class AttenuationSpecification:
    function: typing.Callable
    args: str | None = None
    kwargs: dict[str, typing.Any] = attrs.field(factory=dict)


Attenuation: typing.TypeAlias = AttenuationParameterisation | AttenuationSpecification
