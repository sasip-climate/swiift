r"""
Attenuation parametrisations
============================

We model forcing waves as the superposition of damped sine functions,
which can be written

.. math::

    \eta(x) = \Im\Big[\sum_j \hat{a}_j \exp\big(\hat{k}_jx\big)\Big],

with :math:`\hat{a}_j,\ \hat{k}_j` respectively the complex amplitude and
wavenumber associated with wavemode :math:`j`.
The non-negative imaginary parts of these wavenumbers,
:math:`\alpha_j := \Im \hat{k}_j`, correspond to spatial attenuation rates.

This module exposes **parametrisations** of the attenuation rates, which are
functions implementing existing attenuation schemes.
Alternatively, it exposes an interface for users to define attenuation
**specifications** in order to express their own schemes.

Parametrisations
----------------
:py:func:`no_attenuation` -- No attenuation.

:py:func:`parameterisation_01` -- Derived from :cite:t:`att-Sutherland2019`.

:py:func:`parameterisation_yu2022` -- Implements :cite:t:`att-Yu2022`.

Specification
-------------
TODO

Bibliography
------------
.. bibliography::
    :keyprefix: att-

"""

import enum
import typing

import attrs
import numpy as np

# TODO: rename occurences of 'parameterisation' to 'parametrisation' for
# consistency
# TODO: add the full param from att-Sutherland2019?


def no_attenuation() -> typing.Literal[0]:
    r"""No attenuation.

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
    r"""Parametrised attenuation for individual wave modes.

    This parametrisation is a simplified form of that of
    :cite:t:`att-Sutherland2019`.

    Parameters
    ----------
    thickness : float
        Ice thickness, in m
    wavenumbers : np.ndarray
        Propagating wavenumbers, in rad m^-1.

    Returns
    -------
    np.ndarray
        Amplitude attenuation, in m^-1.

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
    r"""Parametrised attenuation for individual wave modes.

    This parametrisation is issued from :cite:t:`att-Yu2022`.

    Parameters
    ----------
    thickness : float
        Ice thickness, in m.
    gravity : float
        Acceleration of gravity, in m s^-2.
    angular_frequencies : np.ndarray
        Angular frequencies, in rad s^-1.

    Returns
    -------
    np.ndarray
        Amplitude attenuation rates, in m^-1.

    Notes
    -----
    The attenuation is defined as:

    .. math::

        \alpha_j = 0.108 \frac{1}{h}\Bigg(\omega\sqrt{\frac{h}{g}}\Bigg)^{4.46}

    where the prefactor and exponents were obtained by a best fit to
    available data :cite:p:`att-Yu2022`.

    .. version-added:: 0.16.0

    """
    prefactor, exponent = 0.108, 4.46
    return (
        prefactor
        * angular_frequencies**exponent
        * thickness ** (exponent / 2 - 1)
        / gravity ** (exponent / 2)
    )


class AttenuationParameterisation(enum.Enum):
    """Unique IDs for attenuation parametrisations.

    Attributes
    ----------
    NO : defer to `no_attenuation`.
    PARAM_01 : defer to `parameterisation_01`.
    PARAM_YU_2022 : defer to `parameterisation_yu2022`.

    """

    NO = 0
    PARAM_01 = 1
    PARAM_YU_2022 = 20


@attrs.frozen
class AttenuationSpecification:
    function: typing.Callable
    args: str | None = None
    kwargs: dict[str, typing.Any] = attrs.field(factory=dict)


Attenuation: typing.TypeAlias = AttenuationParameterisation | AttenuationSpecification
