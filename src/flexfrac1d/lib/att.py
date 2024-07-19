"""Attenuation parameterisations."""

from __future__ import annotations

import abc
import attrs
import numpy as np
import typing

from ..model import model


@attrs.define(frozen=True, repr=False)
class AttenuationParameterisation(abc.ABC):
    """Base class to be derived by parameterisations.

    Parameters
    ----------
    ice : FloatingIce
        An object instance encapsulating mechanical properties
    wavenumbers : 1d np.ndarray of float
       Propagating wavenumbers to attenuate, in rad m^-1

    """

    ice: model.FloatingIce
    wavenumbers: np.ndarray

    @abc.abstractmethod
    def compute(self):
        raise NotImplementedError


@attrs.define(frozen=True)
class NoAttenuation(AttenuationParameterisation):
    """No attenuation.

    Waves propagate indifinitely, as if the ice cover is perfectly elastic and
    the fluid perfectly inviscid.

    Notes
    -----
    The attenuation is defined as:

    .. math::

        \alpha_j = 0 \forall j.

    """

    def compute(self) -> typing.Literal[0]:
        """Compute attenuation.

        Returns
        -------
        typing.Literal[0]
            Amplitude attenuation, in m^-1

        """
        return 0


@attrs.define(frozen=True)
class AttenuationParametrisation01(AttenuationParameterisation):
    """Parameterised attenuation for individual wave modes.

    Attenuation proportional to the squared wavenumber and the ice thickness.

    Notes
    -----
    The attenuation is defined as:

    .. math::

        \alpha_j = \frac{1}{4} {k_j}^2 h.

    """

    def compute(self) -> np.ndarray:
        """Compute attenuation.

        Returns
        -------
        np.ndarray
            Amplitude attenuation, in m^-1

        """
        return self.wavenumbers**2 * self.ice.thickness / 4
