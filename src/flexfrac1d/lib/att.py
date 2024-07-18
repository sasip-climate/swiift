# /usr/bin/env python3

from __future__ import annotations

import abc
import attrs
import numpy as np

from ..model import model


@attrs.define(frozen=True, repr=False)
class _AttenuationParameterisation(abc.ABC):
    ice: model.FloatingIce
    wavenumbers: np.ndarray

    @abc.abstractmethod
    def compute(self):
        raise NotImplementedError


@attrs.define(frozen=True)
class NoAttenuation(_AttenuationParameterisation):
    def compute(self):
        return 0


@attrs.define(frozen=True)
class AttenuationParametrisation01(_AttenuationParameterisation):
    def compute(self):
        return self.wavenumbers**2 * self.ice.thickness / 4
