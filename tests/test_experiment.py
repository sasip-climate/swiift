from hypothesis import given
from sortedcontainers import SortedList

from .conftest import physical_strategies, coupled_floe, spec_mono
from flexfrac1d.api.api import Experiment
import flexfrac1d.lib.att as att
import flexfrac1d.model.frac_handlers as fh
from flexfrac1d.model.model import Domain


@given(spectrum=spec_mono(), **coupled_floe)
def test_initialisation(gravity, spectrum, ocean, floe, ice):
    experiment = Experiment.from_discrete(gravity, spectrum, ocean)

    assert experiment.time == 0
    assert isinstance(experiment.domain, Domain)
    assert (
        isinstance(experiment.domain.subdomains, SortedList)
        and len(experiment.domain.subdomains) == 0
    )
    assert (
        isinstance(experiment.domain.attenuation, att.AttenuationParameterisation)
        and experiment.domain.attenuation == att.AttenuationParameterisation.PARAM_01
    )
    assert isinstance(experiment.history, dict) and len(experiment.history) == 0
    assert isinstance(experiment.fracture_handler, fh.BinaryFracture)
