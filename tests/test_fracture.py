import pathlib
from typing import Type

import numpy as np
import pytest

import swiift.lib.phase_shift as ps
import swiift.model.frac_handlers as fh
import swiift.model.model as model
from swiift.model.model import DiscreteSpectrum, Domain, Floe, Ice, Ocean
from tests.utils import fracture_handler_types

scattering_handler_types = (
    ps.ContinuousScatteringHandler,
    ps.UniformScatteringHandler,
    ps.PerturbationScatteringHandler,
)

# Data for stability tests
PATH_FRACTURE_TARGETS = pathlib.Path("tests/target/fracture")
binary_energy_no_growth_target = np.loadtxt(
    PATH_FRACTURE_TARGETS.joinpath("binary_fracture.ssv")
)
binary_energy_with_growth_target = np.loadtxt(
    PATH_FRACTURE_TARGETS.joinpath("binary_fracture_with_growth.ssv")
)
binary_strain_no_growth_target = np.loadtxt(
    PATH_FRACTURE_TARGETS.joinpath("binary_strain_fracture.ssv")
)
binary_strain_with_growth_target = np.loadtxt(
    PATH_FRACTURE_TARGETS.joinpath("binary_strain_fracture_with_growth.ssv")
)
multi_strain_no_growth_archive = np.load(
    PATH_FRACTURE_TARGETS.joinpath("multi_strain_fracture.npz")
)
multi_strain_no_growth_target = (
    (row, multi_strain_no_growth_archive[f"res{i:02d}"])
    for i, row in enumerate(multi_strain_no_growth_archive["params"])
)
multi_strain_with_growth_archive = np.load(
    PATH_FRACTURE_TARGETS.joinpath("multi_strain_fracture_with_growth.npz")
)
multi_strain_with_growth_target = (
    (row, multi_strain_with_growth_archive[f"res{i:02d}"])
    for i, row in enumerate(multi_strain_with_growth_archive["params"])
)


def make_wuf(array: np.ndarray, growth_params: tuple | None) -> model.WavesUnderFloe:
    (
        frac_toughness,
        strain_threshold,
        thickness,
        youngs_modulus,
        depth,
        gravity,
        amplitude,
        frequency,
        length,
        left_edge,
        phase,
    ) = array

    ice = Ice(
        frac_toughness=frac_toughness,
        strain_threshold=strain_threshold,
        thickness=thickness,
        youngs_modulus=youngs_modulus,
    )
    ocean = Ocean(depth=depth)
    spectrum = DiscreteSpectrum(
        amplitudes=amplitude, frequencies=frequency, phases=phase
    )
    domain = Domain.from_discrete(gravity, spectrum, ocean, None, growth_params)
    floe = Floe(left_edge=left_edge, length=length, ice=ice)
    domain.add_floes(floe)
    return domain.subdomains[0]


def test_abstract():
    # Abstract classes, should not be instantiated
    with pytest.raises(TypeError):
        fh._FractureHandler()
    with pytest.raises(TypeError):
        fh._StrainFracture()


@pytest.mark.parametrize("fracture_handler_type", fracture_handler_types)
@pytest.mark.parametrize("scattering_spec_type", scattering_handler_types)
def test_initialisation_scattering(
    fracture_handler_type: Type[fh._FractureHandler],
    scattering_spec_type: Type[ps._ScatteringHandler],
):
    def make_handler_from_spec(scattering_spec_type):
        rng_seed = 13
        loc, scale = 0.3, 0.005

        match scattering_spec_type:
            case ps.ContinuousScatteringHandler:
                return scattering_spec_type()
            case ps.UniformScatteringHandler:
                return scattering_spec_type.from_seed(rng_seed)
            case ps.PerturbationScatteringHandler:
                return scattering_spec_type.from_seed(rng_seed, loc, scale)
            case _:  # pragma: no cover
                raise TypeError("Unsupported scattering handler")

    fracture_handler: fh._FractureHandler = fracture_handler_type(
        scattering_handler=make_handler_from_spec(scattering_spec_type)
    )
    assert isinstance(fracture_handler, fracture_handler_type)
    assert isinstance(fracture_handler.scattering_handler, scattering_spec_type)


# Stability tests
@pytest.mark.slow
@pytest.mark.parametrize("row", binary_energy_no_growth_target)
def test_binary_energy_no_growth(row: np.ndarray):
    growth_params = None
    an_sol = True
    binary_handler = fh.BinaryFracture()

    wuf = make_wuf(row[:-1], growth_params)
    xf = binary_handler.search(wuf, growth_params, an_sol, None)
    if xf is not None:
        assert np.allclose(row[-1], xf)
    else:
        assert np.isnan(row[-1])


@pytest.mark.slow
@pytest.mark.parametrize("row", binary_energy_with_growth_target)
def test_binary_energy_with_growth(row: np.ndarray):
    binary_handler = fh.BinaryFracture()
    growth_params = np.atleast_2d(row[-3]), row[-2]
    an_sol = None

    wuf = make_wuf(row[:-3], growth_params)
    xf = binary_handler.search(wuf, growth_params, an_sol, None)
    if xf is not None:
        assert np.allclose(row[-1], xf)
    else:
        assert np.isnan(row[-1])


@pytest.mark.parametrize("row", binary_strain_no_growth_target)
def test_binary_strain_no_growth(row: np.ndarray):
    growth_params = None
    an_sol = True
    binary_handler = fh.BinaryStrainFracture()

    wuf = make_wuf(row[:-1], growth_params)
    xf = binary_handler.search(wuf, growth_params, an_sol, None)
    if xf is not None:
        assert np.allclose(row[-1], xf)
    else:
        assert np.isnan(row[-1])


@pytest.mark.parametrize("row", binary_strain_with_growth_target)
def test_binary_strain_with_growth(row: np.ndarray):
    binary_handler = fh.BinaryStrainFracture()
    growth_params = np.atleast_2d(row[-3]), row[-2]
    an_sol = None

    wuf = make_wuf(row[:-3], growth_params)
    xf = binary_handler.search(wuf, growth_params, an_sol, None)
    if xf is not None:
        assert np.allclose(row[-1], xf)
    else:
        assert np.isnan(row[-1])


@pytest.mark.parametrize("row, target", multi_strain_no_growth_target)
def test_multi_strain_no_growth(row: np.ndarray, target: np.ndarray):
    growth_params = None
    an_sol = True
    handler = fh.MultipleStrainFracture()

    wuf = make_wuf(row, growth_params)
    xfs = handler.search(wuf, growth_params, an_sol, None)
    if xfs is not None:
        assert np.allclose(target, xfs)
    else:
        assert np.isnan(target)


@pytest.mark.parametrize("row, target", multi_strain_with_growth_target)
def test_multi_strain_with_growth(row: np.ndarray, target: np.ndarray):
    handler = fh.MultipleStrainFracture()
    # xf not part of the array here, so slightly different slicing
    growth_params = np.atleast_2d(row[-2]), row[-1]
    an_sol = None

    wuf = make_wuf(row[:-2], growth_params)
    xfs = handler.search(wuf, growth_params, an_sol, None)
    if xfs is not None:
        assert np.allclose(target, xfs)
    else:
        assert np.isnan(target)
