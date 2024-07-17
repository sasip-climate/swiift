import numpy as np
import pathlib
import pytest


from flexfrac1d.model.model import Ice, Ocean, DiscreteSpectrum, Floe, Domain
import flexfrac1d.model.frac_handlers as fh

PATH_FRACTURE_TARGETS = pathlib.Path("tests/target/fracture")
binary_energy_no_growth_target = np.loadtxt(
    PATH_FRACTURE_TARGETS.joinpath("binary_fracture.ssv")
)
binary_strain_no_growth_target = np.loadtxt(
    PATH_FRACTURE_TARGETS.joinpath("binary_strain_fracture.ssv")
)
multi_strain_no_growth_archive = np.load(
    PATH_FRACTURE_TARGETS.joinpath("multi_strain_fracture.npz")
)
multi_strain_no_growth_target = (
    (row, multi_strain_no_growth_archive[f"res{i:02d}"])
    for i, row in enumerate(multi_strain_no_growth_archive["params"])
)


def make_wuf(array, growth_params):
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
    domain = Domain.from_discrete(gravity, spectrum, ocean, growth_params)
    floe = Floe(left_edge=left_edge, length=length, ice=ice)
    domain.add_floes(floe)
    return domain.subdomains[0]


def test_abstract():
    # Abstract classes, should not be instantiated
    with pytest.raises(TypeError):
        fh._FractureHandler()
    with pytest.raises(TypeError):
        fh._StrainFracture()


@pytest.mark.slow
@pytest.mark.parametrize("row", binary_energy_no_growth_target)
def test_binary_energy_no_growth(row):
    growth_params = None
    an_sol = True
    binary_handler = fh.BinaryFracture()

    wuf = make_wuf(row[:-1], growth_params)
    xf = binary_handler.search(wuf, growth_params, an_sol, None)
    if xf is not None:
        assert np.allclose(row[-1] - xf, 0)
    else:
        assert np.isnan(row[-1])


@pytest.mark.parametrize("row", binary_strain_no_growth_target)
def test_binary_strain_no_growth(row):
    growth_params = None
    an_sol = True
    binary_handler = fh.BinaryStrainFracture()

    wuf = make_wuf(row[:-1], growth_params)
    xf = binary_handler.search(wuf, growth_params, an_sol, None)
    if xf is not None:
        assert np.allclose(row[-1] - xf, 0)
    else:
        assert np.isnan(row[-1])


@pytest.mark.parametrize("row, target", multi_strain_no_growth_target)
def test_multi_strain_no_growth(row, target):
    growth_params = None
    an_sol = True
    handler = fh.MultipleStrainFracture()

    wuf = make_wuf(row, growth_params)
    xfs = handler.search(wuf, growth_params, an_sol, None)
    if xfs is not None:
        assert np.allclose(target - xfs, 0)
    else:
        assert np.isnan(target)
