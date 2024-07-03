import numpy as np
import pathlib
import pytest


from flexfrac1d.model.model import Ice, Ocean, DiscreteSpectrum, Floe, Domain
import flexfrac1d.model.frac_handlers as fh

PATH_BIN_EGY = pathlib.Path("tests/target/fracture")


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


@pytest.mark.slow
def test_binary_energy_no_growth():
    growth_params = None
    an_sol = True
    binary_handler = fh.BinaryFracture()
    target = np.loadtxt(PATH_BIN_EGY.joinpath("binary_fracture.ssv"))

    for row in target:
        wuf = make_wuf(row[:-1], growth_params)
        xf = binary_handler.search(wuf, growth_params, an_sol, None)
        if xf is not None:
            assert np.allclose(row[-1] - xf, 0)
        else:
            assert np.isnan(row[-1])


def test_binary_strain_no_growth():
    growth_params = None
    an_sol = True
    binary_handler = fh.BinaryStrainFracture()
    target = np.loadtxt(PATH_BIN_EGY.joinpath("binary_strain_fracture.ssv"))

    for row in target:
        wuf = make_wuf(row[:-1], growth_params)
        xf = binary_handler.search(wuf, growth_params, an_sol, None)
        if xf is not None:
            assert np.allclose(row[-1] - xf, 0)
        else:
            assert np.isnan(row[-1])
