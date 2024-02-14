#!/usr/bin/env python3

from hypothesis import assume, given, strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import scipy.integrate as integrate

from flexfrac1d.flexfrac1d import Ocean, OceanCoupled, DiscreteSpectrum
from flexfrac1d.flexfrac1d import Floe, FloeCoupled, Ice, IceCoupled

# from flexfrac1d.libraries.WaveUtils import free_surface, elas_mass_surface


def fun(x, w, *, floe: FloeCoupled, spectrum: DiscreteSpectrum):
    wprime = np.vstack(
        (
            w[1],
            w[2],
            w[3],
            -(
                floe.ice._red_elastic_number
                * (
                    w[0]
                    + floe._wavefield(x, floe._dis_par_amps(spectrum))
                    - floe._mean_wavefield(spectrum)
                )
            ),
        )
    )
    return wprime


def bc(wa, wb):
    return np.array((wa[2], wb[2], wa[3], wb[3]))


float_kw = {
    "min_value": 5e-39,
    "max_value": 4e51,
    "allow_nan": False,
    "exclude_min": True,
    "allow_subnormal": False,
}


@st.composite
def setup(draw):
    num_freqs = draw(st.integers(min_value=1, max_value=50))
    # 100 Hz should be moooooore than enough, and will avoid wavenumber finding warnings
    freqs = draw(
        st.lists(
            st.floats(**(float_kw | {"max_value": 100.0})),
            min_size=num_freqs,
            max_size=num_freqs,
        )
    )
    # Make sure we stay far from overflowing when summing the amplitudes
    amps = draw(
        st.lists(
            st.floats(
                **(float_kw | {"max_value": float_kw["max_value"] / (2 * num_freqs)})
            ),
            min_size=num_freqs,
            max_size=num_freqs,
        )
    )
    spec = DiscreteSpectrum(amps, freqs)
    phases = draw(
        st.lists(st.floats(**float_kw), min_size=num_freqs, max_size=num_freqs)
    )

    rho_i, rho_w = sorted(
        draw(st.lists(st.floats(**float_kw), min_size=2, max_size=2, unique=True))
    )

    depth = draw(st.floats(**float_kw) | st.just(np.inf))
    ocean = Ocean(depth=depth, density=rho_w)

    ice = draw(
        st.builds(
            Ice,
            density=st.just(rho_i),
            frac_energy=st.floats(**float_kw),
            poissons_ratio=st.floats(
                min_value=-1,
                max_value=0.5,
                allow_nan=False,
                allow_subnormal=False,
                exclude_min=True,
            ),
            thickness=st.floats(
                **(
                    float_kw
                    | {
                        "max_value": min(
                            float_kw["max_value"] / 10,
                            np.nextafter(ocean.depth * rho_w / rho_i, 0),
                        )  # ensure H -d > 0
                    }
                )
            ),
            youngs_modulus=st.floats(**float_kw),
        )
    )
    assume(ocean.depth - ice.thickness * ice.density / ocean.density > 0)

    gravity = draw(st.floats(**float_kw))

    co = OceanCoupled(ocean, spec, gravity)
    ci = IceCoupled(ice, co, spec, None, gravity)

    # Correspond to max 20 initial nodes with 2x Shannon's frequency wrt the floe length
    ubound_length = 10 * np.pi / ci.wavenumbers.max()
    lbound_length = 5 * ice.thickness
    scaling_factor = 2**0.5 * ci.elastic_length
    if scaling_factor < 1:
        ubound_length = np.nextafter(
            min(ubound_length, float_kw["max_value"] * scaling_factor), -np.inf
        )
    else:
        lbound_length = np.nextafter(
            max(lbound_length, float_kw["min_value"] * scaling_factor), np.inf
        )
    assume(lbound_length < ubound_length)

    length = draw(
        st.floats(
            **(float_kw | {"min_value": lbound_length, "max_value": ubound_length})
        )
    )
    left_edge = draw(st.just(10.0))  # should be irrelevant for now
    floe = Floe(left_edge, length, ice, None)
    cf = FloeCoupled(floe, ci, spec, phases)

    return cf, spec


int_kw = {"min_value": 1, "max_value": 50}


@given(
    # ocean=st.builds(
    #     Ocean,
    #     depth=st.floats(**float_kw) | st.just(np.inf),
    #     density=st.floats(**float_kw),
    # ),
    # spec=st.builds
    #     DiscreteSpectrum,
    #     st.lists(
    #         st.floats(**float_kw),
    #         min_size=st.shared(st.integers(**int_kw), key="nf"),
    #         max_size=st.shared(st.integers(**int_kw), key="nf"),
    #     ),
    #     st.lists(
    #         st.floats(**float_kw),
    #         min_size=st.shared(st.integers(**int_kw), key="nf"),
    #         max_size=st.shared(st.integers(**int_kw), key="nf"),
    #     ),
    # ),
    # ice=st.builds(
    #     Ice,
    #     density=st.floats(**float_kw),
    #     frac_energy=st.floats(**float_kw),
    #     poissons_ratio=st.floats(
    #         min_value=-1,
    #         max_value=0.5,
    #         allow_nan=False,
    #         allow_subnormal=False,
    #         exclude_min=True,
    #     ),
    #     thickness=st.floats(**float_kw),
    #     youngs_modulus=st.floats(**float_kw),
    # ),
    # left_edge=st.just(10),
    # length=st.floats(**float_kw),
    # phases=st.lists(
    #     st.floats(**float_kw),
    #     min_size=st.shared(st.integers(**int_kw), key="nf"),
    #     max_size=st.shared(st.integers(**int_kw), key="nf"),
    # ),
    # gravity=st.floats(**float_kw),
    args=setup(),
)
def test_displacement(
    # ocean: Ocean,
    # ice: Ice,
    # left_edge: float,
    # length: float,
    # gravity: float,
    args: tuple[DiscreteSpectrum, list[float]],
):
    # spec, phases = args
    # assume(ocean.density > ice.density)
    # assume(length > ice.thickness)
    # Has to be done manually as instantiation of the IceCoupled object can
    # fail if not respected
    # assume(ocean.depth - ice.thickness * ice.density / ocean.density > 0)
    # co = OceanCoupled(ocean, spec, gravity)
    # ci = IceCoupled(ice, co, spec, None, gravity)
    # floe = Floe(left_edge, length, ice, None)
    # cf = FloeCoupled(floe, ci, spec, phases)
    # assert cf.ice is not None

    cf, spec = args
    n_mesh = max(
        4, np.ceil(4 * cf.length / (2 * np.pi / cf.ice.wavenumbers.max())).astype(int)
    )
    x = np.linspace(0, cf.length, n_mesh)
    w = np.zeros((4, x.size))
    sol = integrate.solve_bvp(
        lambda x, w: fun(x, w, floe=cf, spectrum=spec),
        bc,
        x,
        w,
        tol=1e-6,
    )
    # x_hd = np.linspace(0, cf.length, n_mesh * 20)
    if sol.success:
        assert np.allclose(cf.displacement(sol.x, spec), sol.y[0])
    # assert np.allclose(cf.displacement(x_hd, spec), sol.sol(x_hd)[0])
