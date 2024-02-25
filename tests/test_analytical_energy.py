#!/usr/bin/env python3

from hypothesis import assume, given, strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import scipy.integrate as integrate

from flexfrac1d.flexfrac1d import Ocean, OceanCoupled, DiscreteSpectrum
from flexfrac1d.flexfrac1d import Floe, FloeCoupled, Ice, IceCoupled

from hypothesis import settings, reproduce_failure, Phase, Verbosity


def fun(x, w, *, floe: FloeCoupled, spectrum: DiscreteSpectrum):
    wprime = np.vstack(
        (
            w[1],
            w[2],
            w[3],
            -(
                w[0]
                + floe._wavefield(x, spectrum._amps * np.exp(1j * floe.phases))
                - floe._mean_wavefield(spectrum._amps)
            )
            / floe.ice._elastic_length_pow4,
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
                        ),  # ensure H -d > 0
                        # "min_value": 1e-9,
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
    # Correspond to sinh(x)^2 > sin(x)^2 == True, numerically
    assume(length * ci._red_elastic_number > 2e-8)
    assume(np.exp(-2 * length * ci._red_elastic_number) != 1)
    left_edge = draw(st.just(10.0))  # should be irrelevant for now
    floe = Floe(left_edge, length, ice, None)
    cf = FloeCoupled(floe, ci, spec, phases)

    return cf, spec, co, gravity


@given(args=setup())
@settings(
    # verbosity=Verbosity.verbose,
    # phases=[Phase.explicit, Phase.reuse, Phase.generate],
    deadline=None,
)
def _test_displacement(
    args: tuple[DiscreteSpectrum, list[float], OceanCoupled, float],
):
    cf, spec, ocean, gravity = args
    n_mesh = max(
        4, np.ceil(4 * cf.length / (2 * np.pi / cf.ice.wavenumbers.max())).astype(int)
    )
    x = np.linspace(0, cf.length, n_mesh)

    if np.all(np.isfinite(cf.displacement(x, spec))):
        w = np.zeros((4, x.size))
        sol = integrate.solve_bvp(
            lambda x, w: fun(x, w, floe=cf, spectrum=spec),
            bc,
            x,
            w,
            tol=1e-4,
        )
        if sol.success:
            try:
                assert np.allclose(cf.displacement(sol.x, spec), sol.y[0])
            except AssertionError:
                # man_res = np.sqrt(
                #     [
                #         integrate.quad(
                #             lambda x: (sol.sol(x)[0] - cf.displacement(x, spec)) ** 2,
                #             sol.x[i],
                #             sol.x[i + 1],
                #         )[0]
                #         for i in range(sol.x.size - 1)
                #     ]
                # )
                man_res = np.sqrt(
                    np.abs(
                        np.ediff1d((sol.y[0] - cf.displacement(sol.x, spec)) ** 2)
                        * np.ediff1d(sol.x)
                        / 2
                    )
                )
                assert np.all(man_res <= sol.rms_residuals)
