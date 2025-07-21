"""Define strategies for physical scalar variables."""

from types import MappingProxyType
import typing

from hypothesis import strategies as st
import numpy as np

from swiift.lib.constants import PI_2
from swiift.model.model import Ice
from tests.utils import FloatSt, float_kw

# Exagerated bounds for all independent physical parameters.
# Allows for testing exotic situations, without going into extremes either.
PHYSICAL_STRATEGIES = MappingProxyType(
    {
        ("floe", "left_edge"): st.floats(0, 5e3, **float_kw),
        ("ocean", "depth"): (
            st.floats(min_value=1e-3, max_value=1000e3, **float_kw) | st.just(np.inf)
        ),
        ("ocean", "density"): st.floats(500, 5e3, **float_kw),
        ("ice", "frac_toughness"): st.floats(1e3, 1e7, **float_kw),
        ("ice", "poissons_ratio"): st.floats(-0.999, 0.5, **float_kw),
        ("ice", "strain_threshold"): st.floats(1e-7, 1e-3, **float_kw),
        ("ice", "thickness"): st.floats(1e-3, 1e3, **float_kw),
        ("ice", "youngs_modulus"): st.floats(1e6, 100e9, **float_kw),
        ("ice", "elastic_length"): st.floats(5e-4, 3e4, **float_kw),
        ("wave", "amplitude"): st.floats(1e-6, 1e3, **float_kw),
        ("wave", "period"): st.floats(min_value=1e-1, max_value=1e4, **float_kw),
        ("wave", "frequency"): st.floats(min_value=1e-4, max_value=10, **float_kw),
        ("wave", "phase"): st.floats(0, PI_2, exclude_max=True, **float_kw),
        ("wave", "wavenumber"): st.floats(7e-4, 600, **float_kw),
        ("gravity",): st.floats(0.1, 30, **float_kw),
    }
)


# For the composite strategies, set somewhat stricter upper bounds than plain
# inequality to avoid head-scratching floating point innacuracies.
@st.composite
def ice_density(draw: st.DrawFn, ocean_density: FloatSt) -> float:
    """Ice density contrained by ocean density.

    We typically want ice to have a lower density for it to float.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis dynamic callable.
    ocean_density : FloatSt
        Ocean density in kg m**-3.

    Returns
    -------
    float
        Ice density in kg m**-3.

    """
    ex = draw(ocean_density)
    return draw(st.floats(10, 0.9999 * ex, allow_nan=False, allow_subnormal=False))


@st.composite
def ice_thickness(
    draw: st.DrawFn,
    ocean_density: FloatSt,
    ocean_depth: FloatSt,
    ice_density: FloatSt,
) -> float:
    """Ice thickness constrained by ocean depth and ice draught.

    We typically want ice not to be grounded. That is, it must be thin enough
    that its draught is less than the water depth.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis dynamic callable.
    ocean_density : FloatSt
        Strategy for ocean density in kg m**-3.
    ocean_depth : FloatSt
        Strategy for ocean depth in m.
    ice_density : FloatSt
        Strategy for ice density in kg m**-3.

    Returns
    -------
    float
        Ice thickness in m.

    """
    kwgs = {"rhow": ocean_density, "rhoi": ice_density, "H": ocean_depth}
    ex = {k: draw(v) for k, v in kwgs.items()}
    upper_bound = 0.9999 * ex["rhow"] / ex["rhoi"] * ex["H"]
    return draw(
        st.floats(
            0.1e-3,
            min(1000, upper_bound),
            exclude_max=True,
            allow_nan=False,
            allow_subnormal=False,
        )
    )


@st.composite
def floe_length(draw: st.DrawFn, ice: st.SearchStrategy[Ice]) -> float:
    """Floe length constrained by ice thickness.

    We typically want floes to be longer than they are thick for stable buoyancy.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis dynamic callable.
    ice : st.SearchStrategy[Ice]
        Strategy for an ice object.

    Returns
    -------
    float
        Floe length in m.

    """
    return draw(
        st.floats(
            2 * draw(ice).thickness, 1000e3, allow_nan=False, allow_subnormal=False
        )
    )


PHYSICAL_STRATEGIES_COMPOSITE: MappingProxyType[str, typing.Callable[..., FloatSt]] = (
    MappingProxyType(
        {
            "floe_length": floe_length,
            "ice_density": ice_density,
            "ice_thickness": ice_thickness,
        }
    )
)

