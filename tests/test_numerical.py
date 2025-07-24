from hypothesis import given, strategies as st
import numpy as np

import swiift.lib.numerical as numerical
from tests.utils import float_kw


@st.composite
def make_use_an_sol_params(
    draw: st.DrawFn,
) -> tuple[bool | None, float, tuple[np.ndarray, float] | None, bool | None]:
    an_sol = draw(st.booleans() | st.none())
    length = draw(st.floats(-100, 100, **float_kw))
    _mean = draw(
        st.lists(st.floats(-100, 100, **float_kw), min_size=1, max_size=5) | st.none()
    )
    if _mean is not None:
        growth_params = np.array(_mean), 1.0  # value of std does not matter
    else:
        growth_params = None
    linear_curvature = draw(st.booleans() | st.none())
    return an_sol, length, growth_params, linear_curvature


@given(params=make_use_an_sol_params())
def test_use_an_sol(params):
    an_sol, length, growth_params, linear_curvature = params
    use_an_sol = numerical._use_an_sol(*params)

    if an_sol is not None:
        assert use_an_sol == an_sol
    else:
        if growth_params is None:
            if linear_curvature is None:
                assert use_an_sol is True
            else:
                assert use_an_sol == linear_curvature
        else:
            if np.any(growth_params[0] < length):
                assert use_an_sol is False
            else:
                if linear_curvature is None:
                    assert use_an_sol is True
                else:
                    assert use_an_sol is linear_curvature
