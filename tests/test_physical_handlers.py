import numpy as np
import pytest

import flexfrac1d.lib.physics as ph

floe_params = (0.36, 7.8)
wave_params = (
    (
        np.array([0.00950574 + 0.13669057j]),
        np.array([0.02660581 + 0.02685434j]),
    ),  # monochromatic
    (
        (
            np.array([0.00950574 + 0.13669057j, 0.02660581 + 0.0265434j]),
            np.array([0.03552382 + 0.05215654j, 0.06214718 + 0.1250975j]),
        )  # polychromatic
    ),
)
growth_params = (None, True)

handlers = [
    ph.FluidSurfaceHandler,
    ph.DisplacementHandler,
    ph.CurvatureHandler,
    ph.StrainHandler,
]
x_as_real = (0, 1, 1.0, 3.18, np.array(1.8))
x_as_list_or_tuple = (
    (0, 1),
    [0, 1],
    (0.3, 1.8),
    [0.3, 1.8],
    (2.5,),
    [2.5],
)
x_as_array = tuple([np.asarray(_x) for _x in x_as_list_or_tuple[1::2]])
thickness = 0.5


def prepare_instance(growth_params, handler, wave_params):
    if growth_params is not None:
        one_and_maybe_two = np.linspace(1, 2, len(wave_params[0]))
        # Set arbitrary growth kernel with correct shape. Setting the mean to a
        # negative number ensures a numerical solution is used.
        growth_params = (-3 * one_and_maybe_two[:, None], 20)

    if handler == ph.FluidSurfaceHandler:
        handler_instance = handler(wave_params, growth_params)
    elif handler == ph.StrainHandler:
        handler_instance = handler(
            ph.CurvatureHandler(floe_params, wave_params, growth_params), thickness
        )
    else:
        handler_instance = handler(floe_params, wave_params, growth_params)
    return handler_instance


@pytest.mark.parametrize("wave_params", wave_params)
@pytest.mark.parametrize("growth_params", growth_params)
@pytest.mark.parametrize("x", x_as_real)
@pytest.mark.parametrize("handler", handlers)
def test_shape_real(wave_params, growth_params, x, handler):
    handler_instance = prepare_instance(growth_params, handler, wave_params)
    res = handler_instance.compute(x)
    with pytest.raises(TypeError):
        len(res)


@pytest.mark.parametrize("wave_params", wave_params)
@pytest.mark.parametrize("growth_params", growth_params)
@pytest.mark.parametrize("x", x_as_list_or_tuple + x_as_array)
@pytest.mark.parametrize("handler", handlers)
def test_shape_array(wave_params, growth_params, x, handler):
    handler_instance = prepare_instance(growth_params, handler, wave_params)
    res = handler_instance.compute(x)
    assert len(x) == res.size
