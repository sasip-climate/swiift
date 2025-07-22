import pytest

import swiift.model.model as md


@pytest.mark.parametrize("left_edge", (-100, 0, 0.8, 45, 34.2))
@pytest.mark.parametrize("length", (10, 13.8, 101.9))
def test_right_edge(left_edge, length):
    sd = md._Subdomain(left_edge, length)
    assert sd.right_edge == left_edge + length
