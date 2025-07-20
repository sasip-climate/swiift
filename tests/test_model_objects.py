import pytest

import swiift.model.model as md


class TestSubdomain:
    @staticmethod
    def test_ordering():
        left_edge, length = 0, 100
        sd = md._Subdomain(left_edge, length)

        left_edge_further_right = left_edge + length + 1
        assert left_edge_further_right > sd
        sd_right = md._Subdomain(left_edge_further_right, length)
        assert sd_right > sd

        left_edge_further_left = -200
        assert left_edge_further_left < sd
        sd_left = md._Subdomain(left_edge_further_left, length)
        assert sd_left < sd

        assert sd == left_edge
        sd_same = md._Subdomain(left_edge, length)
        assert sd_same == sd
        sd_same_but_different_length = md._Subdomain(left_edge, 2 * length)
        assert sd_same_but_different_length == sd

        with pytest.raises(TypeError):
            sd > []  # noqa: B015

    @staticmethod
    @pytest.mark.parametrize("left_edge", (-100, 0, 0.8, 45, 34.2))
    @pytest.mark.parametrize("length", (10, 13.8, 101.9))
    def test_right_edge(left_edge, length):
        sd = md._Subdomain(left_edge, length)
        assert sd.right_edge == left_edge + length
