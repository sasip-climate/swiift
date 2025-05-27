import numpy as np
import pytest

from swiift.lib import graphics as gr


@pytest.mark.parametrize("resolution", (1, 0.5, 2.1, 4))
def test_num_for_linspace(resolution):
    lengths = (23, 12.8, 36.7)
    nums = gr._linspace_nums(resolution, lengths)
    assert len(nums) == len(lengths)
    for num, length in zip(nums, lengths):
        x = np.linspace(0, length, num)
        assert x[0] <= resolution
