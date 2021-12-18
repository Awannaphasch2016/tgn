#!/usr/bin/env python3

from utils.utils import get_different_edges_mask_left

import numpy as np
import pytest

@pytest.mark.usefixtures("edges", "sources", "destination")
def test_get_different_edges_mask_left(mocker, edges, sources, destination):
    left_edges = np.array([[1,4],[2,5],[3,6],[3,5],[3,5]])
    right_edges = np.array([[1,4],[3,6],[7,8]])
    only_left_edges_mask = get_different_edges_mask_left(left_edges,right_edges)

    print(only_left_edges_mask)
    assert len(only_left_edges_mask) == left_edges.shape[0]
    assert sum(only_left_edges_mask) == 3

    left_edges = np.array([[1,2,3,3,3],[4,5,6,5,5]]).T
    right_edges = np.array([[1,3,7],[4,6,8]]).T
    only_left_edges_mask = get_different_edges_mask_left(left_edges,right_edges)

    print(only_left_edges_mask)
    assert len(only_left_edges_mask) == left_edges.shape[0]
    assert sum(only_left_edges_mask) == 3

    left_edges = np.array([[1,4],[3,6],[7,8]])
    right_edges = np.array([[1,4],[2,5],[3,6],[3,5],[3,5]])
    only_left_edges_mask = get_different_edges_mask_left(left_edges,right_edges)

    print(only_left_edges_mask)
    assert len(only_left_edges_mask) == left_edges.shape[0]
    assert sum(only_left_edges_mask) == 1




if __name__ == '__main__':
    pass
