#!/usr/bin/env python3


from utils.utils import get_different_edges_mask_left, get_uniq_nodes_freq_in_window, get_uniq_edges_freq_in_window, compute_nf, compute_ef, compute_iwf, compute_n_window_containing_nodes, compute_n_window_containing_edges, compute_iwf, convert_dict_values_to_np, compute_xf_iwf, get_uniq_x_freq_in_window

import numpy as np
import pytest
import math

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

def test_get_uniq_edges_freq_in_window():
    edges = np.array([[2,101],
                      [1,101],
                      [1,303],
                      [2,202],
                      [2,202],
                      ])
    (uniq_edges, uniq_idx, uniq_edges_freq) = get_uniq_edges_freq_in_window(edges)

    print(uniq_edges)
    assert sum(np.array([1,1,1,2]) -  uniq_edges_freq) == 0
    assert np.array_equal(np.array([[1,101],[1,303],[2,101],[2,202]]), uniq_edges)

def test_get_uniq_nodes_freq_in_window():
    edges = np.array([[2,101],
                      [1,101],
                      [1,303],
                      [2,202],
                      [2,202],
                      ])
    (src_uniq_nodes, src_uniq_idx, uniq_src_nodes_freq) = get_uniq_nodes_freq_in_window(edges[:,0])
    (dst_uniq_nodes, dst_uniq_idx, uniq_dst_nodes_freq) = get_uniq_nodes_freq_in_window(edges[:,1])

    assert sum(np.unique(edges[:,0], return_counts=True)[1] - uniq_src_nodes_freq) == 0
    assert np.unique(edges[:,0], return_counts=True)[0].shape[0] == 2
    assert not np.all(np.array([1,2]) - src_uniq_nodes) # test if all element is 0
    assert np.array_equal(src_uniq_idx, np.array([1,0]))

    assert sum(np.unique(edges[:,1], return_counts=True)[1] - uniq_dst_nodes_freq) == 0
    assert np.unique(edges[:,1], return_counts=True)[0].shape[0] == 3
    assert not np.all(np.array([101,202,303]) - dst_uniq_nodes) # test if all element is 0
    assert np.array_equal(dst_uniq_idx, np.array([0,3,2]))

def test_compute_ef():
    edges = np.array([[2,101],
                      [1,101],
                      [2,202],
                      [1,303],
                      [2,202],
                      ])

    window_size = 2
    current_instances_idx = window_size
    assert window_size < edges.shape[0]

    # src_in_past_windows = edges[:current_instances_idx,0]
    # dst_in_past_windows = edges[:current_instances_idx,1]

    edges_in_current_windows = edges[current_instances_idx:]

    nf = compute_ef(edges_in_current_windows, window_size)

    assert nf.shape[0] == 2
    assert nf.sum() == 1
    assert np.array_equal(nf, np.array([1,2])/3)

def test_compute_nf():
    edges = np.array([[2,101],
                      [1,101],
                      [2,202],
                      [1,303],
                      [2,202],
                      ])

    window_size = 2
    current_instances_idx = window_size
    assert window_size < edges.shape[0]

    # src_in_past_windows = edges[:current_instances_idx,0]
    # dst_in_past_windows = edges[:current_instances_idx,1]

    src_in_current_windows = edges[current_instances_idx:,0]
    dst_in_current_windows = edges[current_instances_idx:,1]

    nf = compute_nf(src_in_current_windows, window_size)

    assert nf.shape[0] == 2
    assert nf.sum() == 1
    assert np.array_equal(nf, np.array([1,2])/3)

    nf = compute_nf(dst_in_current_windows, window_size)

    assert nf.shape[0] == 2
    assert nf.sum() == 1
    assert np.array_equal(nf, np.array([2,1])/3)


def test_compute_n_window_containing_edges():
    edges = np.array([[2,101],
                      [1,101],
                      [2,202],
                      [1,303],
                      [2,202],
                      [2,202],
                      ])

    window_size = 2
    current_instances_idx = window_size
    assert window_size < edges.shape[0]

    edges_in_past_windows = edges[:-window_size]
    edges_in_current_windows = edges[-window_size:]

    n_past_window_contain_current_edges_dict = compute_n_window_containing_edges(edges_in_past_windows, edges_in_current_windows, window_size)

    assert n_past_window_contain_current_edges_dict[(2,202)] == 2
    with pytest.raises(KeyError):
        print(n_past_window_contain_current_edges_dict[(1,101)])
    with pytest.raises(KeyError):
        print(n_past_window_contain_current_edges_dict[(1,303)])
    with pytest.raises(KeyError):
        print(n_past_window_contain_current_edges_dict[(1,201)])

def test_compute_n_window_containing_nodes():
    edges = np.array([[2,101],
                      [1,101],
                      [2,202],
                      [1,303],
                      [2,202],
                      [2,202],
                      ])

    window_size = 2
    current_instances_idx = window_size
    assert window_size < edges.shape[0]

    src_in_past_windows = edges[:-window_size,0]
    dst_in_past_windows = edges[:-window_size,1]

    src_in_current_windows = edges[-window_size:,0]
    dst_in_current_windows = edges[-window_size:,1]

    n_past_window_contain_current_src_dict = compute_n_window_containing_nodes(src_in_past_windows, src_in_current_windows, window_size)
    n_past_window_contain_current_dst_dict = compute_n_window_containing_nodes(dst_in_past_windows, dst_in_current_windows, window_size)

    # n_past_window_contain_current_src_dict = convert_dict_values_to_np(n_past_window_contain_current_src_dict)
    # n_past_window_contain_current_dst_dict = convert_dict_values_to_np(n_past_window_contain_current_dst_dict)

    assert n_past_window_contain_current_src_dict[2] == 3
    with pytest.raises(KeyError):
        print(n_past_window_contain_current_src_dict[1])

    assert n_past_window_contain_current_dst_dict[202] == 2
    with pytest.raises(KeyError):
        print(n_past_window_contain_current_dst_dict[101])
    with pytest.raises(KeyError):
        print(n_past_window_contain_current_dst_dict[303])


    window_size = 2
    src_in_past_windows = edges[:window_size,0]
    dst_in_past_windows = edges[:window_size,1]

    src_in_current_windows = edges[window_size:window_size*2:,0]
    dst_in_current_windows = edges[window_size:window_size*2:,1]

    n_past_window_contain_current_src_dict = compute_n_window_containing_nodes(src_in_past_windows, src_in_current_windows, window_size)
    n_past_window_contain_current_dst_dict = compute_n_window_containing_nodes(dst_in_past_windows, dst_in_current_windows, window_size)

    assert n_past_window_contain_current_src_dict[1] == 2
    assert n_past_window_contain_current_src_dict[2] == 2

    assert n_past_window_contain_current_dst_dict[202] == 1
    assert n_past_window_contain_current_dst_dict[303] == 1
    with pytest.raises(KeyError):
        print(n_past_window_contain_current_dst_dict[101])

    window_size = 3
    current_instances_idx = window_size
    assert window_size < edges.shape[0]

    src_in_past_windows = edges[:-window_size,0]
    dst_in_past_windows = edges[:-window_size,1]

    src_in_current_windows = edges[-window_size:,0]
    dst_in_current_windows = edges[-window_size:,1]

    n_past_window_contain_current_src_dict = compute_n_window_containing_nodes(src_in_past_windows, src_in_current_windows, window_size)
    n_past_window_contain_current_dst_dict = compute_n_window_containing_nodes(dst_in_past_windows, dst_in_current_windows, window_size)

    assert n_past_window_contain_current_src_dict[1] == 2
    assert n_past_window_contain_current_src_dict[2] == 2
    assert n_past_window_contain_current_dst_dict[202] == 2
    assert n_past_window_contain_current_dst_dict[303] == 1
    with pytest.raises(KeyError):
        print(n_past_window_contain_current_dst_dict[101])

def test_compute_iwf():
    edges = np.array([[2,101],
                      [1,101],
                      [2,202],
                      [1,303],
                      [2,202],
                      [2,303],
                      ])

    window_size = 2
    current_instances_idx = window_size
    assert window_size < edges.shape[0]


    src_in_past_windows = edges[:-window_size,0]
    dst_in_past_windows = edges[:-window_size,1]

    src_in_current_window = edges[-window_size:,0]
    dst_in_current_window = edges[-window_size:,1]

    edges_in_past_windows = edges[:-window_size]
    edges_in_current_window = edges[-window_size:]



    # ### test compute_as_nodes == True
    iwf = compute_iwf(src_in_past_windows, src_in_current_window, window_size)
    print(iwf)
    print(np.array(list(map(math.log,3/np.array([3])))))
    assert np.array_equal(iwf,  np.array(list(map(math.log,3/np.array([3])))))
    assert sum(np.where(iwf==float('inf'))[0]) == 0
    assert iwf.min() >= 0

    iwf = compute_iwf(dst_in_past_windows, dst_in_current_window, window_size)
    assert np.array_equal(iwf,  np.array(list(map(math.log,3/np.array([2,2])))))
    assert sum(np.where(iwf==float('inf'))[0]) == 0
    assert iwf.min() >= 0

    ### test compute_as_nodes == False
    iwf = compute_iwf(edges_in_past_windows, edges_in_current_window, window_size, compute_as_nodes=False)
    assert np.array_equal(iwf,  np.array(list(map(math.log,3/np.array([2,1])))))
    assert sum(np.where(iwf==float('inf'))[0]) == 0
    assert iwf.min() >= 0

def test_compute_xf_iwf():
    edges = np.array([[2,101],
                      [1,101],
                      [2,202],
                      [1,303],
                      [2,202],
                      [2,202],
                      ])

    window_size = 2
    current_instances_idx = window_size
    assert window_size < edges.shape[0]

    src_in_past_windows = edges[:-window_size,0]
    dst_in_past_windows = edges[:-window_size,1]

    src_in_current_window = edges[-window_size:,0]
    dst_in_current_window = edges[-window_size:,1]

    edges_in_past_windows = edges[:-window_size]
    edges_in_current_window = edges[-window_size:]


    ### test 1
    nf_iwf = compute_xf_iwf(src_in_past_windows, src_in_current_window, window_size)
    assert nf_iwf ==  1 + (np.array([2/2])) * (np.array(list(map(math.log,3/np.array([3])))))
    assert np.all(nf_iwf >= 1)

    ef_iwf = compute_xf_iwf(edges_in_past_windows, edges_in_current_window, window_size, compute_as_nodes=False)
    assert ef_iwf ==  1 + (np.array([2])/2) * (np.array(list(map(math.log,3/np.array([2])))))
    assert np.all(ef_iwf >= 1)

    ### test 2
    current_uniq_nodes, _, _ = get_uniq_x_freq_in_window(src_in_current_window, compute_as_nodes=True)
    nf_iwf, nf_iwf_window_dict = compute_xf_iwf(src_in_past_windows, src_in_current_window, window_size, return_x_value_dict=True)
    assert np.array_equal(np.array(list(nf_iwf_window_dict.keys())), current_uniq_nodes)

    current_uniq_edges, _, _ = get_uniq_x_freq_in_window(edges_in_current_window, compute_as_nodes=False)
    ef_iwf, ef_iwf_window_dict  = compute_xf_iwf(edges_in_past_windows, edges_in_current_window, window_size, compute_as_nodes=False, return_x_value_dict=True)
    assert np.array_equal(np.array(list(ef_iwf_window_dict.keys())), current_uniq_edges)


if __name__ == '__main__':
    pass
