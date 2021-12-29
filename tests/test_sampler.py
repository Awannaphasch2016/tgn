#!/usr/bin/env python3
from utils.utils import get_different_edges_mask_left
from utils.sampler import RandEdgeSampler, RandEdgeSampler_v1, RandEdgeSampler_v2, RandEdgeSampler_v3, EdgeSampler_NF_IWF

import numpy as np
import pytest
from train_self_supervised import args
from utils.data_processing import get_data


@pytest.mark.usefixtures("data", "seed")
def test_RandomEdgeSampler_sample(mocker, data, seed):
    size = 10 # randomly chosen
    sources = data.sources
    destination = data.destinations

    train_rand_sampler = RandEdgeSampler(sources, destination, None, None,None, None,seed=seed)
    src_negatives_batch , des_negatives_batch = train_rand_sampler.sample(size)

    assert len(src_negatives_batch) == len(des_negatives_batch)
    assert len(src_negatives_batch) == size

    for i,j in zip(src_negatives_batch, des_negatives_batch):
        assert  i in sources and i not in destination
        assert  j in destination and j not in sources

@pytest.mark.usefixtures("data", "seed", "fixed_edges")
def test_EdgeSampler_rank_unique_node_in_window_based_on_nf_iwf(mocker, data, seed, fixed_edges):

    edges = fixed_edges
    size = 10 # randomly chosen
    sources = data.sources
    destination = data.destinations

    ### test 1
    start_idx = 10
    end_idx = 15

    train_rand_sampler = EdgeSampler_NF_IWF(sources, destination, edges, None, start_idx, end_idx,seed=seed)
    ranked_current_node_uniq_nodes, ranked_uniq_node_nf_in_current_window, _ = train_rand_sampler.rank_unique_node_in_window_based_on_nf_iwf(edges[:,0], start_idx, end_idx)

    assert np.array_equal(ranked_current_node_uniq_nodes, np.array([5]))
    assert np.array_equal(ranked_uniq_node_nf_in_current_window, np.array([5/15]))
    assert np.array_equal(ranked_current_node_uniq_nodes, np.array([5]))

    #### test 2
    start_idx = 0
    end_idx = 5

    train_rand_sampler = EdgeSampler_NF_IWF(sources, destination, edges, None, start_idx, end_idx,seed=seed)
    ranked_current_node_uniq_nodes, ranked_uniq_node_nf_in_current_window, _ = train_rand_sampler.rank_unique_node_in_window_based_on_nf_iwf(edges[:,0], start_idx, end_idx)

    assert np.array_equal(ranked_current_node_uniq_nodes, np.array([3,2,1]))
    assert np.array_equal(ranked_uniq_node_nf_in_current_window, np.array([3,2,1])/15)
    assert np.array_equal(ranked_current_node_uniq_nodes, np.array([3,2,1]))

@pytest.mark.usefixtures("data", "seed", "fixed_edges")
def test_EdgeSampler_sample_nf_iwf(mocker, data, seed, fixed_edges):
    edges = fixed_edges

    size = 10 # randomly chosen
    sources = data.sources
    destination = data.destinations

    ### test 1
    start_idx = 10
    end_idx = 15
    batch_size = end_idx - start_idx

    train_rand_sampler = EdgeSampler_NF_IWF(sources, destination, edges, None, start_idx, end_idx,seed=seed)

    src_batch, dst_batch = train_rand_sampler.sample_nf_iwf(batch_size, size)

    assert src_batch.shape[0] == dst_batch.shape[0]
    assert src_batch.shape[0] == size
    assert np.array_equal(src_batch, np.array([5 for i in range(size)]))

    #### test 2
    start_idx = 0
    end_idx = 5

    train_rand_sampler = EdgeSampler_NF_IWF(sources, destination, edges, None, start_idx, end_idx,seed=seed)
    src_batch, dst_batch = train_rand_sampler.sample_nf_iwf(batch_size, size)

    assert src_batch.shape[0] == dst_batch.shape[0]
    assert src_batch.shape[0] == size
    for i in src_batch:
        assert i in [1,2,3]

@pytest.mark.usefixtures("data", "seed", "current_edge_idx")
def test_RandomEdgeSampler_v1_get_positive_idx(data, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations

    train_rand_sampler = RandEdgeSampler_v1(sources, destination, None, None, None, None, seed=seed)
    pos_edge = train_rand_sampler.get_positive_idx(current_edge_idx)
    assert isinstance(pos_edge, int)

@pytest.mark.usefixtures("data", "edges", "ref_window_size", "n_hard_negative", "seed", "current_edge_idx")
def test_RandomEdgeSampler_v1_get_easy_and_hard_negative_window_mask(mocker, data, edges, ref_window_size, n_hard_negative, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations
    edges_idx = data.edge_idxs

    train_rand_sampler = RandEdgeSampler_v1(sources, destination, edges_idx, ref_window_size, None, None, seed=seed)

    edges_easy_negatives_mask_idx, edges_hard_negatives_mask_idx = train_rand_sampler.get_easy_and_hard_negative_window_mask(current_edge_idx)

    assert sum(edges_hard_negatives_mask_idx) + sum(edges_easy_negatives_mask_idx) == train_rand_sampler.edge_list.shape[0]
    assert sum(edges_hard_negatives_mask_idx) == train_rand_sampler.ref_window_size
    assert sum(edges_easy_negatives_mask_idx) == train_rand_sampler.edge_list.shape[0] - train_rand_sampler.ref_window_size


@pytest.mark.usefixtures("data", "edges", "ref_window_size", "n_hard_negative", "seed", "current_edge_idx")
def test_RandomEdgeSampler_v1_sample_hard_negative_idx(mocker, data, edges, ref_window_size, n_hard_negative, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations
    edges_idx = data.edge_idxs

    train_rand_sampler = RandEdgeSampler_v1(sources, destination, edges_idx, ref_window_size, None, None, seed=seed)

    _, edges_hard_negatives_mask_idx = train_rand_sampler.get_easy_and_hard_negative_window_mask(current_edge_idx)

    edges_hard_negatives_idx_batch = train_rand_sampler.sample_hard_negative_idx(edges_hard_negatives_mask_idx, n_hard_negative)

    assert len(edges_hard_negatives_idx_batch) == n_hard_negative

    edges_hard_negatives_batch = edges[edges_hard_negatives_idx_batch]
    mask_edges = edges[edges_hard_negatives_mask_idx, :]

    assert sum(get_different_edges_mask_left(edges_hard_negatives_batch, mask_edges)) == 0

@pytest.mark.usefixtures("data", "edges", "ref_window_size", "n_easy_negative", "seed", "current_edge_idx")
def test_RandomEdgeSampler_v1_sample_easy_negative(mocker, data, edges, ref_window_size, n_easy_negative, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations
    edges_idx = data.edge_idxs

    train_rand_sampler = RandEdgeSampler_v1(sources, destination, edges_idx, ref_window_size, None, None, seed=seed)

    edges_easy_negatives_mask_idx, edges_hard_negatives_mask_idx = train_rand_sampler.get_easy_and_hard_negative_window_mask(current_edge_idx)

    edges_easy_negatives_idx_batch = train_rand_sampler.sample_easy_negative_idx(edges_easy_negatives_mask_idx, n_easy_negative)

    assert len(edges_easy_negatives_idx_batch) == n_easy_negative

    easy_mask_edges = edges[edges_easy_negatives_mask_idx, :]
    hard_mask_edges = edges[edges_hard_negatives_mask_idx, :]

    edges_easy_negatives_batch = easy_mask_edges[edges_easy_negatives_idx_batch]

    # assert sum(get_different_edges_mask_left(edges_easy_negatives_batch, easy_mask_edges)) == 0
    assert sum(get_different_edges_mask_left(edges_easy_negatives_batch, hard_mask_edges)) == edges_easy_negatives_batch.shape[0]

@pytest.mark.usefixtures("data", "edges", "ref_window_size", "n_easy_negative", "n_hard_negative","seed", "current_edge_idx")
def test_RandomEdgeSampler_v1_sample_easy_negative_not_in_hard_negative(mocker, data, edges, ref_window_size, n_easy_negative, n_hard_negative, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations
    edges_idx = data.edge_idxs

    train_rand_sampler = RandEdgeSampler_v1(sources, destination, edges_idx, ref_window_size, None, None, seed=seed)

    edges_easy_negatives_mask_idx, edges_hard_negatives_mask_idx = train_rand_sampler.get_easy_and_hard_negative_window_mask(current_edge_idx)

    edges_easy_negatives_idx_batch = train_rand_sampler.sample_easy_negative_idx(edges_easy_negatives_mask_idx, n_easy_negative)

    edges_hard_negatives_idx_batch = train_rand_sampler.sample_hard_negative_idx(edges_hard_negatives_mask_idx, n_hard_negative)

    assert len(edges_easy_negatives_idx_batch) == n_easy_negative
    assert len(edges_hard_negatives_idx_batch) == n_hard_negative

    easy_mask_edges = edges[edges_easy_negatives_mask_idx, :]

    edges_easy_negatives_batch = easy_mask_edges[edges_easy_negatives_idx_batch]
    edges_hard_negatives_batch = edges[edges_hard_negatives_idx_batch]

    assert sum(get_different_edges_mask_left(edges_easy_negatives_batch, edges_hard_negatives_batch)) == edges_easy_negatives_batch.shape[0]

@pytest.mark.usefixtures("data", "edges", "ref_window_size", "n_hard_negative", "seed", "current_edge_idx")
def test_RandomEdgeSampler_v2_get_pos_and_hard_negative_window_mask(mocker, data, edges, ref_window_size, n_hard_negative, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations
    edges_idx = data.edge_idxs

    train_rand_sampler = RandEdgeSampler_v2(sources, destination, edges_idx, ref_window_size, None, None, seed=seed)

    edges_pos_mask_idx, edges_hard_negatives_mask_idx = train_rand_sampler.get_pos_and_hard_negative_window_mask(current_edge_idx)

    assert sum(edges_hard_negatives_mask_idx) + sum(edges_pos_mask_idx) == train_rand_sampler.edge_list.shape[0]
    assert sum(edges_hard_negatives_mask_idx) == train_rand_sampler.edge_list.shape[0] - train_rand_sampler.ref_window_size
    assert sum(edges_pos_mask_idx) == train_rand_sampler.ref_window_size

@pytest.mark.usefixtures("data", "edges", "ref_window_size", "n_hard_negative", "seed", "current_edge_idx")
def test_RandomEdgeSampler_v2_sample_hard_negative_idx(data, edges, ref_window_size, n_hard_negative, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations
    edges_idx = data.edge_idxs

    train_rand_sampler = RandEdgeSampler_v2(sources, destination, edges_idx, ref_window_size, None, None, seed=seed)

    edges_pos_mask_idx, edges_hard_negatives_mask_idx = train_rand_sampler.get_pos_and_hard_negative_window_mask(current_edge_idx)

    edges_hard_negatives_idx_batch = train_rand_sampler.sample_hard_negative_idx(edges_hard_negatives_mask_idx, n_hard_negative)

    assert len(edges_hard_negatives_idx_batch) == n_hard_negative

    pos_mask_edges = edges[edges_pos_mask_idx, :]
    hard_mask_edges = edges[edges_hard_negatives_mask_idx, :]

    edges_hard_negatives_batch = hard_mask_edges[edges_hard_negatives_idx_batch]

    # assert sum(get_different_edges_mask_left(edges_hard_negatives_batch, hard_mask_edges)) == 0
    assert sum(get_different_edges_mask_left(edges_hard_negatives_batch, pos_mask_edges)) == edges_hard_negatives_batch.shape[0]

@pytest.mark.usefixtures("data", "seed", "current_edge_idx")
def test_RandomEdgeSampler_v2_get_positive_idx( data, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations

    train_rand_sampler = RandEdgeSampler_v2(sources, destination, None, current_edge_idx, None, None, seed=seed)
    pos_idx = train_rand_sampler.get_positive_idx(current_edge_idx)
    assert pos_idx.shape[0] == train_rand_sampler.ref_window_size
    assert pos_idx[0] == current_edge_idx
    assert pos_idx[-1] == current_edge_idx+train_rand_sampler.ref_window_size -1

# @pytest.mark.usefixtures("data", "seed", "edges", "current_edge_idx", "n_easy_negative", "ref_window_size")
# def test_RandomSampler_v2_sample_easy_negative_idx(data, seed, edges, current_edge_idx, n_easy_negative, ref_window_size):

#     args.data = 'reddit_10000'

#     node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data,  new_node_test_data, timestamps, observed_edges_mask = get_data(args.data,
#                                 different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

#     data = full_data
#     sources = data.sources
#     destination = data.destinations
#     edges_idx = data.edge_idxs
#     edges = np.vstack((sources, destination)).T

#     train_rand_sampler = RandEdgeSampler_v2(sources, destination, edges_idx, ref_window_size, None, None, seed=seed)

#     edges_pos_mask_idx, edges_hard_negatives_mask_idx = train_rand_sampler.get_pos_and_hard_negative_window_mask(current_edge_idx)

#     uniq_src_idx = np.unique(edges[edges_pos_mask_idx])
#     src_list, dst_list = train_rand_sampler.sample_easy_negative_idx(uniq_src_idx, n_easy_negative)

#     easy_edges_list = np.vstack((src_list, dst_list))


#     only_pos = get_different_edges_mask_left(easy_edges_list, edges[edges_pos_mask_idx,:])

#     assert sum(only_pos) == only_pos.shape[0]
