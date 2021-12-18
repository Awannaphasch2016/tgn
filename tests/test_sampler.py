#!/usr/bin/env python3
from utils.utils import get_different_edges_mask_left
from utils.sampler import RandEdgeSampler, RandEdgeSampler_v1, RandEdgeSampler_v2, RandEdgeSampler_v3

import numpy as np
import pytest


@pytest.mark.usefixtures("data", "seed")
def test_RandomEdgeSampler_sample(mocker, data, seed):
    size = 10 # randomly chosen
    sources = data.sources
    destination = data.destinations

    train_rand_sampler = RandEdgeSampler(sources, destination, None, None,seed=seed)
    src_negatives_batch , des_negatives_batch = train_rand_sampler.sample(size)

    assert len(src_negatives_batch) == len(des_negatives_batch)
    assert len(src_negatives_batch) == size

    for i,j in zip(src_negatives_batch, des_negatives_batch):
        assert  i in sources and i not in destination
        assert  j in destination and j not in sources


@pytest.mark.usefixtures("data", "seed", "current_edge_idx")
def test_RandomEdgeSampler_get_easy_and_hard_negative_window_mask(data, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations

    train_rand_sampler = RandEdgeSampler(sources, destination, None, None,seed=seed)
    train_rand_sampler.sample_positive_idx(current_edge_idx)

@pytest.mark.usefixtures("data", "edges", "hard_negative_window_size", "n_hard_negative", "seed", "current_edge_idx")
def test_RandomEdgeSampler_get_easy_and_hard_negative_window_mask(mocker, data, edges, hard_negative_window_size, n_hard_negative, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations
    edges_idx = data.edge_idxs

    train_rand_sampler = RandEdgeSampler_v1(sources, destination, edges_idx, hard_negative_window_size, seed=seed)

    edges_easy_negatives_mask_idx, edges_hard_negatives_mask_idx = train_rand_sampler.get_easy_and_hard_negative_window_mask(current_edge_idx)

    assert sum(edges_hard_negatives_mask_idx) + sum(edges_easy_negatives_mask_idx) == train_rand_sampler.edge_list.shape[0]
    assert sum(edges_hard_negatives_mask_idx) == train_rand_sampler.hard_negative_window_size
    assert sum(edges_easy_negatives_mask_idx) == train_rand_sampler.edge_list.shape[0] - train_rand_sampler.hard_negative_window_size


@pytest.mark.usefixtures("data", "edges", "hard_negative_window_size", "n_hard_negative", "seed", "current_edge_idx")
def test_RandomEdgeSampler_sample_hard_negative_idx(mocker, data, edges, hard_negative_window_size, n_hard_negative, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations
    edges_idx = data.edge_idxs

    train_rand_sampler = RandEdgeSampler_v1(sources, destination, edges_idx, hard_negative_window_size, seed=seed)

    _, edges_hard_negatives_mask_idx = train_rand_sampler.get_easy_and_hard_negative_window_mask(current_edge_idx)

    edges_hard_negatives_idx_batch = train_rand_sampler.sample_hard_negative_idx(edges_hard_negatives_mask_idx, n_hard_negative)

    assert len(edges_hard_negatives_idx_batch) == n_hard_negative

    edges_hard_negatives_batch = edges[edges_hard_negatives_idx_batch]
    mask_edges = edges[edges_hard_negatives_mask_idx, :]

    assert sum(get_different_edges_mask_left(edges_hard_negatives_batch, mask_edges)) == 0

@pytest.mark.usefixtures("data", "edges", "hard_negative_window_size", "n_easy_negative", "seed", "current_edge_idx")
def test_RandomEdgeSampler_sample_easy_negative(mocker, data, edges, hard_negative_window_size, n_easy_negative, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations
    edges_idx = data.edge_idxs

    train_rand_sampler = RandEdgeSampler_v1(sources, destination, edges_idx, hard_negative_window_size, seed=seed)

    edges_easy_negatives_mask_idx, edges_hard_negatives_mask_idx = train_rand_sampler.get_easy_and_hard_negative_window_mask(current_edge_idx)

    edges_easy_negatives_idx_batch = train_rand_sampler.sample_easy_negative_idx(edges_easy_negatives_mask_idx, n_easy_negative)

    assert len(edges_easy_negatives_idx_batch) == n_easy_negative

    easy_mask_edges = edges[edges_easy_negatives_mask_idx, :]
    hard_mask_edges = edges[edges_hard_negatives_mask_idx, :]

    edges_easy_negatives_batch = easy_mask_edges[edges_easy_negatives_idx_batch]

    print(edges_easy_negatives_batch)
    print(hard_mask_edges)

    assert sum(get_different_edges_mask_left(edges_easy_negatives_batch, easy_mask_edges)) == 0
    assert sum(get_different_edges_mask_left(edges_easy_negatives_batch, hard_mask_edges)) == edges_easy_negatives_batch.shape[0]

@pytest.mark.usefixtures("data", "edges", "hard_negative_window_size", "n_easy_negative", "n_hard_negative","seed", "current_edge_idx")
def test_RandomEdgeSampler_sample_easy_negative_not_in_hard_negative(mocker, data, edges, hard_negative_window_size, n_easy_negative, n_hard_negative, seed, current_edge_idx):
    sources = data.sources
    destination = data.destinations
    edges_idx = data.edge_idxs

    train_rand_sampler = RandEdgeSampler_v1(sources, destination, edges_idx, hard_negative_window_size, seed=seed)

    edges_easy_negatives_mask_idx, edges_hard_negatives_mask_idx = train_rand_sampler.get_easy_and_hard_negative_window_mask(current_edge_idx)

    edges_easy_negatives_idx_batch = train_rand_sampler.sample_easy_negative_idx(edges_easy_negatives_mask_idx, n_easy_negative)

    edges_hard_negatives_idx_batch = train_rand_sampler.sample_hard_negative_idx(edges_hard_negatives_mask_idx, n_hard_negative)

    assert len(edges_easy_negatives_idx_batch) == n_easy_negative
    assert len(edges_hard_negatives_idx_batch) == n_hard_negative

    easy_mask_edges = edges[edges_easy_negatives_mask_idx, :]

    edges_easy_negatives_batch = easy_mask_edges[edges_easy_negatives_idx_batch]
    edges_hard_negatives_batch = edges[edges_hard_negatives_idx_batch]

    assert sum(get_different_edges_mask_left(edges_easy_negatives_batch, edges_hard_negatives_batch)) == edges_easy_negatives_batch.shape[0]

@pytest.mark.usefixtures("data", "edges", "hard_negative_window_size", "n_hard_negative", "seed")
def test_RandomEdgeSampler_v3_get_easy_and_hard_negative_window_mask(mocker, data, edges, hard_negative_window_size, n_hard_negative, seed):

    sources = data.sources
    destination = data.destinations
    edges_idx = data.edge_idxs

    train_rand_sampler = RandEdgeSampler_v3(sources, destination, edges_idx, hard_negative_window_size, seed=seed)

    edges_easy_negatives_mask_idx, edges_hard_negatives_mask_idx = train_rand_sampler.get_easy_and_hard_negative_window_mask()

    assert edges_hard_negatives_mask_idx.shape[0] == train_rand_sampler.edge_list.shape[0]
    assert edges_hard_negatives_mask_idx.shape[0] == sum(edges_hard_negatives_mask_idx)
    assert edges_easy_negatives_mask_idx is None
