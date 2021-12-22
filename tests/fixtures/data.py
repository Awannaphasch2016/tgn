#!/usr/bin/env python3

from utils.data_processing import Data
import numpy as np
import random
import pytest

@pytest.fixture
def seed():
    return 3

@pytest.fixture
def current_edge_idx():
    return 2

@pytest.fixture
def ref_window_size():
    return 3

@pytest.fixture
def n_hard_negative():
    return 2
@pytest.fixture
def n_easy_negative(n_hard_negative):
    return n_hard_negative

@pytest.fixture
def sources():
    sources = [1,2,3,4,5]
    return sources

@pytest.fixture
def destination():
    destination = [101,202,303,404,505]
    return destination

@pytest.fixture
def edges(sources ,destination):
    # edges = np.array([random.choices(sources, k=11), random.choices(destination,k=11)]).T
    # assert edges.shape[1] == 2
    # return edges

    user_edges = []
    for i in sources:
        for j in random.choices(destination, k=3):
            user_edges.append((i, j))

    edges = np.array(user_edges)
    assert edges.shape[1] == 2
    return edges

@pytest.fixture
def data(edges):
    sources = edges[:,0]
    destinations = edges[:,1]
    timestamps = np.array(range(edges.shape[0]))
    edge_idxs = np.array(range(edges.shape[0]))
    labels = np.array(random.choices(sources, k=edges.shape[0])) # don't need it for testing yet.

    data = Data(sources, destinations, timestamps, edge_idxs, labels)
    return data
