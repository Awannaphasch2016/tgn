#!/usr/bin/env python3

from utils.utils import get_edges_dtype, get_different_edges_mask_left
import numpy as np
import torch
from random import choices
import random

class Sampler(object):
  def __init__(self, src_list, dst_list, edge_list, ref_window_size, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)
    self.edges =  np.vstack([src_list, dst_list]).T
    self.edge_list = edge_list
    self.ref_window_size = ref_window_size

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)

  def is_current_edges_idx_okay(self, current_edges_idx):
    return current_edges_idx < (self.edge_list.shape[0] - self.ref_window_size)

  # def sample(self, size):
  #   if self.seed is None:
  #     src_index = np.random.randint(0, len(self.src_list), size)
  #     dst_index = np.random.randint(0, len(self.dst_list), size)
  #   else:
  #     src_index = self.random_state.randint(0, len(self.src_list), size)
  #     dst_index = self.random_state.randint(0, len(self.dst_list), size)
  #   return self.src_list[src_index], self.dst_list[dst_index]

  def sample(self, size):
    return self.sample_v1(self.src_list, size)

  def sample_v1(self, src_list, size):
    """
    this sapmle_v1 method is created to extend sample method to accept unique set of src_list as arguments.
    Therefore, the function will sample nodes uniformly. (the original sample method sample nodes depends on nodes degree)
    """
    if self.seed is None:
      src_index = np.random.randint(0, len(src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:
      src_index = self.random_state.randint(0, len(src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

class RandEdgeSampler(Sampler):
  pass


class RandEdgeSampler_v1(Sampler):

  def get_easy_and_hard_negative_window_mask(self, current_edge_idx):
    """
    illustration scenario 1

    [window]
    [train window]
    [x| ->     |x]
    [       z  |x]
    [       y    ]

    where
    y = z + x
    y is train window
    x is hard negative sample window
    z contains all positive edges in whose negative sample will be sampled

    hard negative is an edge sampled from inside of x
    easy negative is an edge sampled from outside of x but inside of y.
    """

    assert self.is_current_edges_idx_okay(current_edge_idx)
    # NOTE: make sure that current_edge_idx is sampled from list of idx started from 0
    hard_negative_idx = [current_edge_idx+i for i in range(self.ref_window_size)]
    # assert len(self.edge_list.shape) == 1
    is_less_than_first_idx = self.edge_list < hard_negative_idx[0]
    is_more_than_first_idx = ~is_less_than_first_idx
    is_less_than_last_idx = self.edge_list <= hard_negative_idx[-1]
    is_more_than_last_idx = ~is_less_than_last_idx
    edge_hard_negatives_mask_idx = np.logical_and(is_more_than_first_idx, is_less_than_last_idx)
    edge_easy_negatives_mask_idx = ~edge_hard_negatives_mask_idx
    return edge_easy_negatives_mask_idx, edge_hard_negatives_mask_idx

  def get_positive_idx(self, current_edge_idx):
    return current_edge_idx

  def sample_hard_negative_idx(self, edges_mask_idx, n_hard_negative):
    """
    edges_mask_idx => idx of edges (within window length) to be sampled
    n_hard_negative => number of hard_negative to be sampled
    """

    mask_edges = self.edge_list[edges_mask_idx]
    random.seed(self.seed)
    return choices(mask_edges, k=n_hard_negative)

  def sample_easy_negative_idx(self, edges_mask, n_easy_negative):
    """
    edges_mask => mask of edges (within window length) to be sampled
    n_easy_negative => number of easy_negative to be sampled
    """
    easy_neg_edges = self.edges[edges_mask]
    hard_neg_edges = self.edges[~edges_mask]

    only_easy_edges_mask = get_different_edges_mask_left(easy_neg_edges, hard_neg_edges)

    easy_neg_edges_ind = np.array(range(easy_neg_edges.shape[0]))
    only_easy_edges_idx = easy_neg_edges_ind[only_easy_edges_mask]

    assert only_easy_edges_idx.shape[0] > 0

    random.seed(self.seed)
    return choices(only_easy_edges_idx, k=n_easy_negative)

class RandEdgeSampler_v2(Sampler):
  def get_pos_and_hard_negative_window_mask(self, current_edge_idx):
    """
    illustration scenario 2

    [window]
    [train window]
    [x| ->     |x]
    [       y    ]

    where
    y = z + x
    y is train window
    x is positive sample

    hard negative is an edge sampled from outside of x but inside of y.
    easy negative is an edge sampled from outside of y
    """
    assert self.is_current_edges_idx_okay(current_edge_idx)
    # NOTE: make sure that current_edge_idx is sampled from list of idx started from 0
    pos_idx = [current_edge_idx+i for i in range(self.ref_window_size)]
    # assert len(self.edge_list.shape) == 1
    is_less_than_first_idx = self.edge_list < pos_idx[0]
    is_more_than_first_idx = ~is_less_than_first_idx
    is_less_than_last_idx = self.edge_list <= pos_idx[-1]
    is_more_than_last_idx = ~is_less_than_last_idx
    edge_pos_mask_idx = np.logical_and(is_more_than_first_idx, is_less_than_last_idx)
    edge_hard_negatives_mask_idx = ~edge_pos_mask_idx
    return edge_pos_mask_idx, edge_hard_negatives_mask_idx

  def sample_hard_negative_idx(self, edges_mask, n_hard_negative):
    """
    edges_mask => mask of edges (within window length) to be sampled
    n_hard_negative => number of hard_negative to be sampled
    """
    hard_neg_edges = self.edges[edges_mask]
    pos_edges = self.edges[~edges_mask]

    only_hard_edges_mask = get_different_edges_mask_left(hard_neg_edges, pos_edges)

    hard_neg_edges_ind = np.array(range(hard_neg_edges.shape[0]))
    only_hard_edges_idx = hard_neg_edges_ind[only_hard_edges_mask]

    assert only_hard_edges_idx.shape[0] > 0

    random.seed(self.seed)
    return choices(only_hard_edges_idx, k=n_hard_negative)

  def sample_easy_negative_idx(self, uniq_src_list, n_easy_negative):
    """
    :NOTE: This is not a good implementation because sample function may output edges that are incidents to unique users inside of x and y winddow.
    """
    return self.sample_v1(uniq_src_list.tolist(), n_easy_negative)

  def get_positive_idx(self, current_edge_idx):
    pos_idx = np.array([current_edge_idx+i for i in range(self.ref_window_size)])
    return pos_idx


class RandEdgeSampler_v3(Sampler):
  """
  :NOTE: version 3 is created to be compatible with version 1.
  illustration

  [window]
  [train window]
  [       x    ]
  [       y    ]
  [       z    ]

  where
  train window x = y = z

  hard negative is an edge sampled from inside of x.
  easy negative is an edge sampled from outside of x.
  """

  # def get_easy_and_hard_negative_window_mask(self):
  #   edge_hard_negatives_mask_idx = np.array([True for i in range(self.edge_list.shape[0])])
  #   edge_easy_negatives_mask_idx = None
  #   return edge_easy_negatives_mask_idx, edge_hard_negatives_mask_idx

  def sample_hard_negative_idx(self, edges_mask_idx, n_hard_negative):
    """
    edges_mask_idx => idx of edges (within window length) to be sampled
    n_hard_negative => number of hard_negative to be sampled
    """
    raise NotImplementedError()

  def get_positive_idx(self, current_edge_idx):
    raise NotImplementedError()
