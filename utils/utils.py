import numpy as np
import torch
from random import choices
import random
import math

def sigmoid(x):
  return torch.nn.functional.sigmoid(torch.from_numpy(x)).cpu().detach().numpy()

def get_start_idx(current_window_idx, window_size):
  """
  assuming the following
  1. all idx starts from 0
  2. end_idx of previous window == start_idx + 1  of the current window.
  """
  return (current_window_idx * window_size)

def get_end_idx(current_window_idx, window_size):
  """
  assuming the following
  1. all idx starts from 0
  2. end_idx of previous window == start_idx + 1  of the current window.
  """
  start_idx = get_start_idx(current_window_idx, window_size)
  return start_idx + window_size

def compute_ef(edges_in_current_window, window_size):
  _, _, current_uniq_edges_freq =  get_uniq_edges_freq_in_window(edges_in_current_window)

  ef = current_uniq_edges_freq
  # ef = current_uniq_edges_freq/edges_in_current_window.shape[0]

  return ef


def compute_nf(nodes_in_current_window, window_size):
  current_src_uniq_nodes, _, current_src_uniq_nodes_freq =  get_uniq_nodes_freq_in_window(nodes_in_current_window)

  nf = current_src_uniq_nodes_freq
  # nf = current_src_uniq_nodes_freq/nodes_in_current_window.shape[0]

  return nf

def compute_n_window_containing_edges(edges_in_past_windows, edges_in_current_window, window_size):
# def compute_n_window_containing_edges(edges_in_past_windows, current_uniq_edges, window_size):
  current_uniq_edges,_, current_uniq_edges_freq = get_uniq_edges_freq_in_window(edges_in_current_window)

  n_past_windows = edges_in_past_windows.shape[0]/window_size

  assert (int(n_past_windows) - n_past_windows) == 0
  n_past_windows = int(n_past_windows)

  n_past_window_contain_current_dict = {tuple(ii.tolist()):1 for ii in current_uniq_edges}

  for i in range(n_past_windows):
    start_idx = get_start_idx(i, window_size)
    end_idx = get_end_idx(i, window_size)
    uniq_edges,_, uniq_edges_freq = get_uniq_edges_freq_in_window(edges_in_past_windows[start_idx:end_idx])

    for j in current_uniq_edges:
      if sum(get_different_edges_mask_left(j.reshape(-1,2),uniq_edges)) == 0: # all edges in the left is in the right.
        n_past_window_contain_current_dict[tuple(j.tolist())] += 1

  return  n_past_window_contain_current_dict

def compute_n_window_containing_nodes(nodes_in_past_windows, nodes_in_current_window, window_size):
  # nodes_in_all_windows = np.vstack((nodes_in_past_windows, nodes_in_current_window))

  current_src_uniq_nodes,_, current_src_uniq_nodes_freq = get_uniq_nodes_freq_in_window(nodes_in_current_window)

  # current_src_uniq_nodes,_, current_src_uniq_nodes_freq = get_uniq_nodes_freq_in_window(nodes_in_all_windows)

  n_past_windows = nodes_in_past_windows.shape[0]/window_size

  assert (int(n_past_windows) - n_past_windows) == 0
  n_past_windows = int(n_past_windows)

  n_past_window_contain_current_src_dict = {ii:1 for ii in current_src_uniq_nodes}

  for i in range(n_past_windows):
    start_idx = get_start_idx(i, window_size)
    end_idx = get_end_idx(i, window_size)
    src_uniq_nodes,_, src_uniq_nodes_freq = get_uniq_nodes_freq_in_window(nodes_in_past_windows[start_idx:end_idx])

    # src_nf = [ii for ii in src_uniq_nodes]
    

    for j in current_src_uniq_nodes:
      # if j in src_nf:
      if j in src_uniq_nodes:
        n_past_window_contain_current_src_dict[j] += 1

  return  n_past_window_contain_current_src_dict

def convert_dict_values_to_np(a_dict):
  return np.array([ii for ii in a_dict.values()])

def compute_iwf(x_in_past_windows, x_in_current_window, window_size, compute_as_nodes=True):
  # assert x_in_past_windows.shape[0] % window_size == 0
  # assert x_in_current_windows.shape[0] % window_size == 0

  n_past_windows = x_in_past_windows.shape[0]/window_size
  n_current_window = x_in_current_window.shape[0]/window_size
  n_all_windows = n_past_windows + n_current_window

  if compute_as_nodes:
    # n_past_windows = x_in_past_windows.shape[0]/window_size
    # n_current_window = x_in_current_window.shape[0]/window_size

    n_all_window_contain_current_x_dict = compute_n_window_containing_nodes(x_in_past_windows, x_in_current_window, window_size)

    n_all_window_contain_current_x = convert_dict_values_to_np(n_all_window_contain_current_x_dict)

  else:
    assert len(x_in_past_windows.shape) == 2
    assert len(x_in_current_window.shape) == 2
    assert x_in_past_windows.shape[1] == 2
    assert x_in_current_window.shape[1] == 2

    # n_past_windows = x_in_past_windows.shape[0]/window_size
    # n_current_window = x_in_current_window.shape[0]/window_size


    n_all_window_contain_current_x_dict = compute_n_window_containing_edges(x_in_past_windows, x_in_current_window, window_size)

    n_all_window_contain_current_x = convert_dict_values_to_np(n_all_window_contain_current_x_dict)


  wf = n_all_windows # number of document that term appears.
  iwf = np.array(list(map(math.log, wf/n_all_window_contain_current_x)))

  iwf_mask = np.where(n_all_window_contain_current_x==0)[0]
  iwf[iwf_mask] = 999999 # replace inf value with very large number.

  # :NOTE: apply sigmoid function to set range of iwf to be [0,1]
  # if compute_ef_iwf_with_sigmoid:
  #   iwf = torch.nn.functional.sigmoid(torch.from_numpy(iwf)).cpu().detach().numpy()

  return iwf


def get_uniq_x_freq_in_window(x_in_current_window, compute_as_nodes):

  if compute_as_nodes:
    return get_uniq_nodes_freq_in_window(x_in_current_window)
  else:
    return get_uniq_edges_freq_in_window(x_in_current_window)

def compute_xf_iwf(x_in_past_windows, x_in_current_window, window_size, compute_as_nodes=True, return_x_value_dict=False, compute_with_sigmoid=False):

  current_uniq_x, uniq_x_idx, current_uniq_x_freq = get_uniq_x_freq_in_window(x_in_current_window, compute_as_nodes)

  if compute_as_nodes:
    xf = compute_nf(x_in_current_window, window_size)
  else:
    xf = compute_ef(x_in_current_window, window_size)

  iwf = compute_iwf(x_in_past_windows, x_in_current_window, window_size, compute_as_nodes=compute_as_nodes)
  xf_iwf =  (xf * iwf)

  if compute_with_sigmoid:
    xf_iwf = torch.nn.functional.sigmoid(torch.from_numpy(xf_iwf)).cpu().detach().numpy()

  xf_iwf += 1 # garantee iwf value to always be > 1

  assert iwf.shape[0] == xf.shape[0]
  assert xf_iwf.shape[0] == iwf.shape[0]
  assert len(current_uniq_x) == iwf.shape[0]
  assert len(current_uniq_x) == len(uniq_x_idx)

  # print(current_uniq_x, xf_iwf)
  if compute_as_nodes:
    x_to_xf_iwf_window_dict = {i:j for i,j in zip(current_uniq_x, xf_iwf)}
  else:
    x_to_xf_iwf_window_dict = {tuple(i):j for i,j in zip(current_uniq_x, xf_iwf)}

  if return_x_value_dict:
    return xf_iwf, x_to_xf_iwf_window_dict
  else:
    return xf_iwf

def get_uniq_edges_freq_in_window(edges_in_current_window):
  assert len(edges_in_current_window.shape) == 2
  uniq_edges, uniq_edges_idx, uniq_edges_freq = np.unique(edges_in_current_window, return_counts=True, return_index=True, axis=0)
  assert uniq_edges.shape[1] == 2
  return (uniq_edges, uniq_edges_idx,uniq_edges_freq)

def get_uniq_nodes_freq_in_window(nodes_in_current_window):
  assert len(nodes_in_current_window.shape) == 1
  uniq_nodes, uniq_nodes_idx, uniq_nodes_freq = np.unique(nodes_in_current_window, return_counts=True, return_index=True)
  return (uniq_nodes, uniq_nodes_idx,uniq_nodes_freq)

def get_edges_dtype(edges):
  """
  https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
  """

  # assert not (edges.flags['C_CONTIGUOUS'] and edges.flags['F_CONTIGUOUS'])
  assert len(edges.shape) == 2
  assert edges.shape[-1] == 2

  nrows, ncols = edges.shape

  if edges.flags['C_CONTIGUOUS']:
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
          'formats':ncols * [edges.dtype]}
  elif edges.flags['F_CONTIGUOUS']:
    raise NotImplementedError()
  else:
    raise NotImplementedError()

  return dtype


def get_different_edges_mask_left(left_edges, right_edges):

  left_edges = np.ascontiguousarray(left_edges)
  right_edges = np.ascontiguousarray(right_edges)

  left_dtype = get_edges_dtype(left_edges)
  right_dtype = get_edges_dtype(right_edges)

  only_left_edges_mask = np.isin(left_edges.view(left_dtype).reshape(-1), right_edges.view(right_dtype).reshape(-1), invert=True).reshape(-1)

  return only_left_edges_mask

def pred_prob_to_pred_labels(pred_prob, selected_ind=None):

  if selected_ind is not None:
    pred_prob = pred_prob[selected_ind]
  if pred_prob.reshape(-1).shape[0] == pred_prob.shape[0]:
    raise NotImplementedError
    pred = pred_prob > 0.5
  else:
    pred = pred_prob.argmax(axis=1)
  return pred

def get_unique_nodes_labels(labels, nodes):
  _, unique_nodes_ind = np.unique(nodes, return_index=True)
  return labels[unique_nodes_ind]

def get_label_distribution(labels):
  u, c = np.unique(labels, return_counts=True)
  uc = np.vstack((u, c))
  uc_str = []
  for uu in range(uc.shape[1]):
    tmp = tuple([cc for cc in uc[:, uu]])
    uc_str.append(tmp)
  return uc_str

def find_nodes_ind_to_be_labelled(selected_nodes_to_label, target_nodes_batch):
  selected_nodes_ind = []
  for ll in selected_nodes_to_label:
    selected_nodes_ind.extend(np.where(target_nodes_batch == ll)[0].tolist())
  return selected_nodes_ind

# def label_new_unique_nodes_with_budget(selected_sources_to_label, data, sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, labels_batch):
def label_new_unique_nodes_with_budget(selected_sources_to_label, data, begin_end_idx_pair=None):
  assert len(begin_end_idx_pair) == 2

  begin_idx, end_idx = begin_end_idx_pair
  full_data = data

  sources_batch = full_data.sources[begin_idx:end_idx]
  # destinations_batch = full_data.destinations[begin_idx:end_idx]
  # timestamps_batch =  full_data.timestamps[begin_idx:end_idx]
  # edge_idxs_batch = full_data.edge_idxs[begin_idx:end_idx]
  # labels_batch = full_data.labels[begin_idx:end_idx]

  # select sources nodes to be labelled.
  # :BUG: see the following link for full explaination of potential problem.  https://mail.google.com/mail/u/1/#sent/QgrcJHsNlSQcfgjngKvJvfWsltLMshplFxg
  # :BUG: https://roamresearch.com/#/app/AdaptiveGraphStucture/page/uIwdA9uav
  # :DEBUG:
  unique_sources = np.unique(sources_batch)
  n_unique_sources = unique_sources.shape[0]
  n_selected_sources = int(full_data.budget * n_unique_sources)
  total_selected_sources = min(len(selected_sources_to_label) + n_selected_sources, full_data.label_budget)
  n_unique_sources_to_add = total_selected_sources - len(selected_sources_to_label)
  assert n_unique_sources_to_add >= 0
  assert n_unique_sources_to_add < data.n_unique_sources
  # create mask for new batch
  if n_unique_sources_to_add > 0:
    selected_sources_to_label_batch_mask = np.array([False for i in range(sources_batch.shape[0])])
    if len(selected_sources_to_label) > 0:
      selected_sources_to_label_batch_mask = np.array(list(map(lambda x: x in selected_sources_to_label, sources_batch)))

    new_sources = sources_batch[~selected_sources_to_label_batch_mask]
    # after mask is applied, random pick new unique node.
    if new_sources.shape[0] > 0:
      unique_sources_ind = np.unique(new_sources, return_index=True)[1]
      unique_sources_ind_to_add = choices(unique_sources_ind, k=n_unique_sources_to_add)
      selected_sources_to_label.extend(sources_batch[unique_sources_ind_to_add])
  assert len(selected_sources_to_label) > 0

  # label nodes that are in sources_batch.
  selected_sources_ind = find_nodes_ind_to_be_labelled(selected_sources_to_label, sources_batch)


  return selected_sources_ind, selected_sources_to_label


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP_multiple_class(torch.nn.Module):
  def __init__(self, dim, n_labels, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, n_labels)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)
    self.n_labels  = n_labels

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    out = self.fc_3(x)
    assert out.shape[1] == self.n_labels
    return out


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)

class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]

  # print(max_node_idx)
  # print(data.destinations.shape)
  # print(data.sources.shape)
  # print(data.edge_idxs.shape)
  # print(data.timestamps.shape)

  for i, (source, destination, edge_idx, timestamp) in enumerate(zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps)):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  # print(len(adj_list))
  # print('------')

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall
    interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """

    # print(len(self.node_to_edge_timestamps))
    # print(src_idx)
    # print(self.node_to_edge_timestamps[src_idx].shape)
    # print(self.node_to_edge_timestamps[src_idx])
    # print(cut_time)

    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    # print('===')
    # exit()

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    # print(source_nodes)

    # print(max(source_nodes)) # 112
    # print(len(self.node_to_neighbors))
    # print('hi')

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times
