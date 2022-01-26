import numpy as np
import torch
from random import choices
import random
import math
from  sklearn import preprocessing
import pandas as pd
import logging
from pathlib import Path

def get_selected_sources_of_batch_idx_relative_to_window_idx(absolute_batch_idx, ws_multiplier):
  raise NotImplementedError()
  relative_batch_idx = get_batch_idx_relative_to_window_idx(absolute_batch_idx, ws_multiplier)
  # absolute_window_idx = convert_batch_idx_to_window_idx(absolute_batch_idx, ws_multiplier)

  relative_batch_idx

def get_batch_idx_relative_to_window_idx(absolute_batch_idx, ws_multiplier):
  relative_batch_idx = absolute_batch_idx % ws_multiplier
  return relative_batch_idx # relative ind start at 0

def convert_prob_list_to_binary_list(prob_list):
  tmp = []
  for i in prob_list:
    tmp.append(convert_prob_to_binary(i))
  return tmp

def convert_prob_to_binary(prob):
  """
  pos = 1  and neg = 1
  """
  val = 1 if prob > 0.5 else 0
  return val

def split_a_window_into_batches(window_begin_ind, window_end_ind, batch_size):
  n_instances = window_end_ind - window_begin_ind + 1
  assert n_instances/batch_size == int(n_instances/batch_size)
  n_batches = int(n_instances/batch_size)

  min_ind = min(window_begin_ind, window_end_ind)
  end_batch_inds = [min_ind]
  for i in range(n_batches):
    end_batch_inds.append(((i+1) * batch_size) + min_ind)
  return end_batch_inds

def right_split_ind_by_window(n_window_to_have_on_right, n_instances, window_size):
  assert n_instances/window_size == int(n_instances/window_size)

  n_windows = int(n_instances/window_size)
  begin_inds = [i * window_size for i in range(n_windows)]
  end_inds = [i + window_size for i in begin_inds]

  begin_ind = 0
  ind_to_split = begin_inds[-n_window_to_have_on_right]
  end_ind = end_inds[-n_window_to_have_on_right]

  return begin_ind, ind_to_split, end_ind
  # return (begin_inds[:-1], end_inds[:-1]), (begin_inds[-1], end_inds[-1])

def convert_ensemble_idx_to_window_idx(ensemble_idx, window_size):
  return (ensemble_idx * window_size) + window_size

def convert_batch_idx_to_window_idx(batch_idx, ws_multiplier):
  """return absolute_window_idx that batch_idx is in"""
  return int(batch_idx/ws_multiplier)

def convert_window_idx_to_batch_idx(window_idx, ws_multiplier):
  return window_idx * ws_multiplier

def convert_ind_to_n_instances(ind):
  return ind + 1
def convert_n_instances_to_ind(n_instances):
  return n_instances - 1


def apply_off_set_ind(ind_list, ind_shift_val):
  tmp = []
  for i in ind_list:
    shifted_ind = i - ind_shift_val
    if shifted_ind >= 0:
      tmp.append(shifted_ind)
  return tmp

def get_conditions(args):
  if args.use_ef_iwf_weight:
    assert args.max_random_weight_range is None
    prefix = 'use_ef_iwf_weight'
    neg_sample_method = "random"
    neg_edges_formation = "original_src_and_sampled_dst"
    weighted_loss_method = "ef_iwf_as_pos_edges_weight"
    compute_xf_iwf_with_sigmoid = False
  elif args.use_sigmoid_ef_iwf_weight:
    raise NotImplementedError("I don't expect this to be used anymore.")
    prefix = "use_sigmoid_ef_iwf_weight"
    neg_sample_method = "random"
    neg_edges_formation = "original_src_and_sampled_dst"
    weighted_loss_method = "ef_iwf_as_pos_edges_weight"
    compute_xf_iwf_with_sigmoid = True
  elif args.use_nf_iwf_neg_sampling:
    assert args.max_random_weight_range is None
    prefix = "use_nf_iwf_neg_sampling"
    neg_sample_method = "nf_iwf"
    neg_edges_formation = "sampled_src_and_sampled_dst"
    weighted_loss_method = "nf_iwf_as_pos_and_neg_edge_weight"
    compute_xf_iwf_with_sigmoid = False
  elif args.use_random_weight_to_benchmark_ef_iwf:
    assert args.max_random_weight_range is not None
    prefix = "use_random_weight_to_benchmark_ef_iwf"
    neg_sample_method = "random"
    neg_edges_formation = "original_src_and_sampled_dst"
    weighted_loss_method = "random_as_pos_edges_weight" # return new random weight from given range for a new window.
    compute_xf_iwf_with_sigmoid = False
  elif args.use_random_weight_to_benchmark_ef_iwf_1:
    assert args.max_random_weight_range is not None
    prefix = "use_share_selected_random_weight_per_window_to_benchmark_ef_iwf" # I decide to change prefix to not be the same as args because args name can change so the prefix should describe behavior instead.
    neg_sample_method = "random"
    neg_edges_formation = "original_src_and_sampled_dst"
    weighted_loss_method = "share_selected_random_weight_per_window" # all instances in each window shares same weight, but each window will be assigned weight randomly.
    compute_xf_iwf_with_sigmoid = False
  else:
    assert args.max_random_weight_range is None
    prefix = "original"
    neg_sample_method = "random"
    neg_edges_formation = "original_src_and_sampled_dst"
    weighted_loss_method = "no_weight"
    compute_xf_iwf_with_sigmoid = False

  # conditions = {}
  # conditions['neg_sample_method'] = neg_sample_method
  # conditions['neg_edges_formation'] = neg_edges_formation
  # conditions['weighted_loss_method'] = weighted_loss_method
  # conditions['compute_xf_iwf_with_sigmoid'] = compute_xf_iwf_with_sigmoid

  return prefix, neg_sample_method, neg_edges_formation, weighted_loss_method, compute_xf_iwf_with_sigmoid


class EF_IWF:
  def __init__(self):
    self.dict_ = {}

class NF_IWF:
  def __init__(self):
    self.dict_ = {}

class SHARE_SELECTED_RANDOM_WEIGHT:
  def __init__(self):
    self.dict_ = {}

class ArgsContraint:

  # def args_constraint(self, prefix, data_size, window_size, batch_size):
  #   args_naming_contraint(prefix)
  #   args_window_sliding_contraint(data_size, window_size, batch_size)

  def args_naming_contraint(self, prefix):
    assert prefix is None or len(prefix) == 0 , "args.prefix is deprecated. use custom_prefix instead"

  def args_window_sliding_contraint(self, data_size, window_size, batch_size):
    assert data_size/window_size == int(data_size/window_size)
    assert window_size/batch_size == int(window_size/batch_size)

  def args_window_sliding_training(self, backprop_every):
    assert backprop_every == 1 # NOTE: current implementation of ensemble will assume backprop_every to be 1

def return_min_length_of_list_members(list_of_vars, is_flatten_list=False):
  min_length = float('inf')
  for i in list_of_vars:
    if is_flatten_list:
      i = np.array(i).reshape(-1)
    len_ = len(i)
    min_length = min(min_length, len_)

    # min_length = min(min_length, len(i))
  return min_length

def get_an_arg_name(my_args, arg_name, return_if_args_value_is_true=True):
    """
    type(arg_name) is str
    """
    raise NotImplementedError("haven't check its correctness yet, because I haven't get a change to use it.")
    return_arg_name = None
    if arg_name in dir(my_args):
        return_arg_name = arg_name

    if return_if_args_value_is_true:
        assert getattr(my_args, arg_name) is True

    assert return_arg_name is not None

    return return_arg_name


class CheckPoint():
    def __init__(self):
        self.data = None
        self.bs = None
        self.prefix = None
        self.ws_idx = None
        self.ws_max = None
        self.custom_prefix = 'None' # custom_prefix can be None. it value is from args.prefix.
        self.epoch_max = None
        self.run_idx = None
        self.log_timestamp = None
        self.is_node_classification = None
        self.max_random_weight_range = None

    def get_checkpoint_path(self):
        assert self.data is not None
        assert self.bs is not None
        assert self.prefix is not None
        assert self.ws_idx is not None
        assert self.ws_max is not None
        assert self.run_idx is not None
        assert self.log_timestamp is not None
        assert self.is_node_classification is not None
        assert self.epoch_max is not None

        if self.max_random_weight_range is None:
          max_random_weight_range = "None"
        else:
          max_random_weight_range = int(self.max_random_weight_range)


        checkpoint_path = None
        general_checkpoint_path = None

        general_checkpoint_path = Path(f'custom_prefix={self.custom_prefix}-prefix={self.prefix}-data={self.data}-ws_max={self.ws_max}-epoch_max={self.epoch_max}-bs={self.bs}-ws_idx{self.ws_idx}-run_idx={self.run_idx}-{self.log_timestamp}-max_weight={max_random_weight_range}.pth')

        if self.is_node_classification:
            checkpoint_dir = 'node-classification'
        else:
            checkpoint_dir = 'link-prediction'

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(Path('saved_checkpoints') /Path(checkpoint_dir) / general_checkpoint_path)

        assert self.data in checkpoint_path
        assert str(self.bs) in checkpoint_path
        assert self.prefix in checkpoint_path
        assert str(self.ws_idx) in checkpoint_path
        assert str(self.ws_max) in checkpoint_path
        assert str(self.log_timestamp) in checkpoint_path
        assert str(self.run_idx) in checkpoint_path
        assert str(self.epoch_max) in checkpoint_path

        return checkpoint_path


def setup_logger(formatter, name, log_file, level=logging.INFO):
    """
    To setup as many loggers as you want

    :NOTE: I am not sure if refactor setup_logger from train_self_supervised into utils will results in information being logged  and stoed correctly. lets run first and I will inspect the results.
    """

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    # fh.terminator = ""

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    ch.setFormatter(formatter)
    # ch.terminator = ""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_share_selected_random_weight_per_window(batch_size, max_weight, batch_idx, share_selected_random_weight_per_window_dict):

  share_selected_random_weight_per_window_dict = add_only_new_values_of_new_window_to_dict(compute_share_selected_random_weight_per_window, batch_size, max_weight)(
      batch_idx,
      share_selected_random_weight_per_window_dict,
      None
    )
  return share_selected_random_weight_per_window_dict[batch_idx]

def compute_share_selected_random_weight_per_window(batch_size, max_weight):
  selected_rand_weight = random.choices(list(range(max_weight)), k=1)
  rand_weight = torch.FloatTensor([selected_rand_weight for i in range(batch_size)]).reshape(-1)
  return rand_weight

def add_only_new_values_of_new_window_to_dict(func, *args,**kwargs):
  # def add_values(*something):
    # print(*something)
  def add_values(batch_idx,a_dict,param_idx):
    idx_of_parameter_to_return_from_func = None if param_idx is None else param_idx
    if batch_idx not in a_dict:
      output = func(*args, **kwargs)
      if isinstance(output, tuple):
        assert idx_of_parameter_to_return_from_func is not None
        a_dict[batch_idx] = output[idx_of_parameter_to_return_from_func]
      else:
        assert idx_of_parameter_to_return_from_func is None
        a_dict[batch_idx] = output

    return a_dict

  return add_values

def get_encoder(n_uniq_labels):
  enc = preprocessing.OneHotEncoder()
  enc.fit(pd.DataFrame(range(n_uniq_labels)))
  return enc

def convert_to_onehot(labels, n_uniq_labels):
  one_hot = enc.transform(labels).toarray()
  return torch.Tensor(one_hot)

def get_nf_iwf(data, batch_idx, batch_size, start_train_idx, end_train_hard_negative_idx, nf_iwf_window_dict, sampled_nodes=None):

  start_past_window_idx = start_train_idx
  end_past_window_idx = end_train_hard_negative_idx
  nodes_in_past_windows = data.sources[:start_past_window_idx]
  nodes_in_current_window = data.sources[start_past_window_idx:end_past_window_idx]
  src_nodes_weight = []

  nf_iwf_window_dict = add_only_new_values_of_new_window_to_dict(compute_xf_iwf, nodes_in_past_windows, nodes_in_current_window , batch_size, compute_as_nodes=True, return_x_value_dict=True)(
    batch_idx, nf_iwf_window_dict, 1)

  # if batch_idx not in nf_iwf_window_dict:
  #   nf_iwf, nodes_to_nf_iwf_current_window_dict = compute_xf_iwf(nodes_in_past_windows, nodes_in_current_window , batch_size, compute_as_nodes=True, return_x_value_dict=True)
  #   nf_iwf_window_dict[batch_idx] = nodes_to_nf_iwf_current_window_dict
  #   # nf_iwf_current_window_dict = nf_iwf_window_dict[batch_idx]

  # there are two sampling cases:
  # 1. original. (This case. sampled src may not be inside of nf_iwf_window_dict)
  # 2. using EdgeSampler_NF_IWF. (This case nodes is garantee to be in the current window)
  selected_nodes = nodes_in_current_window
  if sampled_nodes is not None:
    selected_nodes = sampled_nodes

  for ii in selected_nodes:
    src_nodes_weight.append(nf_iwf_window_dict[batch_idx][ii])

  # assert len(src_nodes_weight) == nodes_in_current_window.shape[0]

  src_nodes_weight = torch.FloatTensor(src_nodes_weight)
  return src_nodes_weight

def get_conditions_node_classification(args):
  if args.use_nf_iwf_weight:
    prefix = "use_nf_iwf_weight"
    weighted_loss_method = 'nf_iwf_as_nodes_weight'
  elif args.use_random_weight_to_benchmark_nf_iwf:
    prefix = "use_random_weight_to_benchmark_nf_iwf"
    weighted_loss_method = 'random_as_node_weight'
  elif args.use_random_weight_to_benchmark_nf_iwf_1:
    prefix = "use_share_selected_random_weight_per_window_to_benchmark_nf_iwf" # I decide to change prefix to not be the same as args because args name can change so the prefix should describe behavior instead.
    weighted_loss_method = "share_selected_random_weight_per_window"
  else:
    prefix = "original"
    weighted_loss_method = 'no_weight'

  return prefix, weighted_loss_method

# def get_conditions(args):
#   if args.use_ef_iwf_weight:
#     assert args.max_random_weight_range is None
#     prefix = 'use_ef_iwf_weight'
#     neg_sample_method = "random"
#     neg_edges_formation = "original_src_and_sampled_dst"
#     weighted_loss_method = "ef_iwf_as_pos_edges_weight"
#     compute_xf_iwf_with_sigmoid = False
#   elif args.use_sigmoid_ef_iwf_weight:
#     raise NotImplementedError("I don't expect this to be used anymore.")
#     prefix = "use_sigmoid_ef_iwf_weight"
#     neg_sample_method = "random"
#     neg_edges_formation = "original_src_and_sampled_dst"
#     weighted_loss_method = "ef_iwf_as_pos_edges_weight"
#     compute_xf_iwf_with_sigmoid = True
#   elif args.use_nf_iwf_neg_sampling:
#     assert args.max_random_weight_range is None
#     prefix = "use_nf_iwf_neg_sampling"
#     neg_sample_method = "nf_iwf"
#     neg_edges_formation = "sampled_src_and_sampled_dst"
#     weighted_loss_method = "nf_iwf_as_pos_and_neg_edge_weight"
#     compute_xf_iwf_with_sigmoid = False
#   elif args.use_random_weight_to_benchmark_ef_iwf:
#     assert args.max_random_weight_range is not None
#     prefix = "use_random_weight_to_benchmark_ef_iwf"
#     neg_sample_method = "random"
#     neg_edges_formation = "original_src_and_sampled_dst"
#     weighted_loss_method = "random_as_pos_edges_weight" # return new random weight from given range for a new window.
#     compute_xf_iwf_with_sigmoid = False
#   elif args.use_random_weight_to_benchmark_ef_iwf_1:
#     assert args.max_random_weight_range is not None
#     prefix = "use_share_selected_random_weight_per_window_to_benchmark_ef_iwf" # I decide to change prefix to not be the same as args because args name can change so the prefix should describe behavior instead.
#     neg_sample_method = "random"
#     neg_edges_formation = "original_src_and_sampled_dst"
#     weighted_loss_method = "share_selected_random_weight_per_window" # all instances in each window shares same weight, but each window will be assigned weight randomly.
#     compute_xf_iwf_with_sigmoid = False
#   else:
#     assert args.max_random_weight_range is None
#     prefix = "original"
#     neg_sample_method = "random"
#     neg_edges_formation = "original_src_and_sampled_dst"
#     weighted_loss_method = "no_weight"
#     compute_xf_iwf_with_sigmoid = False

#   # conditions = {}
#   # conditions['neg_sample_method'] = neg_sample_method
#   # conditions['neg_edges_formation'] = neg_edges_formation
#   # conditions['weighted_loss_method'] = weighted_loss_method
#   # conditions['compute_xf_iwf_with_sigmoid'] = compute_xf_iwf_with_sigmoid

#   return prefix, neg_sample_method, neg_edges_formation, weighted_loss_method, compute_xf_iwf_with_sigmoid

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

  # ef = current_uniq_edges_freq * 0.1
  ef = 1/current_uniq_edges_freq * 0.1
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

def compute_iwf_from_wf(wf,n_all_window_contain_current_x):
  """
  type(wf) is int
  type(n_all_window_contain_current_x) is numpy array
  """
  return np.array(list(map(math.log, wf/n_all_window_contain_current_x)))

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

  # iwf = np.array(list(map(math.log, wf/n_all_window_contain_current_x)))
  iwf = compute_iwf_from_wf(wf, n_all_window_contain_current_x)

  iwf_mask = np.where(n_all_window_contain_current_x==0)[0]
  iwf[iwf_mask] = 999999 # replace inf value with very large number.

  # :NOTE: apply sigmoid function to set range of iwf to be [0,1]
  # if compute_ef_iwf_with_sigmoid:
  #   iwf = torch.nn.functional.sigmoid(torch.from_numpy(iwf)).cpu().detach().numpy()
  # print(wf)

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

def get_unique_sources_ind_to_be_added(new_sources, sources_batch, n_unique_sources_to_add):
  unique_sources_ind = np.unique(new_sources, return_index=True)[1]

  if unique_sources_ind.shape[0] < n_unique_sources_to_add:
    unique_sources_ind_to_add = unique_sources_ind
  else:
    unique_sources_ind_to_add = random.sample(unique_sources_ind.tolist(),n_unique_sources_to_add)

  return unique_sources_ind_to_add

def get_list_of_all_unique_nodes_to_be_labeled(selected_sources_to_label, sources_batch, n_unique_sources_to_add):

  # create mask for new batch
  if n_unique_sources_to_add > 0:
    selected_sources_to_label_batch_mask = np.array([False for i in range(sources_batch.shape[0])])
    if len(selected_sources_to_label) > 0:
      selected_sources_to_label_batch_mask = np.array(list(map(lambda x: x in selected_sources_to_label, sources_batch)))

    new_sources = sources_batch[~selected_sources_to_label_batch_mask]
    # existing_sources = sources_batch[selected_sources_to_label_batch_mask]
    # assert np.intersect1d(existing_sources, new_sources).shape[0] == 0

    unique_sources_ind_to_add = None

    # after mask is applied, random pick new unique node.
    if new_sources.shape[0] > 0:
      unique_sources_ind_to_add = get_unique_sources_ind_to_be_added(new_sources, sources_batch, n_unique_sources_to_add)
      selected_sources_to_label.extend(new_sources[unique_sources_ind_to_add])

      assert np.unique(selected_sources_to_label).shape[0] == len(selected_sources_to_label)

  return selected_sources_to_label


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

  # label nodes that are in sources_batch.
  # selected_sources_ind = find_nodes_ind_to_be_labelled(selected_sources_to_label, sources_batch)
  selected_sources_to_label =  get_list_of_all_unique_nodes_to_be_labeled(selected_sources_to_label,sources_batch, n_unique_sources_to_add)

  selected_sources_ind = find_nodes_ind_to_be_labelled(
    selected_sources_to_label ,
    sources_batch)

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
