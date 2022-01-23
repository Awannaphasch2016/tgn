#!/usr/bin/env python3
#
import math

import numpy as np
import time
import pickle
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.utils import EarlyStopMonitor, get_neighbor_finder, compute_xf_iwf, compute_nf,  get_nf_iwf, add_only_new_values_of_new_window_to_dict, compute_share_selected_random_weight_per_window, get_share_selected_random_weight_per_window, EF_IWF, NF_IWF, SHARE_SELECTED_RANDOM_WEIGHT
from utils.sampler import RandEdgeSampler, EdgeSampler_NF_IWF
from utils.data_processing import Data
from tqdm import tqdm
import random

from evaluation.evaluation import get_sampler, get_negative_nodes_batch, init_pos_neg_labels, get_criterion, compute_edges_probabilities_with_custom_sampled_nodes, get_edges_weight, compute_loss

class SlidingWindow:
  def __init__(self, args):
    self.args = args

  def add_dataset(self, full_data):
    self.full_data = full_data

  def pre_evaluation(self):
    # :TODO: write test on these. raise exception for all cases that wasn't intended or designed for.
    self.prefix, self.neg_sample_method, self.neg_edges_formation, self.weighted_loss_method, self.compute_xf_iwf_with_sigmoid = self.get_conditions()

    self.num_instances_shift, self.init_train_data, self.total_num_ws, self.init_num_ws, self.left_num_ws = self.get_sliding_window_params(self.full_data.data_size, self.args.bs, self.args.ws_multiplier)


  def add_hardware_params(self, device):
    self.device = device

  def add_checkpoints(self, check_point):
    self.check_point = check_point

  def add_loggers(self, logger, logger_2):
    self.logger = logger
    self.logger_2 = logger_2


  def get_sliding_window_params(self, num_instance, batch_size, ws_multiplier):

    num_instances_shift = batch_size * ws_multiplier # 100

    # init_train_data = BATCH_SIZE
    # init_train_data = math.ceil(num_instance * 0.001)
    # init_train_data = batch_size * max(1,math.floor(init_train_data/num_instances_shift))
    init_train_data = batch_size * ws_multiplier


    total_num_ws =  math.floor(num_instance/num_instances_shift) # 6
    init_num_ws = math.floor(init_train_data/num_instances_shift) #6
    left_num_ws = total_num_ws - init_num_ws


    return num_instances_shift, init_train_data, total_num_ws, init_num_ws, left_num_ws

  # def get_conditions(self, args):
  def get_conditions(self):
    args = self.args
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


  def add_model_training_params(self, optimizer):
    self.optimizer = optimizer

  def add_model_params(self, num_neighbors):
    self.num_neighbors = num_neighbors

  def add_model(self, model):
    self.model = model

  # def set_begin_end_batch_idx(self, batch_idx, backprop_idx, batch_size):

  def evaluate_epoch(self, epoch, num_batch):

    args = self.args
    check_point = self.check_point
    optimizer = self.optimizer
    full_data = self.full_data
    init_train_data = self.init_train_data
    neg_sample_method = self.neg_sample_method
    neg_edges_formation = self.neg_edges_formation
    NUM_NEIGHBORS = self.num_neighbors
    device = self.device
    weighted_loss_method = self.weighted_loss_method
    compute_xf_iwf_with_sigmoid = self.compute_xf_iwf_with_sigmoid

    num_instance = full_data.data_size
    max_weight = args.max_random_weight_range
    BATCH_SIZE = args.bs
    USE_MEMORY = args.use_memory

    ef_iwf_window_dict = self.ef_iwf_window_dict.dict_
    nf_iwf_window_dict = self.nf_iwf_window_dict.dict_
    share_selected_random_weight_per_window_dict = self.share_selected_random_weight_per_window_dict.dict_


    check_point.epoch_idx = epoch
    check_point.get_checkpoint_path()

    self.logger.debug('--epoch = {}'.format(epoch))
    self.logger_2.info('--epoch = {}'.format(epoch))
    self.logger.info(f'max_weight = {max_weight}')

    start_epoch = time.time()
    ### Training :DOC:

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
        self.model.memory.__init_memory__()

    for k in range(0, num_batch, args.backprop_every):
      # logger.debug('---batch = {}'.format(k))
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        # logger.debug('----backprop_every = {}'.format(j))
        batch_idx = k + j
        start_train_idx = batch_idx * BATCH_SIZE

        batch_ref_window_size = 0
        assert batch_ref_window_size < BATCH_SIZE

        end_train_idx = min(init_train_data-batch_ref_window_size, start_train_idx + BATCH_SIZE)
        end_train_idx = min(end_train_idx, num_instance-batch_ref_window_size) # edge case for hard sampling window.
        end_train_hard_negative_idx = end_train_idx + batch_ref_window_size

        assert end_train_hard_negative_idx <= init_train_data

        if end_train_idx <= (num_instance - batch_ref_window_size):
            assert (end_train_hard_negative_idx - batch_ref_window_size) == end_train_idx

        assert start_train_idx < end_train_idx, "number of batch to run for each epoch was not set correctly."

        sources_batch, destinations_batch = full_data.sources[start_train_idx:end_train_idx], \
                                            full_data.destinations[start_train_idx:end_train_idx]
        edge_idxs_batch = full_data.edge_idxs[start_train_idx: end_train_idx]
        timestamps_batch = full_data.timestamps[start_train_idx:end_train_idx]

        edge_hard_samples_batch = full_data.edge_idxs[start_train_idx:end_train_hard_negative_idx]
        timestamps_hard_samples_batch = full_data.timestamps[start_train_idx:end_train_hard_negative_idx]

        train_mask = full_data.timestamps < full_data.timestamps[end_train_idx]

        # Initialize training neighbor finder to retrieve temporal graph
        train_data = Data(full_data.sources[train_mask],
                            full_data.destinations[train_mask],
                            full_data.timestamps[train_mask],
                            full_data.edge_idxs[train_mask],
                            full_data.labels[train_mask])


        # print(train_data.n_interactions, train_data.n_unique_nodes)
        train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
        self.model.set_neighbor_finder(train_ngh_finder)

        size = len(sources_batch)

        train_rand_sampler = get_sampler(train_data, batch_ref_window_size, start_train_idx, end_train_hard_negative_idx, neg_sample_method)

        negatives_src_batch, negatives_dst_batch = get_negative_nodes_batch(train_rand_sampler, BATCH_SIZE, size, neg_sample_method)

        pos_label, neg_label = init_pos_neg_labels(size, device)

        criterion = get_criterion()


        pos_prob, neg_prob = compute_edges_probabilities_with_custom_sampled_nodes(self.model, neg_edges_formation, negatives_dst_batch, negatives_src_batch, sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

        pos_edges_weight, neg_edges_weight = get_edges_weight(train_data,k, BATCH_SIZE,max_weight,start_train_idx, end_train_hard_negative_idx, ef_iwf_window_dict, nf_iwf_window_dict, share_selected_random_weight_per_window_dict, weighted_loss_method, sampled_nodes=negatives_src_batch, compute_xf_iwf_with_sigmoid=compute_xf_iwf_with_sigmoid)

        self.logger_2.info(f'pos_edges_weight = {pos_edges_weight}')
        self.logger_2.info(f'neg_edges_weight = {neg_edges_weight}')

        loss = compute_loss(pos_label, neg_label, pos_prob, neg_prob, pos_edges_weight, neg_edges_weight, batch_idx, criterion, loss, weighted_loss_method)

        loss /= args.backprop_every

        loss.backward()
        optimizer.step()
        self.m_loss.append(loss.item())

        # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
        # the start of time
        if USE_MEMORY:
          self.model.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    self.epoch_times.append(epoch_time)

    self.logger.info('start validation...')
    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    ### Validation
    # Validation uses the full graph
    self.model.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
        # Backup memory at the end of training, so later we can restore it and use it for the
        # validation on unseen nodes
        train_memory_backup = self.model.memory.backup_memory()

    VAL_BATCH_SIZE = BATCH_SIZE * 10
    sources_batch = full_data.sources[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
    destinations_batch  = full_data.destinations[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
    timestamps_batch  = full_data.timestamps[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
    edge_idxs_batch  = full_data.edge_idxs[end_train_idx:end_train_idx + VAL_BATCH_SIZE]


    size = len(sources_batch)
    _, negative_samples = train_rand_sampler.sample(size)

    pos_prob, neg_prob = self.model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, NUM_NEIGHBORS)


    pred_score = np.concatenate([pos_prob.cpu().data.detach().numpy(), neg_prob.cpu().data.detach().numpy()])
    true_label = np.concatenate([np.ones(size), np.zeros(size)])

    from sklearn.metrics import average_precision_score, roc_auc_score

    val_auc = average_precision_score(true_label, pred_score)
    val_ap = roc_auc_score(true_label, pred_score)

    if USE_MEMORY:
        val_memory_backup = self.model.memory.backup_memory()
        # Restore memory we had at the end of training to be used when validating on new nodes.
        # Also backup memory after validation so it can be used for testing (since test edges are
        # strictly later in time than validation edges)
        self.model.memory.restore_memory(train_memory_backup)

    total_epoch_time = time.time() - start_epoch
    self.total_epoch_times.append(total_epoch_time)

    self.logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    self.logger.info('Epoch mean loss: {}'.format(np.mean(self.m_loss)))
    self.logger.info(
        'val auc: {}'.format(val_auc))
    self.logger.info(
        'val ap: {}'.format(val_ap))

  def save_checkpoint_per_ws(self):
    if self.args.save_checkpoint:
      torch.save(self.model.state_dict(), self.check_point.get_checkpoint_path())

  def init_params_that_tracks_history_of_ws(self):
    self.ef_iwf_window_dict = EF_IWF()
    self.nf_iwf_window_dict = NF_IWF()
    self.share_selected_random_weight_per_window_dict = SHARE_SELECTED_RANDOM_WEIGHT()


  def evaluate_ws(self, n_ws_to_run):
    args = self.args
    full_data = self.full_data
    check_point = self.check_point
    device = self.device
    optimizer = self.optimizer

    BATCH_SIZE = args.bs
    NUM_EPOCH = args.n_epoch
    USE_MEMORY = args.use_memory

    prefix  = self.prefix
    neg_sample_method = self.neg_sample_method
    neg_edges_formation  = self.neg_edges_formation
    weighted_loss_method  = self.weighted_loss_method

    num_instances_shift = self.num_instances_shift
    # init_train_data = self.init_train_data
    total_num_ws = self.total_num_ws
    # init_num_ws = self.init_num_ws

    check_point.data = args.data
    check_point.prefix = prefix
    check_point.bs = args.bs
    check_point.epoch_max = args.n_epoch
    check_point.ws_max = total_num_ws
    check_point.max_random_weight_range = args.max_random_weight_range

    pos_edges_weight = None
    neg_edges_weight = None
    self.epoch_times = []
    self.total_epoch_times = []
    end_train_idx = None

    for ws in range(n_ws_to_run):
      check_point.ws_idx = ws
      num_batch = math.ceil((self.init_train_data)/BATCH_SIZE)
      self.logger.debug('-ws = {}'.format(ws))
      self.logger_2.info('-ws = {}'.format(ws))
      self.m_loss = []

      for epoch in range(NUM_EPOCH):
        self.evaluate_epoch(epoch, num_batch)

      self.save_checkpoint_per_ws()

      self.init_num_ws += 1
      self.init_train_data = self.init_num_ws * num_instances_shift

      # Training has finished, we have loaded the best model, and we want to backup its current
      # memory (which has seen validation edges) so that it can also be used when testing on unseen

      if USE_MEMORY:
        val_memory_backup = self.model.memory.backup_memory()
  def evaluate(self):
    self.evaluate_ws(self.left_num_ws)
