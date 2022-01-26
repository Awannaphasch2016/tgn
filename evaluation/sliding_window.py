#!/usr/bin/env python3
#
import math
import pandas as pd
import numpy as np
import time
import pickle
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.utils import EarlyStopMonitor, get_neighbor_finder, compute_xf_iwf, compute_nf,  get_nf_iwf, add_only_new_values_of_new_window_to_dict, compute_share_selected_random_weight_per_window, get_share_selected_random_weight_per_window, EF_IWF, NF_IWF, SHARE_SELECTED_RANDOM_WEIGHT, get_conditions, apply_off_set_ind, convert_n_instances_to_ind, convert_ind_to_n_instances, right_split_ind_by_window, convert_window_idx_to_batch_idx, convert_window_idx_to_batch_idx, split_a_window_into_batches, convert_prob_list_to_binary_list, get_conditions_node_classification, label_new_unique_nodes_with_budget, pred_prob_to_pred_labels,get_batch_idx_relative_to_window_idx, get_encoder, convert_to_onehot,get_label_distribution, get_unique_nodes_labels
from utils.sampler import RandEdgeSampler, EdgeSampler_NF_IWF
from utils.data_processing import Data
from tqdm import tqdm
import random
# from modules.sliding_window_framework import WindowSlidingForward, WindowSlidingEnsemble


from evaluation.evaluation import get_sampler, get_negative_nodes_batch, init_pos_neg_labels, get_criterion, compute_edges_probabilities_with_custom_sampled_nodes, get_edges_weight, compute_loss, compute_evaluation_score, compute_precision, compute_auc_for_ensemble

from evaluation.eval_node_classification import select_decoder_and_loss_node_classification, get_nodes_weight, my_eval_node_classification

from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from modules.ensemble import get_all_ensemble_training_data_inds

class SlidingWindow:
  def __init__(self, args):
    self.args = args

  def set_decoder(self):
    raise NotImplementedError()

  # def set_model(self, ModelClass,
  def set_encoder(self, ModelClass,
                  neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False):
    """
    NOTE: later on, I may need to implement this function using discousure property of python.
    """
    self.model_kwargs = {
      "neighbor_finder":neighbor_finder,
      "node_features":node_features,
      "edge_features":edge_features,
      "device":device,
      "n_layers":n_layers,
      "n_heads":n_heads,
      "dropout":dropout,
      "use_memory":use_memory,
      "message_dimension":message_dimension,
      "memory_dimension":memory_dimension,
      "memory_update_at_start":memory_update_at_start,
      "embedding_module_type":embedding_module_type,
      "message_function":message_function,
      "aggregator_type":aggregator_type,
      "memory_updater_type":memory_updater_type,
      "n_neighbors":n_neighbors,
      "mean_time_shift_src":mean_time_shift_src,
      "std_time_shift_src":std_time_shift_src,
      "mean_time_shift_dst":mean_time_shift_dst,
      "std_time_shift_dst":std_time_shift_dst,
      "use_destination_embedding_in_message":use_destination_embedding_in_message,
      "use_source_embedding_in_message":use_source_embedding_in_message,
      "dyrep":dyrep
    }
    
    self.ModelClass = ModelClass

  def save_checkpoint_per_ws(self):
    if self.args.save_checkpoint:
      torch.save(self.model.state_dict(), self.check_point.get_checkpoint_path())

  def init_params_that_tracks_history_of_ws_node_classification(self):
    self.selected_sources_to_label = []
    self.selected_sources_ind = None

  def init_params_that_tracks_history_of_ws(self):
    self.ef_iwf_window_dict = EF_IWF()
    self.nf_iwf_window_dict = NF_IWF()
    self.share_selected_random_weight_per_window_dict = SHARE_SELECTED_RANDOM_WEIGHT()

  def get_conditions(self):
    raise NotImplementedError()

  # def add_dataset(self, full_data):
  #   self.full_data = full_data

  def add_dataset(self, full_data, node_features, edge_features, train_data, val_data, test_data):
    self.full_data = full_data
    self.node_features = node_features
    self.train_data = train_data
    self.val_data = val_data
    self.test_data = test_data

  def add_data(self, data_transformed_collection):
    self.data_transformed_collection = data_transformed_collection
    self.add_dataset(
      self.data_transformed_collection.full_data,
      self.data_transformed_collection.node_features,
      self.data_transformed_collection.edge_features,
      self.data_transformed_collection.train_data,
      self.data_transformed_collection.val_data,
      self.data_transformed_collection.test_data)

  def add_hardware_params(self, device):
    self.device = device

  def add_checkpoints(self, check_point):
    self.check_point = check_point

  def add_loggers(self, logger, logger_2):
    self.logger = logger
    self.logger_2 = logger_2

  def add_model_training_params(self, n_epoch):
    self.n_epoch = n_epoch

  def add_model_params(self, num_neighbors, use_memory):

    self.num_neighbors = num_neighbors
    self.use_memory = use_memory

  def add_model(self):
    raise NotImplementedError

  def pre_evaluation(self):
    raise NotImplementedError 

  def get_sliding_window_params(self, num_instance, batch_size, ws_multiplier):

    # init_train_data = BATCH_SIZE
    # init_train_data = math.ceil(num_instance * 0.001)
    # init_train_data = batch_size * max(1,math.floor(init_train_data/num_instances_shift))
    window_size = batch_size * ws_multiplier
    num_init_data = window_size
    init_train_data = window_size # :NOTE: Bad naming, but I keep this for compatibility reason.

    num_instances_shift = init_train_data

    total_num_ws =  math.floor(num_instance/num_instances_shift) # 6
    init_num_ws = math.floor(init_train_data/num_instances_shift) #6
    left_num_ws = total_num_ws - init_num_ws

    return window_size, num_init_data, num_instances_shift, init_train_data, total_num_ws, init_num_ws, left_num_ws

  def set_sliding_window_framework(self, ws_framework):
    if ws_framework == "forward":
      self.WSFramework = WindowSlidingForward
    elif ws_framework == "ensemble":
      self.WSFramework = WindowSlidingEnsemble
    else:
      raise NotImplementedError()

  def set_begin_end_batch_idx(self, size_of_current_concat_windows, batch_idx, batch_size, batch_ref_window_size):
    """
    :NOTE: at the time of writing, I only care about begin and end batch.
    """
    assert batch_ref_window_size == 0, "I don't use this param as of now."
    assert batch_ref_window_size < batch_size

    start_train_idx = batch_idx * batch_size

    batch_ref_window_size = 0
    assert batch_ref_window_size < batch_size

    end_train_idx = min(size_of_current_concat_windows-batch_ref_window_size, start_train_idx + batch_size)
    end_train_idx = min(end_train_idx, self.full_data.data_size - batch_ref_window_size) # edge case for hard sampling window.
    assert start_train_idx < end_train_idx, "number of batch to run for each epoch was not set correctly."

    end_train_hard_negative_idx = end_train_idx + batch_ref_window_size
    assert end_train_hard_negative_idx <= size_of_current_concat_windows
    if end_train_idx <= (self.full_data.data_size - batch_ref_window_size):
        assert (end_train_hard_negative_idx - batch_ref_window_size) == end_train_idx


    return start_train_idx, end_train_idx

  # def set_params_window(self, start_idx, end_idx):
  #   self.set_params_batch(start_idx, end_idx)

  def set_params_mask(self, left, right ):
    raise NotImplementedError()

  def set_params_batch(self, start_train_idx, end_train_idx):

    sources_batch, destinations_batch = self.full_data.sources[start_train_idx:end_train_idx], \
                                        self.full_data.destinations[start_train_idx:end_train_idx]
    edge_idxs_batch = self.full_data.edge_idxs[start_train_idx: end_train_idx]
    timestamps_batch = self.full_data.timestamps[start_train_idx:end_train_idx]
    labels_batch = self.full_data.labels[start_train_idx:end_train_idx]

    # edge_hard_samples_batch = full_data.edge_idxs[start_train_idx:end_train_hard_negative_idx]
    # timestamps_hard_samples_batch = full_data.timestamps[start_train_idx:end_train_hard_negative_idx]

    return  sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, labels_batch


  def evaluate_batch(self):
    raise NotImplementedError
  
  # def evaluate_epoch(self, epoch, num_batch):
  def evaluate_epoch(self):
    raise NotImplementedError()

  def evaluate_ws(self):
    raise NotImplementedError()

  def evaluate(self):
    raise NotImplementedError

class WindowSlidingForward(SlidingWindow):

  # def add_dataset(self, full_data):
  #   self.full_data = full_data

  # def add_dataset(self, data_name, different_new_nodes_between_val_and_test, randomize_features):
  #   self.node_features, self.edge_features, self.full_data, self.train_data, self.val_data, self.test_data, self.new_node_val_data, self.new_node_test_data, self.timestamps, self.observed_edges_mask = get_data(DATA, different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

  def get_conditions(self):
    # :TODO: write test on these. raise exception for all cases that wasn't intended or designed for.
    self.prefix, self.neg_sample_method, self.neg_edges_formation, self.weighted_loss_method, self.compute_xf_iwf_with_sigmoid = get_conditions(self.args)

  def set_decoder(self):
    """
    :NOTE: I am aware taht optimizer and criterion doesn't belong to decoder only, but it needs somewhere to belong, and here it is.
    """
    # select_decoder_and_loss
    # NOTE: For link prediction, decoder is defined in TGN as self.affinity_score which is used in compute_edge_probabilities
    decoder = None
    criterion = get_criterion()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
    return decoder, optimizer, criterion

  def add_model(self):
    self.model = self.ModelClass(**self.model_kwargs)
    self.decoder, self.optimizer, self.criterion = self.set_decoder()
    # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
    self.model.to(self.device)
    begin_ind, idx_to_split, _ = right_split_ind_by_window(1, self.full_data.data_size, self.window_size)

    # self.window_begin_inds, self.window_end_inds = get_all_ensemble_training_data_inds(begin_ind, idx_to_split-1, self.window_size, fix_begin_ind=True)

    self.batch_inds = split_a_window_into_batches(begin_ind, idx_to_split- 1, self.args.bs)
    self.batch_begin_inds = self.batch_inds[:-1]
    self.batch_end_inds = self.batch_inds[1:]

  def pre_evaluation(self):

    self.window_size, self.num_init_data,self.num_instances_shift, self.init_train_data, self.total_num_ws, self.init_num_ws, self.left_num_ws = self.get_sliding_window_params(self.full_data.data_size, self.args.bs, self.args.ws_multiplier)

    self.add_model()

  def set_params_mask(self, start_idx, end_idx ):
    full_data = self.full_data

    # train_mask = full_data.timestamps < full_data.timestamps[end_train_idx]
    left =  full_data.timestamps < full_data.timestamps[end_idx]
    right = full_data.timestamps[0] <= full_data.timestamps
    mask = np.logical_and(right, left)

    return mask

  def evaluate_batch(self, model, k, backprop_every):
    args = self.args
    criterion = self.criterion
    optimizer = self.optimizer
    full_data = self.full_data
    # init_train_data = self.init_train_data
    NUM_NEIGHBORS = self.num_neighbors
    neg_sample_method = self.neg_sample_method
    neg_edges_formation = self.neg_edges_formation
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

    loss = 0
    optimizer.zero_grad()

    # Custom loop to allow to perform backpropagation only every a certain number of batches
    for j in range(backprop_every):
      # logger.debug('----backprop_every = {}'.format(j))

      batch_idx = k + j
      batch_ref_window_size = 0 # :NOTE: added for compatibility reason


      ## Indexing
      start_train_idx = self.batch_begin_inds[batch_idx]
      end_train_idx = self.batch_end_inds[batch_idx]
      # start_train_idx, end_train_idx = self.set_begin_end_batch_idx(self.init_train_data,batch_idx, BATCH_SIZE, batch_ref_window_size)

      end_train_hard_negative_idx = end_train_idx # :NOTE: added for compatibility reason.

      ## Masking
      train_mask = self.set_params_mask(start_train_idx, end_train_idx)

      # Initialize training neighbor finder to retrieve temporal graph
      train_data = Data(full_data.sources[train_mask],
                          full_data.destinations[train_mask],
                          full_data.timestamps[train_mask],
                          full_data.edge_idxs[train_mask],
                          full_data.labels[train_mask])

      # print(train_data.n_interactions, train_data.n_unique_nodes)
      train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
      model.set_neighbor_finder(train_ngh_finder)

      sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, _ = self.set_params_batch(start_train_idx, end_train_idx)

      ## Sampler
      size = len(sources_batch)
      train_rand_sampler = get_sampler(train_data, batch_ref_window_size, start_train_idx, end_train_hard_negative_idx, neg_sample_method)

      negatives_src_batch, negatives_dst_batch = get_negative_nodes_batch(train_rand_sampler, BATCH_SIZE, size, neg_sample_method)

      pos_label, neg_label = init_pos_neg_labels(size, device)

      # criterion = get_criterion()


      pos_prob, neg_prob = compute_edges_probabilities_with_custom_sampled_nodes(model, neg_edges_formation, negatives_dst_batch, negatives_src_batch, sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

      pos_edges_weight, neg_edges_weight = get_edges_weight(train_data,k, BATCH_SIZE,max_weight,start_train_idx, end_train_hard_negative_idx, ef_iwf_window_dict, nf_iwf_window_dict, share_selected_random_weight_per_window_dict, weighted_loss_method, sampled_nodes=negatives_src_batch, compute_xf_iwf_with_sigmoid=compute_xf_iwf_with_sigmoid)

      self.logger_2.info(f'pos_edges_weight = {pos_edges_weight}')
      self.logger_2.info(f'neg_edges_weight = {neg_edges_weight}')

      loss = compute_loss(pos_label, neg_label, pos_prob, neg_prob, pos_edges_weight, neg_edges_weight, batch_idx, criterion, loss, weighted_loss_method)

    loss /= backprop_every

    loss.backward()
    optimizer.step()
    self.m_loss.append(loss.item())

    # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
    # the start of time
    if USE_MEMORY:
      model.memory.detach_memory()

    self.start_train_idx = start_train_idx
    self.end_train_idx = end_train_idx
    self.train_rand_sampler = train_rand_sampler

  def evaluate_epoch(self, model, epoch, num_batch):
    args = self.args
    full_data = self.full_data
    # init_train_data = self.init_train_data
    NUM_NEIGHBORS = self.num_neighbors

    BATCH_SIZE = args.bs
    USE_MEMORY = args.use_memory

    start_epoch = time.time()

    ### Training :DOC:
    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      model.memory.__init_memory__()

    for k in range(0, num_batch, args.backprop_every):
    # logger.debug('---batch = {}'.format(k))
      self.evaluate_batch(model, k, args.backprop_every)
      start_train_idx = self.start_train_idx
      end_train_idx = self.end_train_idx
      train_rand_sampler = self.train_rand_sampler

    epoch_time = time.time() - start_epoch
    self.epoch_times.append(epoch_time)

    self.logger.info('start validation...')
    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    ### Validation
    # Validation uses the full graph
    model.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
        # Backup memory at the end of training, so later we can restore it and use it for the
        # validation on unseen nodes
        train_memory_backup = model.memory.backup_memory()

    # VAL_BATCH_SIZE = BATCH_SIZE * 1
    VAL_BATCH_SIZE = self.window_size
    sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, _ = self.set_params_batch(end_train_idx, end_train_idx + VAL_BATCH_SIZE)


    size = len(sources_batch)
    _, negative_samples = train_rand_sampler.sample(size)

    pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, NUM_NEIGHBORS)



    pred_score = np.concatenate([pos_prob.cpu().data.detach().numpy(), neg_prob.cpu().data.detach().numpy()])
    true_label = np.concatenate([np.ones(size), np.zeros(size)])

    val_auc, val_ap = compute_evaluation_score(true_label,pred_score)

    if USE_MEMORY:
        val_memory_backup = model.memory.backup_memory()
        # Restore memory we had at the end of training to be used when validating on new nodes.
        # Also backup memory after validation so it can be used for testing (since test edges are
        # strictly later in time than validation edges)
        model.memory.restore_memory(train_memory_backup)

    total_epoch_time = time.time() - start_epoch
    self.total_epoch_times.append(total_epoch_time)

    self.logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    self.logger.info('Epoch mean loss: {}'.format(np.mean(self.m_loss)))
    self.logger.info(
        'val auc: {}'.format(val_auc))
    self.logger.info(
        'val ap: {}'.format(val_ap))

  def evaluate(self):
    # raise NotImplementedError()

    args = self.args
    self.check_point.data = args.data
    self.check_point.prefix = self.prefix
    self.check_point.bs = args.bs
    self.check_point.epoch_max = args.n_epoch
    self.check_point.ws_max = self.total_num_ws
    self.check_point.max_random_weight_range = args.max_random_weight_range

    for ws in range(self.left_num_ws):
      self.check_point.ws_idx = ws
      self.logger.debug('-ws = {}'.format(ws))
      self.logger_2.info('-ws = {}'.format(ws))
      self.evaluate_ws(ws, self.init_train_data, self.args.bs)


  def evaluate_ws(self, ws, size_of_current_concat_windows, batch_size):
    num_batch = math.ceil((size_of_current_concat_windows)/batch_size)
    self.m_loss = []
    # for epoch in range(NUM_EPOCH):
    self.epoch_times = []
    self.total_epoch_times = []

    for epoch in range(self.n_epoch):
      self.check_point.epoch_idx = epoch
      self.check_point.get_checkpoint_path()

      self.logger.debug('--epoch = {}'.format(epoch))
      self.logger_2.info('--epoch = {}'.format(epoch))
      self.logger.info(f'max_weight = {self.args.max_random_weight_range}')

      self.evaluate_epoch(self.model, epoch, num_batch)

    self.save_checkpoint_per_ws()

    self.init_num_ws += 1
    self.init_train_data = self.init_num_ws * self.num_instances_shift

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen

    # if USE_MEMORY:
    if self.use_memory:
      val_memory_backup = self.model.memory.backup_memory()



class WindowSlidingEnsemble(SlidingWindow):
  def __init__(self, args):
    self.args = args
    self.models = {}

  # def add_dataset(self, full_data):
  #   self.full_data = full_data

  def get_conditions(self):
    self.prefix, self.neg_sample_method, self.neg_edges_formation, self.weighted_loss_method, self.compute_xf_iwf_with_sigmoid = get_conditions(self.args)

  def pre_evaluation(self):
    self.window_size, self.num_init_data,self.num_instances_shift, self.init_train_data, self.total_num_ws, self.init_num_ws, self.left_num_ws = self.get_sliding_window_params(self.full_data.data_size, self.args.bs, self.args.ws_multiplier)

    begin_ind, idx_to_split, _ = right_split_ind_by_window(1, self.full_data.data_size, self.window_size)

    self.ensemble_begin_inds, self.ensemble_end_inds = get_all_ensemble_training_data_inds(begin_ind, idx_to_split-1, self.window_size, fix_begin_ind=False)

  def add_model(self, idx):

    self.model = self.ModelClass(**self.model_kwargs)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
    self.model.to(self.device)
    # self.models[idx]["model"] = self.model
    # self.models[idx].setdefault("model", self.model)
    # self.models[idx] = {"model": self.model}
    self.models.setdefault(idx, {"model": self.model})

    self.batch_inds = split_a_window_into_batches(self.ensemble_begin_inds[idx], self.ensemble_end_inds[idx], self.args.bs)
    self.batch_begin_inds = self.batch_inds[:-1]
    self.batch_end_inds = self.batch_inds[1:]

  def evaluate_batch(self, ensemble_idx, k, backprop_every):
    model = self.models[ensemble_idx]["model"]
    args = self.args
    full_data = self.full_data
    # init_train_data = self.init_train_data
    NUM_NEIGHBORS = self.num_neighbors

    optimizer = self.optimizer
    neg_sample_method = self.neg_sample_method
    neg_edges_formation = self.neg_edges_formation
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

    loss = 0
    optimizer.zero_grad()

    # Custom loop to allow to perform backpropagation only every a certain number of batches
    for j in range(backprop_every):
      # logger.debug('----backprop_every = {}'.format(j))

      batch_idx = k + j
      batch_ref_window_size = 0 # :NOTE: added for compatibility reason


      ## Indexing
      start_train_idx = self.batch_begin_inds[batch_idx]
      end_train_idx = self.batch_end_inds[batch_idx]
      # start_train_idx, end_train_idx = self.set_begin_end_batch_idx(self.init_train_data,batch_idx, BATCH_SIZE, batch_ref_window_size)

      end_train_hard_negative_idx = end_train_idx # :NOTE: added for compatibility reason.
      sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, _ = self.set_params_batch(start_train_idx, end_train_idx)

      ## Masking
      # train_mask = full_data.timestamps < full_data.timestamps[end_train_idx]
      left =  full_data.timestamps < full_data.timestamps[end_train_idx]
      right = full_data.timestamps[0] <= full_data.timestamps
      train_mask = np.logical_and(right, left)

      # Initialize training neighbor finder to retrieve temporal graph
      train_data = Data(full_data.sources[train_mask],
                          full_data.destinations[train_mask],
                          full_data.timestamps[train_mask],
                          full_data.edge_idxs[train_mask],
                          full_data.labels[train_mask])
      # print(train_data.n_interactions, train_data.n_unique_nodes)
      train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
      model.set_neighbor_finder(train_ngh_finder)

      ## Sampler
      size = len(sources_batch)
      train_rand_sampler = get_sampler(train_data, batch_ref_window_size, start_train_idx, end_train_hard_negative_idx, neg_sample_method)

      negatives_src_batch, negatives_dst_batch = get_negative_nodes_batch(train_rand_sampler, BATCH_SIZE, size, neg_sample_method)

      pos_label, neg_label = init_pos_neg_labels(size, device)

      criterion = get_criterion()


      pos_prob, neg_prob = compute_edges_probabilities_with_custom_sampled_nodes(model, neg_edges_formation, negatives_dst_batch, negatives_src_batch, sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

      pos_edges_weight, neg_edges_weight = get_edges_weight(train_data,k, BATCH_SIZE,max_weight,start_train_idx, end_train_hard_negative_idx, ef_iwf_window_dict, nf_iwf_window_dict, share_selected_random_weight_per_window_dict, weighted_loss_method, sampled_nodes=negatives_src_batch, compute_xf_iwf_with_sigmoid=compute_xf_iwf_with_sigmoid)

      self.logger_2.info(f'pos_edges_weight = {pos_edges_weight}')
      self.logger_2.info(f'neg_edges_weight = {neg_edges_weight}')

      loss = compute_loss(pos_label, neg_label, pos_prob, neg_prob, pos_edges_weight, neg_edges_weight, batch_idx, criterion, loss, weighted_loss_method)

    loss /= backprop_every

    loss.backward()
    optimizer.step()
    self.m_loss.append(loss.item())

    # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
    # the start of time
    if USE_MEMORY:
      model.memory.detach_memory()

    self.start_train_idx = start_train_idx
    self.end_train_idx = end_train_idx
    self.train_rand_sampler = train_rand_sampler

  def evaluate_epoch(self, ensemble_idx, epoch, num_batch):
    model = self.models[ensemble_idx]["model"]
    args = self.args
    full_data = self.full_data
    # init_train_data = self.init_train_data
    NUM_NEIGHBORS = self.num_neighbors

    BATCH_SIZE = args.bs
    USE_MEMORY = args.use_memory

    start_epoch = time.time()

    ### Training :DOC:
    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      model.memory.__init_memory__()

    for k in range(0, num_batch, args.backprop_every):
    # logger.debug('---batch = {}'.format(k))
      self.evaluate_batch(ensemble_idx, k, args.backprop_every)
      start_train_idx = self.start_train_idx
      end_train_idx = self.end_train_idx
      train_rand_sampler = self.train_rand_sampler

    epoch_time = time.time() - start_epoch
    self.epoch_times.append(epoch_time)

    self.logger.info('start validation...')
    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    ### Validation
    # Validation uses the full graph
    model.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
        # Backup memory at the end of training, so later we can restore it and use it for the
        # validation on unseen nodes
        train_memory_backup = model.memory.backup_memory()

    # VAL_BATCH_SIZE = BATCH_SIZE * 1
    VAL_BATCH_SIZE = self.window_size
    sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, _ = self.set_params_batch(end_train_idx, end_train_idx + VAL_BATCH_SIZE)


    size = len(sources_batch)
    _, negative_samples = train_rand_sampler.sample(size)

    pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, NUM_NEIGHBORS)



    pred_score = np.concatenate([pos_prob.cpu().data.detach().numpy(), neg_prob.cpu().data.detach().numpy()])
    true_label = np.concatenate([np.ones(size), np.zeros(size)])

    self.models[ensemble_idx]["pred_score"] = pred_score.reshape(-1)
    self.models[ensemble_idx]["true_label"] = true_label.reshape(-1)

    val_auc, val_ap = compute_evaluation_score(true_label,pred_score)

    if USE_MEMORY:
        val_memory_backup = model.memory.backup_memory()
        # Restore memory we had at the end of training to be used when validating on new nodes.
        # Also backup memory after validation so it can be used for testing (since test edges are
        # strictly later in time than validation edges)
        model.memory.restore_memory(train_memory_backup)

    total_epoch_time = time.time() - start_epoch
    self.total_epoch_times.append(total_epoch_time)

    self.logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    self.logger.info('Epoch mean loss: {}'.format(np.mean(self.m_loss)))
    self.logger.info(
        'val auc: {}'.format(val_auc))
    self.logger.info(
        'val ap: {}'.format(val_ap))

  def evaluate_ensemble(self, n_ensemble):
    pred_val_list = []
    raw_pred_val_list = []
    for ensemble_idx in range(n_ensemble):
      raw_pred_val = self.models[ensemble_idx]["pred_score"].tolist()
      pred_val = convert_prob_list_to_binary_list(raw_pred_val)
      pred_val_list.append(pred_val)
      raw_pred_val_list.append(raw_pred_val)

    sum_vote = np.sum(np.array(pred_val_list), axis=0)
    mean_vote = np.mean(np.array(raw_pred_val_list), axis=0)

    voted_pred_list = []
    for v in sum_vote:
       voted_pred = 1 if v > n_ensemble/2 else 0
       voted_pred_list.append(voted_pred)

    true_label = self.models[ensemble_idx]["true_label"]
    cm = confusion_matrix(true_label, voted_pred_list)
    precision = compute_precision(cm)
    auc_for_ensemble = compute_auc_for_ensemble(true_label, mean_vote)
    return precision, auc_for_ensemble
    

  def evaluate(self):
    args = self.args
    self.check_point.data = args.data
    self.check_point.prefix = self.prefix
    self.check_point.bs = args.bs
    self.check_point.epoch_max = args.n_epoch
    self.check_point.ws_max = self.total_num_ws
    self.check_point.max_random_weight_range = args.max_random_weight_range

    assert len(self.ensemble_begin_inds) == len(self.ensemble_end_inds)

    n_ensembles = len(self.ensemble_begin_inds)
    for idx in range(n_ensembles):
      self.check_point.ws_idx = 99999
      self.logger.debug('--ensemble_idx = {}'.format(idx))
      self.logger_2.info('--ensemble_idx = {}'.format(idx))
      self.m_loss = []
      self.epoch_times = []
      self.total_epoch_times = []
      self.add_model(idx)

      self.ef_iwf_window_dict.dict_ = {}
      self.nf_iwf_window_dict.dict_ = {}
      self.share_selected_random_weight_per_window_dict.dict_ = {}

      size_of_current_concat_windows = self.ensemble_begin_inds[idx] - self.ensemble_begin_inds[idx] + 1
      num_batch = math.ceil((size_of_current_concat_windows)/args.bs)

      for epoch in range(self.n_epoch):
        self.check_point.epoch_idx = epoch
        self.check_point.get_checkpoint_path()

        self.logger.debug('--epoch = {}'.format(epoch))
        self.logger_2.info('--epoch = {}'.format(epoch))
        self.logger.info(f'max_weight = {self.args.max_random_weight_range}')
        self.evaluate_epoch(idx, epoch, num_batch)

    precision, auc_for_ensemble = self.evaluate_ensemble(n_ensembles)
    self.logger.info(f'precision of enemble = {precision}')
    self.logger.info(f'auc of enemble = {auc_for_ensemble}')


class WindowSlidingForwardNodeClassification(WindowSlidingForward):

  def get_conditions(self):
    self.prefix, self.weighted_loss_method = get_conditions_node_classification(self.args)

  def set_decoder(self):
    args = self.args
    device = self.device

    feat_dim = self.node_features.shape[1]
    n_unique_labels = self.full_data.n_unique_labels

    decoder_optimizer, decoder, decoder_loss_criterion = select_decoder_and_loss_node_classification(args,device,feat_dim, n_unique_labels, self.weighted_loss_method)
    return decoder, decoder_optimizer, decoder_loss_criterion

  def add_model(self):
    self.model = self.ModelClass(**self.model_kwargs)
    self.decoder, self.optimizer, self.criterion = self.set_decoder()
    # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
    self.model.to(self.device)
    begin_ind, idx_to_split, _ = right_split_ind_by_window(1, self.full_data.data_size, self.window_size)

    # self.window_begin_inds, self.window_end_inds = get_all_ensemble_training_data_inds(begin_ind, idx_to_split-1, self.window_size, fix_begin_ind=True)

    self.batch_inds = split_a_window_into_batches(begin_ind, idx_to_split- 1, self.args.bs)
    self.batch_begin_inds = self.batch_inds[:-1]
    self.batch_end_inds = self.batch_inds[1:]

  def set_params_mask(self, start_idx, end_idx ):
    BATCH_SIZE = self.args.bs
    full_data = self.full_data

    time_after_end_of_current_batch = full_data.timestamps > full_data.timestamps[end_idx]
    time_before_end_of_next_batch = full_data.timestamps <= full_data.timestamps[end_idx + BATCH_SIZE]
    mask = np.logical_and(time_after_end_of_current_batch, time_before_end_of_next_batch)

    return mask

  def evaluate_batch(self, model, k, backprop_every):
    args = self.args
    criterion = self.criterion
    full_data = self.full_data
    # init_train_data = self.init_train_data
    NUM_NEIGHBORS = self.num_neighbors
    device = self.device
    decoder_optimizer = self.optimizer
    decoder_loss_criterion = self.criterion
    WINDOW_SIZE = self.window_size

    num_instance = full_data.data_size
    max_weight = args.max_random_weight_range
    BATCH_SIZE = args.bs
    USE_MEMORY = args.use_memory

    n_unique_labels = self.full_data.n_unique_labels
    weighted_loss_method = self.weighted_loss_method

    ef_iwf_window_dict = self.ef_iwf_window_dict.dict_
    nf_iwf_window_dict = self.nf_iwf_window_dict.dict_
    share_selected_random_weight_per_window_dict = self.share_selected_random_weight_per_window_dict.dict_

    loss = 0
    decoder_optimizer.zero_grad()

    # Custom loop to allow to perform backpropagation only every a certain number of batches
    for j in range(args.backprop_every):

      batch_idx = k + j

      # start_train_idx = batch_idx * BATCH_SIZE
      # end_train_idx = min(end_ws_idx, start_train_idx + BATCH_SIZE)
      start_train_idx = self.batch_begin_inds[batch_idx]
      end_train_idx = self.batch_end_inds[batch_idx]

      assert (self.end_ws_idx - self.begin_ws_idx) <= WINDOW_SIZE, "if false, *_batch will encounter out of  bound error. Maybe intial number of data is more than BATCH_SIZE."
      assert start_train_idx < end_train_idx, "number of batch to run for each epoch was not set correctly."
      # assert len(selected_sources_ind) >= end_ws_idx
      # print(start_train_idx, end_train_idx)

      # sources_batch, destinations_batch = full_data.sources[start_train_idx:end_train_idx], \
      #                                     full_data.destinations[start_train_idx:end_train_idx]
      # edge_idxs_batch = full_data.edge_idxs[start_train_idx: end_train_idx]
      # timestamps_batch = full_data.timestamps[start_train_idx:end_train_idx]
      # labels_batch = full_data.labels[start_train_idx:end_train_idx]

      sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, labels_batch = self.set_params_batch(start_train_idx, end_train_idx)
      # total_labels_batch = labels_batch
      self.current_window_labels.extend(labels_batch)

      # size = len(sources_batch)

      source_embedding, destination_embedding, _ = model.compute_temporal_embeddings(sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, sampled_source_nodes=None, n_neighbors=NUM_NEIGHBORS)

      # labels_batch = labels_batch[self.selected_sources_ind]
      # sources_batch = sources_batch[self.selected_sources_ind]

      ## offset by self.selected_sources_ind - relative_batch_idx and select only ind within range of 0 < x < batch_size.
      # get_selected_sources_of_batch_idx_relative_to_window_idx(k, )

      # absolute_window_idx = convert_batch_idx_to_window_idx(absolute_batch_idx, ws_multiplier)
      relative_batch_idx = get_batch_idx_relative_to_window_idx(k, args.ws_multiplier)
      offset_val = relative_batch_idx * BATCH_SIZE

      relative_sources_ind = np.array(self.selected_sources_ind) - offset_val
      left = relative_sources_ind >= 0
      right = relative_sources_ind < BATCH_SIZE
      selected_relative_sources_ind_mask = np.logical_and(left, right)

      selected_relative_sources_ind = relative_sources_ind[selected_relative_sources_ind_mask].tolist()

      ## NOTE: only mask labels (selected from window range) that are within current batch range.
      # labels_batch = labels_batch[self.selected_sources_ind]
      # sources_batch = sources_batch[self.selected_sources_ind]
      labels_batch = labels_batch[selected_relative_sources_ind]
      sources_batch = sources_batch[selected_relative_sources_ind]

      # raise NotImplementedError("added arugment to get_nodes_weight and I haven't test it in node_classification yet.")
      nodes_weight = get_nodes_weight(full_data, batch_idx, BATCH_SIZE, max_weight, start_train_idx, end_train_idx, nf_iwf_window_dict, n_unique_labels, weighted_loss_method, share_selected_random_weight_per_window_dict)

      self.logger_2.info(f'nodes_weight = {nodes_weight}')

      # nodes_weight_batch = nodes_weight[self.selected_sources_ind]
      if nodes_weight is not None:
        nodes_weight_batch = nodes_weight[selected_relative_sources_ind]


      if full_data.n_unique_labels == 2: # :NOTE: for readability, train_data should be replaced by full_data, but I am unsure about side effect.
        raise NotImplementedError
        pred = self.decoder(source_embedding).sigmoid()
        labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
        decoder_loss = decoder_loss_criterion(weight=nodes_weight_batch)(pred, labels_batch_torch)
      elif full_data.n_unique_labels == 4:
        # pred = self.decoder(source_embedding[self.selected_sources_ind]).softmax(dim=1) # :BUG: I am not sure if selected_sources_ind can be appplied here without effecting model learning
        pred = self.decoder(source_embedding[selected_relative_sources_ind]).softmax(dim=1) # :BUG: I am not sure if selected_sources_ind can be appplied here without effecting model learning
        labels_batch_torch = torch.from_numpy(self.onehot_encoder.transform(pd.DataFrame(labels_batch)).toarray()).long().to(device)
        labels_batch_torch = np.argmax(labels_batch_torch, axis=1)
        if nodes_weight is None:
          decoder_loss = decoder_loss_criterion()(pred, labels_batch_torch)
        else:
          decoder_loss = decoder_loss_criterion(weight=nodes_weight_batch)(pred, labels_batch_torch)

      pred = pred_prob_to_pred_labels(pred.cpu().detach().numpy()) # :NOTE: not sure what this is used for.

      loss += decoder_loss.item()

    loss /= args.backprop_every

    decoder_loss.backward()
    decoder_optimizer.step()
    self.m_loss.append(decoder_loss.item())

    # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
    # the start of time
    if USE_MEMORY:
      model.memory.detach_memory()

    self.start_train_idx = start_train_idx
    self.end_train_idx = end_train_idx
    # self.train_rand_sampler = train_rand_sampler
    self.logger.info(
      f"train labels batch distribution = {get_label_distribution(labels_batch)}")
    self.logger.info(
      f"predicted train labels batch distribution = {get_label_distribution(pred)}")
    self.logger.info(
      f"train labels batch distribution (disregard frequency of unique node) = "
      f"{get_label_distribution(get_unique_nodes_labels(labels_batch, sources_batch))}")
    self.logger.info(
      f"predicted train labels batch distribution (disregard frequency of unique node) = "
      f"{get_label_distribution(get_unique_nodes_labels(pred, sources_batch))}") # note that it is possible that model predict different labels for the same nodes. (I will omit this metric until it is shown to be needed.)

  def evaluate_epoch(self, model, epoch, num_batch):
    args = self.args
    full_data = self.full_data
    # init_train_data = self.init_train_data
    NUM_NEIGHBORS = self.num_neighbors
    decoder = self.decoder

    BATCH_SIZE = args.bs
    USE_MEMORY = args.use_memory

    start_epoch = time.time()

    ### Training :DOC:
    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      model.memory.__init_memory__()

    self.decoder = self.decoder.train()
    self.current_window_labels = []

    for k in range(0, num_batch, args.backprop_every):
    # logger.debug('---batch = {}'.format(k))
      self.evaluate_batch(model, k, args.backprop_every)
      start_train_idx = self.start_train_idx
      end_train_idx = self.end_train_idx
      # train_rand_sampler = self.train_rand_sampler

    # self.logger.info(f"total labels batch epoch distribution = {get_label_distribution(self.total_labels_batch)}")
    self.logger.info(f"total labels distribution = {get_label_distribution(self.current_window_labels)}")

    epoch_time = time.time() - start_epoch
    self.epoch_times.append(epoch_time)

    self.logger.info(f'total number of labelled uniqued nodes = {len(self.selected_sources_to_label)}')
    self.logger.info('start validation...')
    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    ### Validation
    if USE_MEMORY:
        # Backup memory at the end of training, so later we can restore it and use it for the
        # validation on unseen nodes
        train_memory_backup = model.memory.backup_memory()

    assert full_data.timestamps.shape[0] >= end_train_idx + BATCH_SIZE


    # VAL_BATCH_SIZE = BATCH_SIZE * 1
    VAL_BATCH_SIZE = self.window_size

    assert full_data.timestamps.shape[0] >= end_train_idx + BATCH_SIZE

    # :DEBUG:
    val_mask = self.set_params_mask(start_train_idx, end_train_idx)
    val_data = Data(full_data.sources[val_mask],
                    full_data.destinations[val_mask],
                    full_data.timestamps[val_mask],
                    full_data.edge_idxs[val_mask],
                    full_data.labels[val_mask],
                    )

    val_auc, val_acc, cm  = my_eval_node_classification(self.logger,
                                                        model,
                                                        decoder,
                                                        val_data,
                                                        VAL_BATCH_SIZE,
                                                        self.selected_sources_to_label,
                                                        n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
        val_memory_backup = model.memory.backup_memory()
        # Restore memory we had at the end of training to be used when validating on new nodes.
        # Also backup memory after validation so it can be used for testing (since test edges are
        # strictly later in time than validation edges)
        model.memory.restore_memory(train_memory_backup)

    total_epoch_time = time.time() - start_epoch
    self.total_epoch_times.append(total_epoch_time)

    self.logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    self.logger.info('Epoch mean loss: {}'.format(np.mean(self.m_loss)))
    # self.logger.info('val ap: {}'.format(val_ap))
    self.logger.info(f'val acc: {val_acc}')
    self.logger.info(f'confusion matrix = \n{cm}')

  def evaluate_ws(self, ws, size_of_current_concat_windows, batch_size):
    full_data = self.full_data

    ## Keeping track of class labels
    self.selected_sources_ind,self.selected_sources_to_label = label_new_unique_nodes_with_budget(self.selected_sources_to_label, full_data, (self.begin_ws_idx, self.end_ws_idx))
    # assert selected_sources_to_label[:len_before] == selected_sources_to_label_before
    assert np.unique(self.selected_sources_to_label).shape[0] == len(self.selected_sources_to_label)

    num_batch = math.ceil((size_of_current_concat_windows)/batch_size)
    self.m_loss = []
    # for epoch in range(NUM_EPOCH):
    self.epoch_times = []
    self.total_epoch_times = []

    for epoch in range(self.n_epoch):
      self.check_point.epoch_idx = epoch
      self.check_point.get_checkpoint_path()

      self.logger.debug('--epoch = {}'.format(epoch))
      self.logger_2.info('--epoch = {}'.format(epoch))
      self.logger.info(f'max_weight = {self.args.max_random_weight_range}')

      self.evaluate_epoch(self.model, epoch, num_batch)

    self.save_checkpoint_per_ws()

    self.init_num_ws += 1
    self.init_train_data = self.init_num_ws * self.num_instances_shift
    self.begin_ws_idx = self.end_ws_idx
    self.end_ws_idx = min(self.init_num_ws * self.num_instances_shift, self.full_data.edge_idxs.shape[0]-1)

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen

    # if USE_MEMORY:
    if self.use_memory:
      val_memory_backup = self.model.memory.backup_memory()

  def pre_evaluation(self):
    self.window_size, self.num_init_data,self.num_instances_shift, self.init_train_data, self.total_num_ws, self.init_num_ws, self.left_num_ws = self.get_sliding_window_params(self.full_data.data_size, self.args.bs, self.args.ws_multiplier)

    self.begin_ws_idx = 0 # pointer for first index of previously added window
    self.end_ws_idx = self.init_train_data # pointer for last index of previously added window

    self.onehot_encoder = get_encoder(self.full_data.n_unique_labels)

    self.add_model()

  def evaluate(self):
    # raise NotImplementedError()

    args = self.args
    self.check_point.data = args.data
    self.check_point.prefix = self.prefix
    self.check_point.bs = args.bs
    self.check_point.epoch_max = args.n_epoch
    self.check_point.ws_max = self.total_num_ws
    self.check_point.max_random_weight_range = args.max_random_weight_range


    for ws in range(self.left_num_ws):
      self.check_point.ws_idx = ws
      self.logger.debug('-ws = {}'.format(ws))
      self.logger_2.info('-ws = {}'.format(ws))
      self.evaluate_ws(ws, self.init_train_data, self.args.bs)
