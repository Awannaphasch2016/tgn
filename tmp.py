#!/usr/bin/env python3
import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.sliding_window import WindowSlidingForwardNodeClassification
from model.tgn import TGN
from utils.utils import setup_logger, CheckPoint, ArgsContraint, get_neighbor_finder
from utils.data_processing import compute_time_statistics, Data, get_data_node_classification, DataTransformedCollection


from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from train import Train
from args import Args

class NodeClassicationArgs(Args):
  def original_arguments(self, parser):

    ### Argument and global variables
    parser = argparse.ArgumentParser('TGN self-supervised training')
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                        default='wikipedia')
    parser.add_argument('--bs', type=int, default=100, help='Batch_size')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                      'backprop')
    parser.add_argument('--use_memory', action='store_true',
                        help='Whether to augment the model with a node memory')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
      "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
      "mlp", "identity"], help='Type of message function')
    parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                            'aggregator')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                    'each user')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--randomize_features', action='store_true',
                        help='Whether to randomize node features')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--n_neg', type=int, default=1)
    parser.add_argument('--use_validation', action='store_true',
                        help='Whether to use a validation set')
    parser.add_argument('--new_node', action='store_true', help='model new node')
    return parser

  def added_arguments(self, parser):
    # anak's argument
    parser.add_argument('--custom_prefix', type=str, default=None, help='Prefix to name the checkpoints')
    parser.add_argument('--save_checkpoint', action='store_true',
                        help='save checkpoint of this run.')
    parser.add_argument('--use_random_weight_to_benchmark_nf_iwf', action='store_true',
                        help='orignal tgn but use random positive weight.')
    parser.add_argument('--use_nf_iwf_weight', action='store_true',
                        help='use nf_iwf as weight of nodes')
    parser.add_argument('--use_random_weight_to_benchmark_nf_iwf_1', action='store_true',
                        help='orignal tgn but use random positive weight such that all instances in each window shares same weight, but each window will be assigned weight randomly.')
    parser.add_argument('--max_random_weight_range', type=int, default=None,
                        help='maximum range of random weight method')
    parser.add_argument('--ws_multiplier', type=int, default=1, help='value of window_size is a multiple of batch_size')
    parser.add_argument('--ws_framework', type=str, default='forward', help='options of window sliding framework')

    return parser

class NodeClassificationDataTransformedCollection(DataTransformedCollection):
  def get_data(self, data_name, use_validation):
    self.full_data, self.node_features, self.edge_features, self.train_data, self.val_data, self.test_data = get_data_node_classification(data_name, use_validation=use_validation)

class TrainNodeClassification(Train):
  def set_model_save_path(self):
    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}' + '\
      node-classification.pth'
    self.MODEL_SAVE_PATH = MODEL_SAVE_PATH

  def set_loggers(self, logger, logger_2):
    self.logger = logger
    self.logger_2 = logger_2

  def run_model(self):
    args = self.args
    MODEL_SAVE_PATH = self.MODEL_SAVE_PATH

    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    UNIFORM = args.uniform
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_LAYER = 1
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    USE_MEMORY = args.use_memory
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    WINDOW_SIZE = args.ws_multiplier * BATCH_SIZE

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    ### Extract data for training, validation and testing
    # full_data, node_features, edge_features, train_data, val_data, test_data = \
    #   get_data_node_classification(DATA, use_validation=args.use_validation)

    node_classification_data_transformed_collection = NodeClassificationDataTransformedCollection()
    node_classification_data_transformed_collection.get_data(DATA, use_validation=args.use_validation)

    full_data = node_classification_data_transformed_collection.full_data
    node_features = node_classification_data_transformed_collection.node_features
    edge_features = node_classification_data_transformed_collection.edge_features
    train_data = node_classification_data_transformed_collection.train_data
    val_data = node_classification_data_transformed_collection.val_data
    test_data  = node_classification_data_transformed_collection.test_data


    assert full_data.n_unique_labels == train_data.n_unique_labels
    assert train_data.n_unique_labels == test_data.n_unique_labels

    # args_constraint(args.prefix, full_data.data_size, WINDOW_SIZE, BATCH_SIZE, args.backprop_every)
    args_constraint = ArgsContraint()
    args_constraint.args_naming_contraint(args.prefix)
    args_constraint.args_window_sliding_contraint(full_data.data_size, WINDOW_SIZE, BATCH_SIZE)
    args_constraint.args_window_sliding_training(args.backprop_every)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
      compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    check_point = CheckPoint()
    check_point.custom_prefix = args.custom_prefix
    check_point.is_node_classification = True
    check_point.log_timestamp = log_time

    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    for i in range(args.n_runs):
      check_point.run_idx = i
      results_path = "results/{}_node_classification_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}_node_classification.pkl".format(args.prefix)
      Path("results/").mkdir(parents=True, exist_ok=True)
      logger.info('run = {}'.format(i))

      if args.ws_framework == "forward":
        sliding_window = WindowSlidingForwardNodeClassification
      elif args.ws_framework == "ensemble":
        raise NotImplementedError()
        sliding_window = WindowSlidingEnsemble
      else:
        raise NotImplementedError()

      sliding_window = sliding_window(args)
      sliding_window.get_conditions()
      sliding_window.add_data(node_classification_data_transformed_collection)
      sliding_window.add_loggers(logger, logger_2)
      sliding_window.add_checkpoints(check_point)
      sliding_window.add_hardware_params(device)
      # sliding_window.set_model(TGN, neighbor_finder=None, node_features=node_features,
      sliding_window.set_encoder(TGN,
        neighbor_finder=train_ngh_finder,
              node_features=node_features,
              edge_features=edge_features,
              device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS,
              dropout=DROP_OUT,
              use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM,
              memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src,
              std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst,
              std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message)
      sliding_window.add_model_training_params(args.n_epoch)
      sliding_window.add_model_params(NUM_NEIGHBORS, USE_MEMORY)
      # sliding_window.add_model(tgn) # use it to create new model
      sliding_window.init_params_that_tracks_history_of_ws()
      sliding_window.init_params_that_tracks_history_of_ws_node_classification()
      sliding_window.set_sliding_window_framework(args.ws_framework)
      sliding_window.pre_evaluation()
      sliding_window.evaluate()

if __name__ == "__main__":

  node_classification_args = NodeClassicationArgs('TGN supervised training')
  args = node_classification_args.set_args()

  ## set up logger
  logging.basicConfig(level=logging.INFO)

  logger_name = "first_logger"
  log_time = str(time.time())
  log_file_name = 'log/{}.log'.format(log_time)
  log_level = logging.DEBUG
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  logger = setup_logger(formatter, logger_name, log_file_name, level=log_level)

  logger_name = "second_logger"
  log_file_name = 'log/nodes_and_edges_weight/{}.log'.format(log_time)
  log_level = logging.DEBUG
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  logger_2 = setup_logger(formatter, logger_name, log_file_name, level=log_level)

  logger.info(args)
  logger_2.info(args)

  # Train link prediction
  train_node_classification = TrainNodeClassification(args)
  train_node_classification.set_random_seed()
  train_node_classification.set_model_save_path()
  train_node_classification.set_loggers(logger, logger_2)
  train_node_classification.run_model()
