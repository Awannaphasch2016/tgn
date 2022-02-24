import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

# from evaluation.evaluation import eval_edge_prediction, sliding_window_evaluation, train_val_test_evaluation
from evaluation.evaluation import eval_edge_prediction, train_val_test_evaluation
from evaluation.sliding_window import WindowSlidingForward, WindowSlidingEnsemble
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, setup_logger, CheckPoint, ArgsContraint
from utils.data_processing import get_data, compute_time_statistics, Data, DataTransformedCollection


from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from train import Train
from args import Args

def get_param_config(n_epoch, batch_size):
  raise NotImplementedError
  config = {}
  config["n_epoch"] = n_epoch
  config["batch_size"] = batch_size
  return config

class LinkPredictionArgs(Args):
  def original_arguments(self, parser):
    ### Argument and global variables
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                        default='wikipedia')
    parser.add_argument('--bs', type=int, default=200, help='Batch_size')
    # parser.add_argument('--bs', type=int, default=1000, help='Batch_size')
    parser.add_argument('--prefix', type=str, default=None, help='Deprecated: Prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
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
    parser.add_argument('--memory_updater', type=str, default="gru", choices=[
        "gru", "rnn"], help='Type of memory updater')
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
    parser.add_argument('--dyrep', action='store_true',
                        help='Whether to run the dyrep model')
    return parser


  def added_arguments(self, parser):
    # anak's argument
    parser.add_argument('--custom_prefix', type=str, default=None, help='Prefix to name the checkpoints')
    parser.add_argument('--save_checkpoint', action='store_true',
                        help='save checkpoint of this run.')
    parser.add_argument('--use_ef_iwf_weight', action='store_true', help='use ef_iwf as weight of positive edges in BCE loss')
    parser.add_argument('--use_nf_iwf_weight', action='store_true', help='use nf_iwf as weight of positive edges in BCE loss')
    parser.add_argument('--use_ef_weight', action='store_true',
                        help='use ef as weight of positive edges in BCE loss')

    parser.add_argument('--use_nf_weight', action='store_true', help='use nf as weight of positive edges in BCE loss')
    parser.add_argument('--use_nf_iwf_neg_sampling', action='store_true',
                        help='use nf_iwf to rank user nodes to sample negative edges pair incident to user nodes.')
    parser.add_argument('--use_sigmoid_ef_iwf_weight', action='store_true',
                        help='same as --use_ef_iwf_weight, but sigmoid is applied to compute ef_iwf ')
    parser.add_argument('--use_random_weight_to_benchmark_ef_iwf', action='store_true',
                        help='orignal tgn but use random positive weight such tha new random weight from given range are generated for a new window')
    parser.add_argument('--use_random_weight_to_benchmark_ef_iwf_1', action='store_true',
                        help='orignal tgn but use random positive weight such that all instances in each window shares same weight, but each window will be assigned weight randomly.')
    parser.add_argument('--run_tuning', action='store_true',
                        help='run hyperparameter tuning.')
    parser.add_argument('--n_tuning_samples', type=int, default=3,
                        help='number of time to draw sample from hyperparameter spaces. ')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='number of gpu to use')
    parser.add_argument('--max_random_weight_range', type=int, default=None,
                        help='maximum range of random weight method')
    parser.add_argument('--ws_multiplier', type=int, default=1, help='value of window_size is a multiple of batch_size')
    parser.add_argument('--ws_framework', type=str, default='forward', help='options of window sliding framework')
    parser.add_argument('--edge_weight_multiplier', type=float, default=1, help='edge_weight_multiplier is used to multiply ef value.')
    parser.add_argument('--use_time_decay', action='store_true', help='apply time decay to xf_iwf.')
    parser.add_argument('--use_time_decay_multiplier', action='store_true', help='apply time decay to unweighted edges.')
    # parser.add_argument('--keep_last_n_window_as_window_slides', action='store_true', help='only keep last window of sliding window as window slides.')
    parser.add_argument('--keep_last_n_window_as_window_slides', type=int, default=None, help='only keep last n window of sliding window as window slides.')
    parser.add_argument('--window_stride_multiplier', type=int, default=1, help='window_stride_multiplier * window_size == window_stride')
    # parser.add_argument('--last_instances_idx', type=int, default=None, help='index of last instances to be run')
    # parser.add_argument('--first_instances_idx', type=int, default=None, help='index of first instances to be run')
    # parser.add_argument('--first_batch_idx', type=int, default=None, help='first batch idx to be run')
    # parser.add_argument('--last_batch_idx', type=int, default=None, help='last batch idx to be run')
    parser.add_argument('--window_idx_to_start_with', type=int, default=None, help='idx of window to start the training. idx start from 0.')
    parser.add_argument('--init_n_instances_as_multiple_of_ws', type=int, default=None, help='initial number of instances as multiple of window size. (currently only support for ensemble)')
    parser.add_argument('--disable_cuda', action='store_true', help='disable cuda')
    parser.add_argument('--fix_begin_data_ind_of_models_in_ensemble', action='store_true', help='data of ensembles models are started at the same index and starting index of all models remain the same as window slides forward.') # :NOTE: I may need to change this argument to follow my sliding window diagram i drew in draw.io.

    return parser

class LinkPredictionDataTransformedCollection(DataTransformedCollection):

  def get_data(self, data_name, different_new_nodes_between_val_and_test, randomize_features):
    self.node_features, self.edge_features, self.full_data, self.train_data, self.val_data, self.test_data, self.new_node_val_data, self.new_node_test_data, self.timestamps, self.observed_edges_mask = get_data(data_name, different_new_nodes_between_val_and_test=different_new_nodes_between_val_and_test, randomize_features=randomize_features)

class TrainLinkPrediction(Train):
  def set_model_save_path(self):
    # model_save_path
    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
    self.MODEL_SAVE_PATH = MODEL_SAVE_PATH

  def set_loggers_class(self, l_1, l_2, l_3):
    self.l_1 = l_1
    self.l_2 = l_2
    self.l_3 = l_3
    self.set_loggers(l_1.logger, l_2.logger, l_3.logger)


  def set_loggers(self, logger, logger_2, logger_3):
    self.logger = logger
    self.logger_2 = logger_2
    self.logger_3 = logger_3

  def run_model(self):
    args = self.args
    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    USE_MEMORY = args.use_memory
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    WINDOW_SIZE = args.ws_multiplier * BATCH_SIZE
    MODEL_SAVE_PATH = self.MODEL_SAVE_PATH

    args = self.args
    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() and not args.disable_cuda else 'cpu'
    device = torch.device(device_string)


    ### Extract data for training, validation and testing
    link_prediction_data_transformed_collection = LinkPredictionDataTransformedCollection()
    link_prediction_data_transformed_collection.get_data(DATA, different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

    # node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
    # new_node_test_data, timestamps, observed_edges_mask = get_data(DATA,
    #                               different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

    full_data = link_prediction_data_transformed_collection.full_data
    edge_features = link_prediction_data_transformed_collection.edge_features
    node_features = link_prediction_data_transformed_collection.node_features

    # args_constraint(args.prefix, full_data.data_size, WINDOW_SIZE, BATCH_SIZE, args.backprop_every)
    args_constraint = ArgsContraint()
    args_constraint.args_naming_contraint(args.prefix)
    args_constraint.args_window_sliding_contraint(full_data.data_size, WINDOW_SIZE, BATCH_SIZE)
    args_constraint.args_window_sliding_training(args.backprop_every)
    args_constraint.args_setting_init_n_instances_as_multiple_of_ws_constraint(args.init_n_instances_as_multiple_of_ws, args.ws_framework)
    args_constraint.args_fix_begin_data_ind_of_models_in_ensemble(args.fix_begin_data_ind_of_models_in_ensemble, args.ws_framework)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
      compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    check_point = CheckPoint()
    check_point.custom_prefix = args.custom_prefix
    check_point.is_node_classification = False
    check_point.log_timestamp = log_time

    for i in range(args.n_runs):
      check_point.run_idx = i
      results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
      Path("results/").mkdir(parents=True, exist_ok=True)
      self.logger.info('run = {}'.format(i))

      if args.ws_framework == "forward":
        sliding_window = WindowSlidingForward
      elif args.ws_framework == "ensemble":
        sliding_window = WindowSlidingEnsemble
      else:
        raise NotImplementedError()

      sliding_window = sliding_window(args)
      sliding_window.set_run_idx(i)
      sliding_window.get_conditions()
      sliding_window.set_sliding_window_params(args.keep_last_n_window_as_window_slides)
      # sliding_window.add_dataset(full_data)
      sliding_window.add_data(link_prediction_data_transformed_collection)
      # sliding_window.add_weight_observer()
      sliding_window.add_observers()
      sliding_window.add_loggers_class(l_1, l_2, l_3)
      # sliding_window.add_loggers(logger, logger_2)
      sliding_window.add_checkpoints(check_point)
      sliding_window.add_hardware_params(device)
      # sliding_window.set_model(TGN, neighbor_finder=None, node_features=node_features,
      sliding_window.set_encoder(TGN, neighbor_finder=None, node_features=node_features,
                edge_features=edge_features, device=device,
                n_layers=NUM_LAYER,
                n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                memory_update_at_start=not args.memory_update_at_end,
                embedding_module_type=args.embedding_module,
                message_function=args.message_function,
                aggregator_type=args.aggregator,
                memory_updater_type=args.memory_updater,
                n_neighbors=NUM_NEIGHBORS,
                mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                use_source_embedding_in_message=args.use_source_embedding_in_message,
                dyrep=args.dyrep)
      sliding_window.add_model_training_params(args.n_epoch)
      sliding_window.add_model_params(NUM_NEIGHBORS, USE_MEMORY)
      # sliding_window.add_model(tgn) # use it to create new model
      sliding_window.init_params_that_tracks_history_of_ws()
      sliding_window.set_sliding_window_framework(args.ws_framework)
      sliding_window.pre_evaluation()
      sliding_window.evaluate()

class Logger:
  def set_logger_params(self, formatter, logger_name, log_time, log_relative_path, log_file_name, log_level):
    self.formatter =  formatter
    self.logger_name =  logger_name
    self.log_file_name = log_file_name
    self.log_level = log_level
    self.log_time = log_time
    self.log_relative_path = log_relative_path

  def set_relative_path(self, relative_path):
    self.relative_path = relative_path

  def setup_logger(self):
    self.logger = setup_logger(self.formatter, self.logger_name, self.log_file_name, level=self.log_level)

if __name__ == "__main__":

  link_prediction_args = LinkPredictionArgs('TGN self-supervised training')
  args = link_prediction_args.set_args()

  # set up logger
  logging.basicConfig(level=logging.INFO)

  l_1  = Logger()
  logger_name = "first_logger"
  log_time = str(time.time())
  log_relative_path = 'log/'
  log_file_name = str(Path(log_relative_path) / '{}.log'.format(log_time))
  log_level = logging.DEBUG
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  l_1.set_logger_params(formatter, logger_name, log_time, log_relative_path, log_file_name, log_level)
  l_1.setup_logger()
  # logger = setup_logger(formatter, logger_name, log_file_name, level=log_level)

  l_2  = Logger()
  logger_name = "second_logger"
  log_relative_path = 'log/nodes_and_edges_weight/'
  # log_file_name = 'log/nodes_and_edges_weight/{}.log'.format(log_time)
  log_file_name = str(Path(log_relative_path) / '{}.log'.format(log_time))
  log_level = logging.DEBUG
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  l_2.set_logger_params(formatter, logger_name, log_time, log_relative_path, log_file_name, log_level)
  l_2.setup_logger()
  # logger_2 = setup_logger(formatter, logger_name, log_file_name, level=log_level)

  l_3  = Logger()
  logger_name = "third_logger"
  log_relative_path = 'log/ensembles/'
  # log_file_name = 'log/nodes_and_edges_weight/{}.log'.format(log_time)
  log_file_name = str(Path(log_relative_path) / '{}.log'.format(log_time))
  log_level = logging.DEBUG
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  l_3.set_logger_params(formatter, logger_name, log_time, log_relative_path, log_file_name, log_level)
  l_3.setup_logger()
  # logger_2 = setup_logger(formatter, logger_name, log_file_name, level=log_level)

  l_1.logger.info(args)
  l_2.logger.info(args)
  l_3.logger.info(args)


  # Train link prediction
  train_link_prediction = TrainLinkPrediction(args)
  train_link_prediction.set_random_seed()
  train_link_prediction.set_model_save_path()
  # train_link_prediction.set_loggers(logger, logger_2)
  train_link_prediction.set_loggers_class(l_1, l_2, l_3)
  train_link_prediction.run_model()
