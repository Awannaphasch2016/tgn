import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np

from model.tgn import TGN
from utils.utils import get_neighbor_finder, MLP, MLP_multiple_class, setup_logger
from utils.data_processing import compute_time_statistics, get_data_node_classification
from evaluation.eval_node_classification import train_val_test_evalulation_node_prediction, sliding_window_evaluation_node_prediction
from utils.data_exploration import collect_burstiness_data_over_sliding_window, collect_burstiness_time_data_over_sliding_window

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

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

# anak's argument
# parser.add_argument('--use_ef_iwf_weight', action='store_true',
#                     help='use ef_iwf as weight of positive edges in BCE loss')
# parser.add_argument('--use_nf_iwf_neg_sampling', action='store_true',
#                     help='use nf_iwf to rank user nodes to sample negative edges pair incident to user nodes.')
parser.add_argument('--use_random_weight_to_benchmark_nf_iwf', action='store_true',
                    help='orignal tgn but use random positive weight.')
parser.add_argument('--use_nf_iwf_weight', action='store_true',
                    help='use nf_iwf as weight of nodes')
parser.add_argument('--use_random_weight_to_benchmark_nf_iwf_1', action='store_true',
                    help='orignal tgn but use random positive weight such that all instances in each window shares same weight, but each window will be assigned weight randomly.')


def prep_args():
  try:
    is_running_test = [True if 'pytest' in i else False for i in sys.argv]
    if any(is_running_test):
      args = parser.parse_args([])
    else:
      args = parser.parse_args()
  except:
    parser.print_help()
    sys.exit(0)
  return args

args = prep_args()

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

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}' + '\
  node-classification.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}' + '\
  node-classification.pth'

### set up logger
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

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
# fh.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.WARN)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# ch.setFormatter(formatter)
# logger.addHandler(fh)
# logger.addHandler(ch)
# logger.info(args)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

if __name__ == "__main__":
  full_data, node_features, edge_features, train_data, val_data, test_data = \
    get_data_node_classification(DATA, use_validation=args.use_validation)

  assert full_data.n_unique_labels == train_data.n_unique_labels
  assert train_data.n_unique_labels == test_data.n_unique_labels

  max_idx = max(full_data.unique_nodes)

  train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)


  # Compute time statistics
  mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

  for i in range(args.n_runs):
    results_path = "results/{}_node_classification_{}.pkl".format(args.prefix,
                                                                  i) if i > 0 else "results/{}_node_classification.pkl".format(
      args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    tgn = TGN(neighbor_finder=train_ngh_finder,
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

    tgn = tgn.to(device)

    # num_instance = len(train_data.sources)
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    # print(num_instance, num_batch) # 571580, 5716
    # exit()

    # print(num_instance)
    # print(num_batch)
    # exit()

    logger.debug('Num of training instances: {}'.format(num_instance))
    logger.debug('Num of batches per epoch: {}'.format(num_batch))

    # train_val_test_evalulation_node_prediction(
    #   logger,
    #   MODEL_SAVE_PATH,
    #   tgn,
    #   device,
    #   num_batch,
    #   BATCH_SIZE,
    #   USE_MEMORY,
    #   num_instance,
    #   node_features,
    #   DROP_OUT,
    #   args,
    #   train_data,
    #   val_data,
    #   test_data,
    #   full_data,
    #   NUM_NEIGHBORS,
    #   results_path,
    #   get_checkpoint_path,
    #   DATA
    #   )

    # feat_dim = node_features.shape[1]
    # n_unique_labels = full_data.n_unique_labels
    # decoder_optimizer, decoder, decoder_loss_criterion = select_decoder_and_loss(args,device,feat_dim, n_unique_labels)

    # ## use with pre-training model to substitute prediction head
    # if train_data.n_unique_labels == 2:
    #   decoder = MLP(node_features.shape[1], drop=DROP_OUT)
    #   decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    #   decoder = decoder.to(device)
    #   decoder_loss_criterion = torch.nn.BCELoss()
    # else:
    #   decoder = MLP_multiple_class(node_features.shape[1], full_data.n_unique_labels ,drop=DROP_OUT)
    #   decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    #   decoder = decoder.to(device)
    #   decoder_loss_criterion = torch.nn.CrossEntropyLoss()

    # collect_burstiness_data_over_sliding_window(
    #   logger,
    #   BATCH_SIZE,
    #   args,
    #   full_data,
    #   results_path,
    #   DATA,
    #   )

    # collect_burstiness_data_over_sliding_window(
    #   logger,
    #   BATCH_SIZE,
    #   args,
    #   full_data,
    #   results_path,
    #   DATA,
    #   )

    # pre_training()
    # fine_tuning()
    # test_contrastive_learning()

    logger.info('run = {}'.format(i))

    sliding_window_evaluation_node_prediction(
      logger,
      logger_2,
      MODEL_SAVE_PATH,
      tgn,
      device,
      BATCH_SIZE,
      USE_MEMORY,
      node_features,
      DROP_OUT,
      args,
      train_data,
      val_data,
      test_data,
      full_data,
      NUM_NEIGHBORS,
      results_path,
      get_checkpoint_path,
      DATA,
      # decoder,
      # decoder_optimizer,
      # decoder_loss_criterion,
      NUM_EPOCH
      )

    # train_val_test_evalulation_node_prediction(
    #     logger,
    #     MODEL_SAVE_PATH,
    #     tgn,
    #     device,
    #     num_batch,
    #     BATCH_SIZE,
    #     USE_MEMORY,
    #     num_instance,
    #     node_features,
    #     DROP_OUT,
    #     args,
    #     train_data,
    #     val_data,
    #     test_data,
    #     full_data,
    #     NUM_NEIGHBORS,
    #     results_path,
    #     get_checkpoint_path,
    #     DATA,
    #     decoder,
    #     decoder_optimizer,
    #     decoder_loss_criterion,
    #   )
