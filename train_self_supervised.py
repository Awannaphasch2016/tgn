import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction, sliding_window_evaluation, train_val_test_evaluation
from model.tgn import TGN
from utils.utils import EarlyStopMonitor
from utils.data_processing import get_data, compute_time_statistics, Data

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
# parser.add_argument('--bs', type=int, default=1000, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
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
parser.add_argument('--use_ef_iwf_weight', action='store_true',
                    help='use ef_iwf as weight of positive edges in BCE loss')
parser.add_argument('--use_nf_iwf_neg_sampling', action='store_true',
                    help='use nf_iwf to rank user nodes to sample negative edges pair incident to user nodes.')



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
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
# use_weight = True
# use_weight = args.use_ef_iwf_weight
# use_nf_iwf_neg_sampling = args.use_nf_iwf_neg_sampling

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'


### set up logger
def setup_logger(formatter, name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

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

Path("log/").mkdir(parents=True, exist_ok=True)

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

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# Path("log/").mkdir(parents=True, exist_ok=True)
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
  ### Extract data for training, validation and testing
  node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
  new_node_test_data, timestamps, observed_edges_mask = get_data(DATA,
                                different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)


  # # Initialize training neighbor finder to retrieve temporal graph
  # train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

  # # Initialize validation and test neighbor finder to retrieve temporal graph
  # full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

  # # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
  # # across different runs
  # # NB: in the inductive setting, negatives are sampled only amongst other new nodes
  # train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
  # val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
  # nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
  #                                       seed=1)
  # test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
  # nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
  #                                        new_node_test_data.destinations,
  #                                        seed=3)


  # Compute time statistics
  mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)


  for i in range(args.n_runs):
    results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    # tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
    tgn = TGN(neighbor_finder=None, node_features=node_features,
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


    # criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    tgn = tgn.to(device)

    num_instance = len(full_data.sources)
    # num_batch = math.ceil(num_instance / BATCH_SIZE)
    # logger.info('num of training instances: {}'.format(num_instance))
    # logger.info('num of batches per epoch: {}'.format(num_batch))

    # idx_list = np.arange(num_instance)

    new_nodes_val_aps = []
    val_aps = []
    total_epoch_times = []
    train_losses = []
    early_stopper = EarlyStopMonitor(max_round=args.patience)


    logger.info('run = {}'.format(i))

    sliding_window_evaluation(tgn,
                              num_instance,
                              BATCH_SIZE,
                              logger,
                              logger_2,
                              USE_MEMORY,
                              MODEL_SAVE_PATH,
                              args,
                              optimizer,
                              # criterion,
                              full_data,
                              device,
                              NUM_NEIGHBORS,
                              )

    # train_val_test_evaluation(tgn,
    #                           num_instance,
    #                           BATCH_SIZE,
    #                           logger,
    #                           USE_MEMORY,
    #                           MODEL_SAVE_PATH,
    #                           args,
    #                           optimizer,
    #                           criterion,
    #                           train_data,
    #                           full_data,
    #                           val_data,
    #                           test_data,
    #                           device,
    #                           NUM_NEIGHBORS,
    #                           early_stopper,
    #                           NUM_EPOCH,
    #                           new_node_val_data,
    #                           new_node_test_data,
    #                           get_checkpoint_path,
    #                           results_path
    #                           )
