#!/usr/bin/env python3

import math
from evaluation.eval_node_classification import eval_node_classification
import time
import numpy as np
import torch

def collect_burstiness_time_data_over_sliding_window(
    logger,
    BATCH_SIZE,
    args,
    full_data,
    results_path,
    DATA,
    ):
  num_instance = len(full_data.sources)

  epoch_times = []
  total_epoch_times = []

  end_train_idx = None

  BATCH_SIZE = 100
  num_instances_shift = BATCH_SIZE * 1 # 100

  total_num_ws =  math.ceil(num_instance/num_instances_shift) # 6

  node_count_per_window = {ii + 1 : 0 for ii in range(full_data.n_unique_nodes)}

  for ws in range(total_num_ws):

    # start_train_idx = ws * BATCH_SIZE
    start_train_idx = ws * num_instances_shift
    end_train_idx = min(num_instance ,start_train_idx + BATCH_SIZE)

    # if ws == 3:
    #   exit()

    assert start_train_idx < end_train_idx, "number of batch to run for each epoch was not set correctly."

    sources_batch, destinations_batch = full_data.sources[start_train_idx:end_train_idx], \
                                        full_data.destinations[start_train_idx:end_train_idx]
    edge_idxs_batch = full_data.edge_idxs[start_train_idx: end_train_idx]
    timestamps_batch = full_data.timestamps[start_train_idx:end_train_idx]
    labels_batch = full_data.labels[start_train_idx:end_train_idx]

    for src, des in zip(sources_batch, destinations_batch):
      node_count_per_window[src] += 1
      node_count_per_window[des] += 1

    nodes_batch = np.concatenate((sources_batch, destinations_batch), axis=None)
    _ , src_unique_count = np.unique(sources_batch, return_counts=True )
    _ , des_unique_count = np.unique(destinations_batch, return_counts=True )
    unique_nodes , nodes_unique_count = np.unique(nodes_batch, return_counts=True )

    print(unique_nodes)
    logger.info(f"number of sources nodes appear is {len(set(sources_batch))}")
    logger.info(f"number of destination nodes appear is {len(set(destinations_batch))}")
    logger.info(f"list of sorted count of unique nodes is {sorted(nodes_unique_count)[::-1]}")

    logger.debug('-ws = {}'.format(ws))


def collect_burstiness_data_over_sliding_window(
    logger,
    BATCH_SIZE,
    args,
    full_data,
    results_path,
    DATA,
    ):
  num_instance = len(full_data.sources)

  epoch_times = []
  total_epoch_times = []

  end_train_idx = None


  BATCH_SIZE = 5000
  num_instances_shift = BATCH_SIZE * 1 # 100


  total_num_ws =  math.ceil(num_instance/num_instances_shift) # 6

  node_count_per_window = {ii + 1 : 0 for ii in range(full_data.n_unique_nodes)}
  # print(max(item_count_per_window.keys()))
  # exit()

  for ws in range(total_num_ws):

    start_train_idx = ws * BATCH_SIZE
    end_train_idx = min(num_instance ,start_train_idx + BATCH_SIZE)
    item_count_per_window = {ii + full_data.n_unique_sources + 1 : 0 for ii in range(full_data.n_unique_destinations)}
    user_count_per_window = {ii + 1: 0 for ii in range(full_data.n_unique_sources)}

    # print(item_count_per_window.values())
    # exit()

    # if ws == 3:
    #   exit()

    assert start_train_idx < end_train_idx, "number of batch to run for each epoch was not set correctly."

    sources_batch, destinations_batch = full_data.sources[start_train_idx:end_train_idx], \
                                        full_data.destinations[start_train_idx:end_train_idx]
    edge_idxs_batch = full_data.edge_idxs[start_train_idx: end_train_idx]
    timestamps_batch = full_data.timestamps[start_train_idx:end_train_idx]
    labels_batch = full_data.labels[start_train_idx:end_train_idx]

    items_count = sum(list(item_count_per_window.values()))

    assert items_count == 0, f"make sure all items count are 0 because moving on. item_count = {item_count}\nitem_count_per_window.values()"

    # print(len(sources_batch))
    for src, des in zip(sources_batch, destinations_batch):
      node_count_per_window[src] += 1
      node_count_per_window[des] += 1

      item_count_per_window[des] += 1
      user_count_per_window[src] += 1

      # if des == 10984:
      #   print(ws)
      #   print(start_train_idx, end_train_idx)
      #   logger.info(f"reddit post count = {list(item_count_per_window.values())} ")
      #   exit()


    # items_count = sum(list(item_count_per_window.values()))
    # if BATCH_SIZE != items_count:
    #   print(f'items are not correctly counted per BATCH. count items = {items_count}')
    #   logger.info(f"reddit post count = {list(item_count_per_window.values())} ")
    #   exit()


    nodes_batch = np.concatenate((sources_batch, destinations_batch), axis=None)
    _ , src_unique_count = np.unique(sources_batch, return_counts=True )
    _ , des_unique_count = np.unique(destinations_batch, return_counts=True )
    unique_nodes , nodes_unique_count = np.unique(nodes_batch, return_counts=True )

    # for post, post_count in item_count_per_window.items():
    #   logger.info(f"reddit post {post} has item count = {post_count} ")

    logger.info(f"reddit post count = {list(item_count_per_window.values())} ")
    logger.info(f"reddit user count = {list(user_count_per_window.values())} ")
    logger.info(f"number of sources nodes appear is {len(set(sources_batch))}")
    logger.info(f"number of destination nodes appear is {len(set(destinations_batch))}")
    # logger.info(f"list of sorted count of unique nodes is {sorted(nodes_unique_count)[::-1]}")
    # exit()

    logger.debug('-ws = {}'.format(ws))
