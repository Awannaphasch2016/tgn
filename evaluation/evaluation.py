import math

import numpy as np
import time
import pickle
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.utils import EarlyStopMonitor, get_neighbor_finder, compute_xf_iwf, compute_nf,  get_nf_iwf, add_only_new_values_of_new_window_to_dict, compute_share_selected_random_weight_per_window, get_share_selected_random_weight_per_window, compute_ef,compute_time_decay_multipler
from utils.sampler import RandEdgeSampler, EdgeSampler_NF_IWF
from utils.data_processing import Data
from tqdm import tqdm
import random

def compute_false_positive(confusion_matrix ):
  FP = confusion_matrix[1,0]
  return FP

def compute_false_negative(confusion_matrix):
  TP = confusion_matrix[0,1]
  return FN

def compute_true_positive(confusion_matrix):
  TP = confusion_matrix[0,0]
  return TP

def compute_true_negative(confusion_matrix):
  TN = confusion_matrix[1,1]
  return TN

def compute_precision(confusion_matrix):
  TP = compute_true_positive(confusion_matrix)
  FP = compute_false_positive(confusion_matrix)
  return TP/(TP+FP)

def compute_auc_for_ensemble(true_label, mean_pred_score):
  """
  :NOTE: I didn't look at literature on how to compute auc for ensemblel. Just trying out things
  """
  return roc_auc_score(true_label, mean_pred_score)

def compute_evaluation_score(true_label, pred_score):
  """
  NOTE: 1/25/2022 I just realised that I switch value of auc and ap. I decided not to change it back just because it would be easier to compare and retreive auc and ap value from log
  """

  # pred_score = np.concatenate([pos_prob.cpu().data.detach().numpy(), neg_prob.cpu().data.detach().numpy()])
  # true_label = np.concatenate([np.ones(size), np.zeros(size)])
  auc = average_precision_score(true_label, pred_score)
  ap = roc_auc_score(true_label, pred_score)

  return auc, ap


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)


def train_val_test_evaluation(tgn,
                            num_instance,
                            BATCH_SIZE,
                            logger,
                            USE_MEMORY,
                            MODEL_SAVE_PATH,
                            args,
                            optimizer,
                            criterion,
                            train_data,
                            full_data,
                            val_data,
                            test_data,
                            device,
                            NUM_NEIGHBORS,
                            early_stopper,
                            NUM_EPOCH,
                            new_node_val_data,
                            new_node_test_data,
                            get_checkpoint_path,
                            results_path):

  train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
  full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

  raise NotImplementedError("haven't yet implement RandEdgeSampler with negative sampling")
  train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
  val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
  nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                        seed=1)
  test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
  nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                         new_node_test_data.destinations,
                                         seed=3)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  # print(num_instance, num_batch) # 389989, 1950

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)


  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)

  for epoch in range(NUM_EPOCH):
  # for epoch in range(1):
    start_epoch = time.time()
    ### Training :OC:

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    for k in tqdm(range(0, num_batch, args.backprop_every)):
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(sources_batch)
        _, negatives_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        tgn = tgn.train()
        pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()

    val_ap, val_auc = eval_edge_prediction(model=tgn,
                                                            negative_edge_sampler=val_rand_sampler,
                                                            data=val_data,
                                                            n_neighbors=NUM_NEIGHBORS)
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes
    nn_val_ap, nn_val_auc = eval_edge_prediction(model=tgn,
                                                                        negative_edge_sampler=val_rand_sampler,
                                                                        data=new_node_val_data,
                                                                        n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
      # Restore memory we had at the end of validation
      tgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info(
      'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

    # Early stopping
    if early_stopper.early_stop_check(val_ap):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  ### Test
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc = eval_edge_prediction(model=tgn,
                                                              negative_edge_sampler=test_rand_sampler,
                                                              data=test_data,
                                                              n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  # Test on unseen nodes
  nn_test_ap, nn_test_auc = eval_edge_prediction(model=tgn,
                                                                          negative_edge_sampler=nn_test_rand_sampler,
                                                                          data=new_node_test_data,
                                                                          n_neighbors=NUM_NEIGHBORS)

  logger.info(
    'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
  logger.info(
    'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving TGN model')
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')

def get_sampler(data, batch_ref_window_size, start_train_idx, end_train_hard_negative_idx, neg_sample_method):
  if neg_sample_method == "nf_iwf":
    # sampler = RandEdgeSampler_v2(data.sources, data.destinations, data.edge_idxs, batch_ref_window_size)
    sampler = EdgeSampler_NF_IWF(data.sources, data.destinations, data.edge_idxs, batch_ref_window_size, start_train_idx, end_train_hard_negative_idx)
  elif neg_sample_method == 'random':
    sampler = RandEdgeSampler(data.sources, data.destinations, data.edge_idxs, batch_ref_window_size, start_train_idx, end_train_hard_negative_idx)
  else:
    raise NotImplementedError()

  return sampler

def get_negative_nodes_batch(sampler, batch_size, size, neg_sample_method):
  if neg_sample_method == "nf_iwf":
    # raise NotImplemented
    # neg_src_batch, neg_dst_batch = sampler.sample(size)
    neg_src_batch, neg_dst_batch = sampler.sample_nf_iwf(batch_size, size)
  elif neg_sample_method == "random":
    neg_src_batch, neg_dst_batch = sampler.sample(size)
  else:
    raise NotImplementedError

  return neg_src_batch, neg_dst_batch

def init_pos_neg_labels(size, device):
  with torch.no_grad():
    pos_label = torch.ones(size, dtype=torch.float, device=device)
    neg_label = torch.zeros(size, dtype=torch.float, device=device)
  return pos_label, neg_label

def get_criterion():
  criterion = torch.nn.BCELoss
  return criterion

# def get_nf_iwf(sampler, data, batch_idx, batch_size, start_train_idx, end_train_hard_negative_idx, nf_iwf_window_dict, use_nf_iwf_neg_sampling=False):
#   if use_nf_iwf_neg_sampling:
#     return sampler.sample_nf_iwf(batch_size, size)
#   else;
#     return None, None

def get_edges_weight(data, batch_idx, batch_size, max_weight,start_train_idx, end_train_hard_negative_idx, nf_iwf_window_dict, ef_iwf_window_dict, share_selected_random_weight_per_window_dict, weighted_loss_method, sampled_nodes=None, compute_xf_iwf_with_sigmoid=False, edge_weight_multiplier=None, use_time_decay=False, time_diffs=None):

  pos_edges_weight = None
  neg_edges_weight = None

  if weighted_loss_method == "ef_iwf_as_pos_edges_weight":
    pos_edges_weight = get_ef_iwf(data, batch_idx, batch_size,start_train_idx, end_train_hard_negative_idx, ef_iwf_window_dict,compute_xf_iwf_with_sigmoid=compute_xf_iwf_with_sigmoid, edge_weight_multiplier=edge_weight_multiplier, use_time_decay=use_time_decay, time_diffs=time_diffs)
  elif weighted_loss_method == "nf_iwf_as_pos_edges_weight":
    assert not compute_xf_iwf_with_sigmoid
    pos_edges_weight = get_nf_iwf(data, batch_idx, batch_size, start_train_idx, end_train_hard_negative_idx, nf_iwf_window_dict, edge_weight_multiplier=edge_weight_multiplier, use_time_decay=use_time_decay, time_diffs=time_diffs)
  elif weighted_loss_method == "nf_iwf_as_pos_and_neg_edge_weight":
    raise NotImplementedError() # not sure what this is for.
    # use nodes as edges weight
    pos_edges_weight = get_nf_iwf(data, batch_idx, batch_size, start_train_idx, end_train_hard_negative_idx, nf_iwf_window_dict)
    # get neg_edges_weight from sampled_nodes.
    neg_edges_weight = get_nf_iwf(data, batch_idx, batch_size, start_train_idx, end_train_hard_negative_idx, nf_iwf_window_dict, sampled_nodes=sampled_nodes)
  elif weighted_loss_method == "random_as_pos_edges_weight":
    # rand_weight = np.array([random.randint(0,500)/100 for _ in range(batch_size)]) # range = [0,5]
    rand_weight = np.array([random.randint(0,500) for _ in range(batch_size)]) # range =[0,500]
    rand_weight = torch.FloatTensor(rand_weight)
    pos_edges_weight = rand_weight
  elif weighted_loss_method == "share_selected_random_weight_per_window":
    # rand_weight = np.array([random.randint(0,500) for _ in range(batch_size)]) # range =[0,500]
    pos_edges_weight = get_share_selected_random_weight_per_window(batch_size, max_weight, batch_idx, share_selected_random_weight_per_window_dict)
  elif weighted_loss_method == "ef_as_pos_edges_weight":
    pos_edges_weight = get_ef(data, batch_idx, batch_size,start_train_idx, end_train_hard_negative_idx, ef_iwf_window_dict,compute_xf_iwf_with_sigmoid=compute_xf_iwf_with_sigmoid, edge_weight_multiplier=edge_weight_multiplier, use_time_decay=use_time_decay, time_diffs=time_diffs)
  elif weighted_loss_method == "apply_time_decay_to_non_weighted_edges":
    assert time_diffs is not None
    pos_edges_weight = get_unweighted_edges_with_time_decay(data, batch_idx, batch_size,start_train_idx, end_train_hard_negative_idx, ef_iwf_window_dict, edge_weight_multiplier=edge_weight_multiplier, time_diffs=time_diffs)
  elif weighted_loss_method == "no_weight":
    pass
  else:
    raise NotImplementedError()

  return pos_edges_weight, neg_edges_weight

def get_unweighted_edges_with_time_decay(data, batch_idx, batch_size, start_train_idx, end_train_hard_negative_idx, time_decay_multipler_dict, edge_weight_multiplier=None, time_diffs=None):

  edges_ = np.vstack((data.sources, data.destinations)).T
  start_past_window_idx = start_train_idx
  end_past_window_idx = end_train_hard_negative_idx
  edges_in_past_windows = edges_[:start_past_window_idx]
  edges_in_current_window = edges_[start_past_window_idx:end_past_window_idx]
  pos_edges_weight = []

  time_decay_multipler_window_dict = add_only_new_values_of_new_window_to_dict(compute_time_decay_multipler, edges_in_current_window, batch_size, return_x_value_dict=True, compute_as_nodes=False, time_diffs=time_diffs)(
    batch_idx, time_decay_multipler_dict, 1)


  # if batch_idx not in ef_iwf_window_dict:
  #   # if batch_idx > 1 and use_ef_iwf_weight:
  #   # if batch_idx >= 0:
  #   ef_iwf, edges_to_ef_iwf_current_window_dict = compute_xf_iwf(edges_in_past_windows, edges_in_current_window , batch_size, compute_as_nodes=False, return_x_value_dict=True, compute_with_sigmoid=compute_xf_iwf_with_sigmoid)
  #   ef_iwf_window_dict[batch_idx] = edges_to_ef_iwf_current_window_dict

  for ii in edges_in_current_window:
    pos_edges_weight.append(time_decay_multipler_dict[batch_idx][tuple(ii)])
  assert len(pos_edges_weight) == edges_in_current_window.shape[0]

  pos_edges_weight = torch.FloatTensor(pos_edges_weight)

  return pos_edges_weight

def get_ef(data, batch_idx, batch_size, start_train_idx, end_train_hard_negative_idx, ef_window_dict, compute_xf_iwf_with_sigmoid=False, edge_weight_multiplier=None, use_time_decay=False, time_diffs=None):
  edges_ = np.vstack((data.sources, data.destinations)).T
  start_past_window_idx = start_train_idx
  end_past_window_idx = end_train_hard_negative_idx
  edges_in_past_windows = edges_[:start_past_window_idx]
  edges_in_current_window = edges_[start_past_window_idx:end_past_window_idx]
  pos_edges_weight = []

  ef_window_dict = add_only_new_values_of_new_window_to_dict(compute_ef, edges_in_current_window, batch_size, edge_weight_multiplier=edge_weight_multiplier, return_x_value_dict=True, time_diffs=time_diffs, use_time_decay=use_time_decay)(
    batch_idx, ef_window_dict, 1)


  # if batch_idx not in ef_iwf_window_dict:
  #   # if batch_idx > 1 and use_ef_iwf_weight:
  #   # if batch_idx >= 0:
  #   ef_iwf, edges_to_ef_iwf_current_window_dict = compute_xf_iwf(edges_in_past_windows, edges_in_current_window , batch_size, compute_as_nodes=False, return_x_value_dict=True, compute_with_sigmoid=compute_xf_iwf_with_sigmoid)
  #   ef_iwf_window_dict[batch_idx] = edges_to_ef_iwf_current_window_dict

  for ii in edges_in_current_window:
    pos_edges_weight.append(ef_window_dict[batch_idx][tuple(ii)])
  assert len(pos_edges_weight) == edges_in_current_window.shape[0]

  pos_edges_weight = torch.FloatTensor(pos_edges_weight)

  return pos_edges_weight

def get_ef_iwf(data, batch_idx, batch_size, start_train_idx, end_train_hard_negative_idx, ef_iwf_window_dict, compute_xf_iwf_with_sigmoid=False, edge_weight_multiplier=None, use_time_decay=False, time_diffs=None):

  edges_ = np.vstack((data.sources, data.destinations)).T
  start_past_window_idx = start_train_idx
  end_past_window_idx = end_train_hard_negative_idx
  edges_in_past_windows = edges_[:start_past_window_idx]
  edges_in_current_window = edges_[start_past_window_idx:end_past_window_idx]
  pos_edges_weight = []

  ef_iwf_window_dict = add_only_new_values_of_new_window_to_dict(compute_xf_iwf, edges_in_past_windows, edges_in_current_window , batch_size, compute_as_nodes=False, return_x_value_dict=True, compute_with_sigmoid=compute_xf_iwf_with_sigmoid, edge_weight_multiplier=edge_weight_multiplier, use_time_decay=use_time_decay, time_diffs=time_diffs)(
    batch_idx, ef_iwf_window_dict, 1)


  # if batch_idx not in ef_iwf_window_dict:
  #   # if batch_idx > 1 and use_ef_iwf_weight:
  #   # if batch_idx >= 0:
  #   ef_iwf, edges_to_ef_iwf_current_window_dict = compute_xf_iwf(edges_in_past_windows, edges_in_current_window , batch_size, compute_as_nodes=False, return_x_value_dict=True, compute_with_sigmoid=compute_xf_iwf_with_sigmoid)
  #   ef_iwf_window_dict[batch_idx] = edges_to_ef_iwf_current_window_dict

  for ii in edges_in_current_window:
    pos_edges_weight.append(ef_iwf_window_dict[batch_idx][tuple(ii)])
  assert len(pos_edges_weight) == edges_in_current_window.shape[0]

  pos_edges_weight = torch.FloatTensor(pos_edges_weight)

  return pos_edges_weight

def compute_loss(pos_label, neg_label, pos_prob, neg_prob, pos_edges_weight, neg_edges_weight,batch_idx, criterion, loss, weighted_loss_method):

  # if batch_idx > 1 and use_ef_iwf_weight:
  # if weighted_loss_method == "ef_iwf_as_pos_edges_weight":
  if weighted_loss_method in ["ef_iwf_as_pos_edges_weight",  "ef_as_pos_edges_weight",  "apply_time_decay_to_non_weighted_edges", "nf_iwf_as_pos_edges_weight"]:
    assert neg_edges_weight is None
    assert pos_edges_weight is not None
    if batch_idx >= 0:
      loss += criterion(weight=pos_edges_weight)(pos_prob.squeeze(), pos_label)+criterion()(neg_prob.squeeze(), neg_label)
    else:
      loss += criterion()(pos_prob.squeeze(), pos_label) + criterion()(neg_prob.squeeze(), neg_label)
  elif weighted_loss_method == "nf_iwf_as_pos_and_neg_edge_weight":
    assert neg_edges_weight is not None
    assert pos_edges_weight is not None
    # raise NotImplementedError()
    loss += criterion(weight=pos_edges_weight)(pos_prob.squeeze(), pos_label)+criterion(weight=neg_edges_weight)(neg_prob.squeeze(), neg_label)
  elif weighted_loss_method in ["random_as_pos_edges_weight", "share_selected_random_weight_per_window" ]:
    assert neg_edges_weight is None
    assert pos_edges_weight is not None
    loss += criterion(weight=pos_edges_weight)(pos_prob.squeeze(), pos_label)+criterion()(neg_prob.squeeze(), neg_label)
  # elif weighted_loss_method == "ef_as_pos_edges_weight":
  #   # raise NotImplementedError()
  #   assert neg_edges_weight is None
  #   assert pos_edges_weight is not None
  #   loss += criterion(weight=pos_edges_weight)(pos_prob.squeeze(), pos_label)+criterion()(neg_prob.squeeze(), neg_label)
  # elif weighted_loss_method == "apply_time_decay_to_non_weighted_edges":
  #   # raise NotImplementedError()
  #   assert neg_edges_weight is None
  #   assert pos_edges_weight is not None
  #   loss += criterion(weight=pos_edges_weight)(pos_prob.squeeze(), pos_label)+criterion()(neg_prob.squeeze(), neg_label)
  elif weighted_loss_method == "no_weight":
    assert neg_edges_weight is None
    assert pos_edges_weight is None
    loss += criterion()(pos_prob.squeeze(), pos_label) + criterion()(neg_prob.squeeze(), neg_label)
  else:
    raise NotImplementedError()

  return loss

def compute_edges_probabilities_with_custom_sampled_nodes(model, neg_edges_formation, negatives_dst_batch, negatives_src_batch, sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS):
  # compute_edge_probability
  if neg_edges_formation == "original_src_and_sampled_dst" :
    pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negatives_dst_batch,
                                                        timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
  elif neg_edges_formation == "sampled_src_and_sampled_dst":
    # raise NotImplementedError
    pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negatives_dst_batch,
                                                        timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS, sampled_source_nodes=negatives_src_batch)
    # _, neg_prob = tgn.compute_edge_probabilities(sources_batch, _, negatives_dst_batch,
    #                                                     timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
  else:
    raise NotImplementedError()

  return model, pos_prob, neg_prob

def set_checkpoint_prefix():
  pass



# def sliding_window_evaluation(tgn,
#                             num_instance,
#                             BATCH_SIZE,
#                             NUM_EPOCH,
#                             logger,
#                             logger_2,
#                             USE_MEMORY,
#                             # MODEL_SAVE_PATH,
#                             args,
#                             optimizer,
#                             # criterion,
#                             full_data,
#                             device,
#                               NUM_NEIGHBORS,
#                               check_point):

#   # :TODO: write test on these. raise exception for all cases that wasn't intended or designed for.
#   prefix, neg_sample_method, neg_edges_formation, weighted_loss_method, compute_xf_iwf_with_sigmoid = get_conditions(args)

#   num_instances_shift, init_train_data, total_num_ws, init_num_ws, left_num_ws = get_sliding_window_params(num_instance, full_data, BATCH_SIZE, args.ws_multiplier)


#   check_point.data = args.data
#   check_point.prefix = prefix
#   check_point.bs = args.bs
#   check_point.epoch_max = args.n_epoch
#   check_point.ws_max = total_num_ws
#   check_point.max_random_weight_range = args.max_random_weight_range

#   ef_iwf_window_dict = {}
#   nf_iwf_window_dict = {}
#   share_selected_random_weight_per_window_dict = {}
#   pos_edges_weight = None
#   neg_edges_weight = None
#   epoch_times = []
#   total_epoch_times = []
#   end_train_idx = None
#   max_weight = args.max_random_weight_range
#   logger.info(f'max_weight = {max_weight}')

#   for ws in range(left_num_ws):

#     check_point.ws_idx = ws
#     num_batch = math.ceil((init_train_data)/BATCH_SIZE)
#     logger.debug('-ws = {}'.format(ws))
#     logger_2.info('-ws = {}'.format(ws))
#     ws_idx = ws
#     m_loss = []
#     for epoch in range(NUM_EPOCH):
#       check_point.epoch_idx = epoch
#       check_point.get_checkpoint_path()
#       logger.debug('--epoch = {}'.format(epoch))
#       logger_2.info('--epoch = {}'.format(epoch))
#       start_epoch = time.time()
#       ### Training :DOC:

#       # Reinitialize memory of the model at the start of each epoch
#       if USE_MEMORY:
#         tgn.memory.__init_memory__()

#       # # Train using only training graph
#       # tgn.set_neighbor_finder(train_ngh_finder)

#       # logger.debug('start {} epoch'.format(epoch))
#       # for k in range(0, init_num_batch, args.backprop_every):
#       for k in range(0, num_batch, args.backprop_every):
#         # logger.debug('---batch = {}'.format(k))
#         loss = 0
#         optimizer.zero_grad()

#         # Custom loop to allow to perform backpropagation only every a certain number of batches
#         for j in range(args.backprop_every):
#           # logger.debug('----backprop_every = {}'.format(j))
#           batch_idx = k + j
#           start_train_idx = batch_idx * BATCH_SIZE

#           batch_ref_window_size = 0
#           assert batch_ref_window_size < BATCH_SIZE

#           end_train_idx = min(init_train_data-batch_ref_window_size, start_train_idx + BATCH_SIZE)
#           end_train_idx = min(end_train_idx, num_instance-batch_ref_window_size) # edge case for hard sampling window.
#           end_train_hard_negative_idx = end_train_idx + batch_ref_window_size

#           assert end_train_hard_negative_idx <= init_train_data

#           if end_train_idx <= (num_instance - batch_ref_window_size):
#             assert (end_train_hard_negative_idx - batch_ref_window_size) == end_train_idx

#           assert start_train_idx < end_train_idx, "number of batch to run for each epoch was not set correctly."

#           sources_batch, destinations_batch = full_data.sources[start_train_idx:end_train_idx], \
#                                               full_data.destinations[start_train_idx:end_train_idx]
#           edge_idxs_batch = full_data.edge_idxs[start_train_idx: end_train_idx]
#           timestamps_batch = full_data.timestamps[start_train_idx:end_train_idx]

#           edge_hard_samples_batch = full_data.edge_idxs[start_train_idx:end_train_hard_negative_idx]
#           timestamps_hard_samples_batch = full_data.timestamps[start_train_idx:end_train_hard_negative_idx]

#           train_mask = full_data.timestamps < full_data.timestamps[end_train_idx]

#           # Initialize training neighbor finder to retrieve temporal graph
#           train_data = Data(full_data.sources[train_mask],
#                             full_data.destinations[train_mask],
#                             full_data.timestamps[train_mask],
#                             full_data.edge_idxs[train_mask],
#                             full_data.labels[train_mask])


#           # print(train_data.n_interactions, train_data.n_unique_nodes)
#           train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
#           tgn.set_neighbor_finder(train_ngh_finder)

#           size = len(sources_batch)

#           train_rand_sampler = get_sampler(train_data, batch_ref_window_size, start_train_idx, end_train_hard_negative_idx, neg_sample_method)

#           negatives_src_batch, negatives_dst_batch = get_negative_nodes_batch(train_rand_sampler, BATCH_SIZE, size, neg_sample_method)

#           pos_label, neg_label = init_pos_neg_labels(size, device)

#           criterion = get_criterion()


#           pos_prob, neg_prob = compute_edges_probabilities_with_custom_sampled_nodes(tgn, neg_edges_formation, negatives_dst_batch, negatives_src_batch, sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)


#           pos_edges_weight, neg_edges_weight = get_edges_weight(train_data,k, BATCH_SIZE,max_weight,start_train_idx, end_train_hard_negative_idx, ef_iwf_window_dict, nf_iwf_window_dict, share_selected_random_weight_per_window_dict, weighted_loss_method, sampled_nodes=negatives_src_batch, compute_xf_iwf_with_sigmoid=compute_xf_iwf_with_sigmoid)

#           logger_2.info(f'pos_edges_weight = {pos_edges_weight}')
#           logger_2.info(f'neg_edges_weight = {neg_edges_weight}')

#           loss = compute_loss(pos_label, neg_label, pos_prob, neg_prob, pos_edges_weight, neg_edges_weight, batch_idx, criterion, loss, weighted_loss_method)

#         loss /= args.backprop_every

#         loss.backward()
#         optimizer.step()
#         m_loss.append(loss.item())

#         # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
#         # the start of time
#         if USE_MEMORY:
#           tgn.memory.detach_memory()

#       epoch_time = time.time() - start_epoch
#       epoch_times.append(epoch_time)

#       logger.info('start validation...')
#       # Initialize validation and test neighbor finder to retrieve temporal graph
#       full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

#       ### Validation
#       # Validation uses the full graph
#       tgn.set_neighbor_finder(full_ngh_finder)

#       if USE_MEMORY:
#         # Backup memory at the end of training, so later we can restore it and use it for the
#         # validation on unseen nodes
#         train_memory_backup = tgn.memory.backup_memory()

#       # val_mask = full_data.timestamps < full_data.timestamps[end_train_idx + BATCH_SIZE]
#       # val_data = Data(full_data.sources[val_mask],
#       #                 full_data.destinations[val_mask],
#       #                 full_data.timestamps[val_mask],
#       #                 full_data.edge_idxs[val_mask],
#       #                 full_data.labels[val_mask],
#       #                 )
#       # val_rand_sampler = RandEdgeSampler(val_data.sources, val_data.destinations, seed=0)
#       #

#       # print('here')
#       # val_ap, val_auc = eval_edge_prediction(model=tgn,
#       #                                                         negative_edge_sampler=val_rand_sampler,
#       #                                                         data=val_data,
#       #                                                         n_neighbors=NUM_NEIGHBORS)
#       # print('now')

#       VAL_BATCH_SIZE = BATCH_SIZE * 10
#       sources_batch = full_data.sources[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
#       destinations_batch  = full_data.destinations[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
#       timestamps_batch  = full_data.timestamps[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
#       edge_idxs_batch  = full_data.edge_idxs[end_train_idx:end_train_idx + VAL_BATCH_SIZE]


#       size = len(sources_batch)
#       _, negative_samples = train_rand_sampler.sample(size)

#       pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch,
#                                                             negative_samples, timestamps_batch,
#                                                             edge_idxs_batch, NUM_NEIGHBORS)


#       pred_score = np.concatenate([pos_prob.cpu().data.detach().numpy(), neg_prob.cpu().data.detach().numpy()])
#       true_label = np.concatenate([np.ones(size), np.zeros(size)])

#       from sklearn.metrics import average_precision_score, roc_auc_score

#       val_auc = average_precision_score(true_label, pred_score)
#       val_ap = roc_auc_score(true_label, pred_score)

#       if USE_MEMORY:
#         val_memory_backup = tgn.memory.backup_memory()
#         # Restore memory we had at the end of training to be used when validating on new nodes.
#         # Also backup memory after validation so it can be used for testing (since test edges are
#         # strictly later in time than validation edges)
#         tgn.memory.restore_memory(train_memory_backup)

#       # val_aps.append(val_ap)
#       # train_losses.append(np.mean(m_loss))

#       # # Save temporary results to disk
#       # pickle.dump({
#       #   "val_aps": val_aps,
#       #   "new_nodes_val_aps": new_nodes_val_aps,
#       #   "train_losses": train_losses,
#       #   "epoch_times": epoch_times,
#       #   "total_epoch_times": total_epoch_times
#       # }, open(results_path, "wb"))

#       total_epoch_time = time.time() - start_epoch
#       total_epoch_times.append(total_epoch_time)

#       logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
#       logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
#       logger.info(
#         'val auc: {}'.format(val_auc))
#       logger.info(
#         'val ap: {}'.format(val_ap))

#       # # Early stopping
#       # if early_stopper.early_stop_check(val_ap):
#       #   logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
#       #   logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
#       #   best_model_path = get_checkpoint_path(early_stopper.best_epoch)
#       #   tgn.load_state_dict(torch.load(best_model_path))
#       #   logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
#       #   tgn.eval()
#       #   break
#       # else:
#       #   torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

#     if args.save_checkpoint:
#       torch.save(tgn.state_dict(), check_point.get_checkpoint_path())

#     # left_num_batch -= 1
#     # init_num_batch += 1
#     init_num_ws += 1
#     # init_train_data = init_num_batch * BATCH_SIZE
#     init_train_data = init_num_ws * num_instances_shift

#     # Training has finished, we have loaded the best model, and we want to backup its current
#     # memory (which has seen validation edges) so that it can also be used when testing on unseen

#     if USE_MEMORY:
#       val_memory_backup = tgn.memory.backup_memory()
