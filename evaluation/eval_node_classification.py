#!/usr/bin/env python3

import math

import numpy as np
import time
import pickle
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.utils import EarlyStopMonitor, label_new_unique_nodes_with_budget, find_nodes_ind_to_be_labelled, get_label_distribution, pred_prob_to_pred_labels, get_unique_nodes_labels, get_conditions_node_classification, get_nf_iwf, get_encoder, get_share_selected_random_weight_per_window, get_sliding_window_params
from utils.data_processing import Data
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from random import choices
from utils.utils import MLP, MLP_multiple_class
import pandas as pd
import random

def accuracy(labels,  pred):

  return (labels == pred).sum()/labels.shape[0]

def my_eval_node_classification(logger, tgn, decoder, data, batch_size, label_sources_in_training, n_neighbors):
  pred_prob = np.zeros((len(data.sources), data.n_unique_labels))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)
  all_selected_sources_ind = []
  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      # print(k)
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      # edge_idxs_batch = edge_idxs[s_idx: e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      # source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
      #                                                                               destinations_batch,
      #                                                                               destinations_batch,
      #                                                                               timestamps_batch,
      #                                                                               edge_idxs_batch)

      self = tgn
      source_nodes = sources_batch
      destination_nodes = destinations_batch
      negative_nodes = destinations_batch
      edge_times = timestamps_batch
      edge_idxs = edge_idxs_batch
      n_neighbors = 20

      n_samples = len(source_nodes)
      nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
      positives = np.concatenate([source_nodes, destination_nodes])
      timestamps = np.concatenate([edge_times, edge_times, edge_times])

      memory = self.memory.get_memory(list(range(self.n_nodes)))
      last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)
      # Compute the embeddings using the embedding module
      node_embedding = self.embedding_module.compute_embedding(memory=memory,

                                                               source_nodes=nodes,
                                                               timestamps=timestamps,
                                                               n_layers=self.n_layers,
                                                               n_neighbors=n_neighbors,
                                                               time_diffs=time_diffs)

      source_node_embedding = node_embedding[:n_samples]
      destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
      negative_node_embedding = node_embedding[2 * n_samples:]

      source_embedding = source_node_embedding

      label_sources_in_test = np.setdiff1d(source_nodes, label_sources_in_training)
      selected_sources_ind = find_nodes_ind_to_be_labelled(label_sources_in_test, source_nodes)
      all_selected_sources_ind.extend(np.array(selected_sources_ind) + s_idx) # selected ind uses s_idx as a starting ind.
      if data.n_unique_labels == 2:
        raise NotImplementedError
        # source_embedding.shape = (100, 172)
        pred_prob_batch = decoder(source_embedding).sigmoid()
        pred_prob[s_idx: e_idx, : ] = pred_prob_batch.cpu().numpy()
        pred_prob = pred_prob.reshape(-1)
      elif data.n_unique_labels == 4:
        pred_prob_batch = decoder(source_embedding[selected_sources_ind]).softmax(dim=1)
        pred_prob[selected_sources_ind, :] = pred_prob_batch.cpu().detach().numpy() #　:BUG: random bug. haven't spend time to figure the cause.
      else:
        raise NotImplementedError

  pred = pred_prob_to_pred_labels(pred_prob, all_selected_sources_ind)

  # if pred_prob.reshape(-1).shape[0] == pred_prob.shape[0]:
  #   raise NotImplementedError
  #   pred = pred_prob > 0.5
  # else:
  #   pred = pred_prob[all_selected_sources_ind].argmax(axis=1)

  logger.info(
    f"test labels distribution = {get_label_distribution(data.labels[all_selected_sources_ind])}")
  logger.info(
    f"predicted test labels distribution = {get_label_distribution(pred)}")
  logger.info(
    f"test labels epoch distribution (disregard frequency of unique node) = "
    f"{get_label_distribution(get_unique_nodes_labels(data.labels[all_selected_sources_ind],data.sources[all_selected_sources_ind]))}")
  logger.info(
    f"predicted test labels epoch distribution (disregard frequency of unique node) = "
    f"{get_label_distribution(get_unique_nodes_labels(pred,data.sources[all_selected_sources_ind]))}")  # note that it is possible that model predict different labels for the same nodes. (I will omit this metric until it is shown to be needed.)
  auc_roc = None
  acc = accuracy(data.labels[all_selected_sources_ind], pred)
  cm = confusion_matrix(data.labels[all_selected_sources_ind], pred, labels=list(range(data.n_unique_labels)))
  if data.n_unique_labels == 2:
    raise NotImplementedError
    try:
      auc_roc = roc_auc_score(data.labels, pred_prob)
    except:
      raise Exception("Something is wrong.")

  return auc_roc, acc, cm


# def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
def eval_node_classification(tgn, decoder, data, batch_size, n_neighbors):
  pred_prob = np.zeros((len(data.sources), data.n_unique_labels))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)
  # print(num_instance, num_batch)  # 672 7

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      print(k)
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      # edge_idxs_batch = edge_idxs[s_idx: e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                    destinations_batch,
                                                                                    destinations_batch,
                                                                                    timestamps_batch,
                                                                                    edge_idxs_batch)

      if data.n_unique_labels == 2:
        # source_embedding.shape = (100, 172)
        pred_prob_batch = decoder(source_embedding).sigmoid()
        pred_prob[s_idx: e_idx, : ] = pred_prob_batch.cpu().numpy()
        pred_prob = pred_prob.reshape(-1)
      elif data.n_unique_labels == 4:
        pred_prob_batch = decoder(source_embedding).softmax(dim=1)
        pred_prob[s_idx: e_idx, :] = pred_prob_batch.cpu().numpy()
      else:
        raise NotImplementedError


  if pred_prob.reshape(-1).shape[0] == pred_prob.shape[0]:
    pred = pred_prob > 0.5
  else:
    pred = pred_prob.argmax(axis=1)

  # pred_prob.shape = (1000867, )
  # data.labels.shape = (1000867, )

  auc_roc = None
  acc = accuracy(data.labels, pred)
  cm = confusion_matrix(data.labels, pred, labels=list(range(data.n_unique_labels)))
  if data.n_unique_labels == 2:
    try:
      auc_roc = roc_auc_score(data.labels, pred_prob)
    except:
      raise Exception("Something is wrong.")

  return auc_roc, acc, cm

  # try:
    # auc_roc = roc_auc_score(data.labels, pred_prob)
    # return auc_roc
  # except:
  #   from sklearn.metrics import confusion_matrix
  #   print(confusion_matrix(data.labels, pred_prob))

  #   return accuracy(data.labels, pred_prob)


def train_val_test_evalulation_node_prediction(
    logger,
    MODEL_SAVE_PATH,
    tgn,
    device,
    num_batch,
    BATCH_SIZE,
    USE_MEMORY,
    num_instance,
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
    decoder,
    decoder_optimizer,
    decoder_loss_criterion,
):

  ## Load models from eheckpoint
  # logger.info('Loading saved TGN model')
  # model_path = f'./saved_models/{args.prefix}-{DATA}.pth'
  # tgn.load_state_dict(torch.load(model_path))
  # tgn.eval()
  # logger.info('TGN models loaded')
  # logger.info('Start training node classification task')

  val_aucs = []
  val_accs = []
  cms = []
  train_losses = []


  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(args.n_epoch):
    start_epoch = time.time()

    # Initialize memory of the model at each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()
    # tgn = tgn.eval()

    tgn = tgn.train()
    # decoder = decoder.train()
    loss = 0

    for k in tqdm(list(range(num_batch))):
      print(f'list(range(num_batch)) = {k}')
      if k == 2:
        break

      start_batch = time.time()
      # tgn = tgn.train()
      s_idx = k * BATCH_SIZE
      e_idx = min(num_instance, s_idx + BATCH_SIZE)

      sources_batch = train_data.sources[s_idx: e_idx]
      destinations_batch = train_data.destinations[s_idx: e_idx]
      timestamps_batch = train_data.timestamps[s_idx: e_idx]
      # edge_idxs_batch = full_data.edge_idxs[s_idx: e_idx]
      edge_idxs_batch = train_data.edge_idxs[s_idx: e_idx]
      labels_batch = train_data.labels[s_idx: e_idx]

      size = len(sources_batch)

      decoder_optimizer.zero_grad()

      # with torch.no_grad():
      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                    destinations_batch,
                                                                                    destinations_batch,
                                                                                    timestamps_batch,
                                                                                    edge_idxs_batch,
                                                                                    NUM_NEIGHBORS)

      if train_data.n_unique_labels == 2:
        pred = decoder(source_embedding).sigmoid()
        labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
        decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
      elif train_data.n_unique_labels == 4:
        # using Crossentropy
        pred = decoder(source_embedding).softmax(dim=1)
        labels_batch_torch = torch.from_numpy(labels_batch).long().to(device)
        decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
      else:
        raise NotImplementedError


      decoder_loss.backward(retain_graph=True)
      decoder_optimizer.step()
      loss_batch = decoder_loss.item()
      # logger.info(f'batch {k}/{num_batch}: train loss: {loss_batch}, time: {time.time() - start_batch}')
      loss += loss_batch


    train_losses.append(loss / num_batch)

    # :DEBUG:
    val_auc, val_acc, cm = my_eval_node_classification(tgn,
                                                    decoder,
                                                    val_data,
                                                    # full_data.edge_idxs,
                                                    BATCH_SIZE,
                                       n_neighbors=NUM_NEIGHBORS)

    # val_auc, val_acc, cm = eval_node_classification(tgn,
    #                                                 decoder,
    #                                                 val_data,
    #                                                 # full_data.edge_idxs,
    #                                                 BATCH_SIZE,
    #                                    n_neighbors=NUM_NEIGHBORS)


    val_accs.append(val_acc)
    cms.append(cm)
    if val_auc is not None:
      val_aucs.append(val_auc)
      logger.info(f'Epoch {epoch}: train loss: {loss / num_batch}, val_acc: {val_acc}, val auc: {val_auc}, time: {time.time() - start_epoch}')
    logger.info(f'Epoch {epoch}: train loss: {loss / num_batch}, val acc: {val_acc}, time: {time.time() - start_epoch}\n confusion matrix = \n{cm}')

  if args.use_validation:
    try:
      if early_stopper.early_stop_check(val_auc):
        logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
        return
      else:
        torch.save(decoder.state_dict(), get_checkpoint_path(epoch))
    except:
      pass

  test_auc = None
  test_acc = None
  test_cm = None

  if args.use_validation:
    logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
    best_model_path = get_checkpoint_path(early_stopper.best_epoch)
    decoder.load_state_dict(torch.load(best_model_path))
    logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
    decoder.eval()

    test_auc, test_acc, cm = eval_node_classification(tgn,
                                                      decoder,
                                                      test_data,
                                                      # full_data.edge_idxs,
                                                      BATCH_SIZE,
                                        n_neighbors=NUM_NEIGHBORS)

  else:
    # If we are not using a validation set, the test performance is just the performance computed
    # in the last epoch
    test_acc = val_accs[-1]
    cm = cms[-1]
    if val_auc is not None:
      test_auc = val_aucs[-1]

  # pickle.dump({
  #   "val_aps": val_aucs,
  #   "test_ap": test_auc,
  #   "train_losses": train_losses,
  #   "epoch_times": [0.0],
  #   "new_nodes_val_aps": [],
  #   "new_node_test_ap": 0,
  # }, open(results_path, "wb"))

  if test_auc is not None:
    logger.info(f'test auc: {test_auc}, test acc: {test_acc}.\nconfusion matrix\n={cm}')
  logger.info(f'test acc: {test_acc}\nconfusion matrix\n={cm}')

def cross_entropy_with_weighted_nodes(weight=None):
  """
  ref: https://www.google.com/imgres?imgurl=https%3A%2F%2Fi.stack.imgur.com%2FgNip2.png&imgrefurl=https%3A%2F%2Fstackoverflow.com%2Fquestions%2F41990250%2Fwhat-is-cross-entropy&tbnid=EvY6xhP4_fjHMM&vet=12ahUKEwiaqcfXy531AhUCEd8KHXVoAfEQMygAegUIARDTAQ..i&docid=mtdj-XG_g1QKxM&w=500&h=208&itg=1&q=cross%20entropy&ved=2ahUKEwiaqcfXy531AhUCEd8KHXVoAfEQMygAegUIARDTAQ

  p(x) (aka labels) is one-hot
  q(x) (aka preds) is probability distribution
  -----
  preds and labels are torch.tensor
  preds shape is (instances, classes)
  labels shape is (instances, classes)
  instances_weight shape is (instances)
  """

  def loss(preds,labels):
    assert weight is not None
    instances_weight = weight
    # labels = convert_to_onehot(labels)

    return 0 - torch.sum(torch.mul(instances_weight, torch.sum(torch.mul(labels, torch.log(preds)), axis=1)))
  return loss

def select_decoder_and_loss(args,device,feat_dim, n_unique_labels, weighted_loss_method):
  ## use with pre-training model to substitute prediction head

  # if args.use_nf_iwf_weight:
  if weighted_loss_method in ["nf_iwf_as_nodes_weight", "random_as_node_weight", "share_selected_random_weight_per_window"]:
    if n_unique_labels == 2:
      raise NotImplementedError()
    else:
      decoder = MLP_multiple_class(feat_dim, n_unique_labels ,drop=args.drop_out)
      decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
      decoder = decoder.to(device)
      decoder_loss_criterion = cross_entropy_with_weighted_nodes
  elif weighted_loss_method == "no_weight":
    if n_unique_labels == 2:
      decoder = MLP(feat_dim, drop=args.drop_out)
      decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
      decoder = decoder.to(device)
      decoder_loss_criterion = torch.nn.BCELoss
    else:
      decoder = MLP_multiple_class(feat_dim, n_unique_labels ,drop=args.drop_out)
      decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
      decoder = decoder.to(device)
      decoder_loss_criterion = torch.nn.CrossEntropyLoss
  else:
    raise NotImplementedError()

  return decoder_optimizer, decoder, decoder_loss_criterion

def get_nodes_weight(full_data, batch_idx, batch_size, max_weight,start_train_idx, end_train_idx, nf_iwf_window_dict, n_unique_labels, weighted_loss_method, share_selected_random_weight_per_window_dict):

  nodes_weight = None

  if n_unique_labels == 2:
    raise NotImplementedError()
  elif n_unique_labels == 4:
    # if args.use_nf_iwf_weight:
    if weighted_loss_method == "nf_iwf_as_nodes_weight":
      # calculate sources weight from sources batch.
      nodes_weight = get_nf_iwf(full_data, batch_idx, batch_size, start_train_idx, end_train_idx, nf_iwf_window_dict)
    elif weighted_loss_method == "random_as_node_weight":
      nodes_weight = [random.randint(0,500) for i in range(batch_size)]
      nodes_weight = torch.FloatTensor(nodes_weight)
    elif weighted_loss_method == "share_selected_random_weight_per_window":
      nodes_weight = get_share_selected_random_weight_per_window(batch_size, max_weight,batch_idx, share_selected_random_weight_per_window_dict)
    elif weighted_loss_method == "no_weight":
      pass
    else:
      raise NotImplementedError()
  else:
    raise NotImplementedError()

  return nodes_weight

def sliding_window_evaluation_node_prediction(
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
    ):
  num_instance = len(full_data.sources)
  end_train_idx = None
  selected_sources_to_label = []
  nf_iwf_window_dict = {}
  share_selected_random_weight_per_window_dict = {}
  epoch_times = []
  total_epoch_times = []

  onehot_encoder = get_encoder(full_data.n_unique_labels)

  weighted_loss_method = get_conditions_node_classification(args)

  # get decoder and loss
  feat_dim = node_features.shape[1]
  n_unique_labels = full_data.n_unique_labels
  decoder_optimizer, decoder, decoder_loss_criterion = select_decoder_and_loss(args,device,feat_dim, n_unique_labels, weighted_loss_method)

  num_instances_shift, init_train_data, total_num_ws, init_num_ws, left_num_ws = get_sliding_window_params(num_instance, full_data, BATCH_SIZE)

  begin_ws_idx = 0 # pointer for first index of previously added window
  end_ws_idx = init_train_data # pointer for last index of previously added window

  for ws in range(left_num_ws):

    num_batch = math.ceil((end_ws_idx)/BATCH_SIZE)

    logger.debug('-ws = {}'.format(ws))
    logger_2.info('--ws = {}'.format(ws))
    ws_idx = ws

    m_loss = []

    # selected_sources_to_label_before = selected_sources_to_label.copy()
    # len_before = len(selected_sources_to_label_before)

    selected_sources_ind,selected_sources_to_label = label_new_unique_nodes_with_budget(selected_sources_to_label, full_data, (begin_ws_idx, end_ws_idx))
    # assert selected_sources_to_label[:len_before] == selected_sources_to_label_before
    assert np.unique(selected_sources_to_label).shape[0] == len(selected_sources_to_label)

    for epoch in range(NUM_EPOCH):
      logger.debug('--epoch = {}'.format(epoch))
      logger_2.info('--epoch = {}'.format(epoch))
      start_epoch = time.time()

      ### Training　

      # Reinitialize memory of the model at the start of each epoch
      if USE_MEMORY:
        tgn.memory.__init_memory__()

      decoder = decoder.train()

      for k in range(0, num_batch, args.backprop_every):

        loss = 0
        decoder_optimizer.zero_grad()

        # Custom loop to allow to perform backpropagation only every a certain number of batches
        for j in range(args.backprop_every):
          batch_idx = k + j
          start_train_idx = batch_idx * BATCH_SIZE

          end_train_idx = min(end_ws_idx, start_train_idx + BATCH_SIZE)

          assert (end_ws_idx - begin_ws_idx) <= BATCH_SIZE, "if false, *_batch will encounter out of  bound error. Maybe intial number of data is more than BATCH_SIZE."
          assert start_train_idx < end_train_idx, "number of batch to run for each epoch was not set correctly."
          # assert len(selected_sources_ind) >= end_ws_idx
          # print(start_train_idx, end_train_idx)

          sources_batch, destinations_batch = full_data.sources[start_train_idx:end_train_idx], \
                                              full_data.destinations[start_train_idx:end_train_idx]
          edge_idxs_batch = full_data.edge_idxs[start_train_idx: end_train_idx]
          timestamps_batch = full_data.timestamps[start_train_idx:end_train_idx]
          labels_batch = full_data.labels[start_train_idx:end_train_idx]
          total_labels_batch = labels_batch



          # size = len(sources_batch)

          source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, sampled_source_nodes=None, n_neighbors=NUM_NEIGHBORS)

          labels_batch = labels_batch[selected_sources_ind]
          sources_batch = sources_batch[selected_sources_ind]

          raise NotImplementedError("added arugment to get_nodes_weight and I haven't test it in node_classification yet.")
          nodes_weight = get_nodes_weight(full_data, batch_idx, BATCH_SIZE, start_train_idx, end_train_idx, nf_iwf_window_dict, n_unique_labels, weighted_loss_method, share_selected_random_weight_per_window_dict)

          logger_2.info(f'nodes_weight = {nodes_weight}')

          # # calculate sources weight from sources batch.
          # nodes_weight = get_nf_iwf(full_data, batch_idx, BATCH_SIZE, start_train_idx, end_train_idx, nf_iwf_window_dict)

          nodes_weight_batch = nodes_weight[selected_sources_ind]

          if train_data.n_unique_labels == 2: # :NOTE: for readability, train_data should be replaced by full_data, but I am unsure about side effect.
            raise NotImplementedError
            pred = decoder(source_embedding).sigmoid()
            labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
            decoder_loss = decoder_loss_criterion(weight=nodes_weight_batch)(pred, labels_batch_torch)
          elif train_data.n_unique_labels == 4:
            pred = decoder(source_embedding[selected_sources_ind]).softmax(dim=1) # :BUG: I am not sure if selected_sources_ind can be appplied here without effecting model learning
            labels_batch_torch = torch.from_numpy(onehot_encoder.transform(pd.DataFrame(labels_batch)).toarray()).long().to(device)
            decoder_loss = decoder_loss_criterion(weight=nodes_weight_batch)(pred, labels_batch_torch)
            # loss += criterion()(pos_prob.squeeze(), pos_label) + criterion()(neg_prob.squeeze(), neg_label)

          pred = pred_prob_to_pred_labels(pred.cpu().detach().numpy()) # :NOTE: not sure what this is used for.

          loss += decoder_loss.item()

        # logger.info(
        #   f"train labels batch distribution = {get_label_distribution(labels_batch)}")
        # logger.info(
        #   f"predicted train labels batch distribution = {get_label_distribution(pred)}")

        loss /= args.backprop_every

        decoder_loss.backward()
        decoder_optimizer.step()
        m_loss.append(decoder_loss.item())

        # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
        # the start of time
        if USE_MEMORY:
          tgn.memory.detach_memory()

      logger.info(f"total labels batch epoch distribution = {get_label_distribution(total_labels_batch)}")
      logger.info(
        f"train labels epoch distribution = {get_label_distribution(labels_batch)}")
      logger.info(
        f"predicted train labels epoch distribution = {get_label_distribution(pred)}")
      logger.info(
        f"train labels epoch distribution (disregard frequency of unique node) = "
        f"{get_label_distribution(get_unique_nodes_labels(labels_batch, sources_batch))}")
      logger.info(
        f"predicted train labels epoch distribution (disregard frequency of unique node) = "
        f"{get_label_distribution(get_unique_nodes_labels(pred, sources_batch))}") # note that it is possible that model predict different labels for the same nodes. (I will omit this metric until it is shown to be needed.)
      epoch_time = time.time() - start_epoch
      epoch_times.append(epoch_time)
      logger.info(f'total number of labelled uniqued nodes = {len(selected_sources_to_label)}')
      logger.info('start validation...')
      # Initialize validation and test neighbor finder to retrieve temporal graph

      ### Validation

      if USE_MEMORY:
        # Backup memory at the end of training, so later we can restore it and use it for the
        # validation on unseen nodes
        train_memory_backup = tgn.memory.backup_memory()

      VAL_BATCH_SIZE = BATCH_SIZE * 1

      # :NOTE: For reddit_with_expert_labels_10000 dataset, full_data.n_interactions is 9999 not 10k. That's why it failed. (data was not the length it was expected.)
      assert full_data.timestamps.shape[0] >= end_train_idx + BATCH_SIZE

      # :DEBUG:
      time_after_end_of_current_batch = full_data.timestamps > full_data.timestamps[end_train_idx]
      time_before_end_of_next_batch = full_data.timestamps <= full_data.timestamps[end_train_idx + BATCH_SIZE]
      val_mask = np.logical_and(time_after_end_of_current_batch, time_before_end_of_next_batch)
      val_data = Data(full_data.sources[val_mask],
                      full_data.destinations[val_mask],
                      full_data.timestamps[val_mask],
                      full_data.edge_idxs[val_mask],
                      full_data.labels[val_mask],
                      )
      # assert val_mask.sum() == VAL_BATCH_SIZE, "size of validation set must be the same as batch size."

      # tgn.eval()
      # decoder.eval()

      # val_auc = my_eval_node_classification(logger,
      #                                       tgn,
      #                                    decoder,
      #                                    val_data,
      #                                    # full_data.edge_idxs[end_train_idx:end_train_idx + VAL_BATCH_SIZE],
      #                                    # full_data.edge_idxs,
      #                                    # val_data.edge_idxs,
      #                                    VAL_BATCH_SIZE,
      #                                    selected_sources_to_label,
      #                                   n_neighbors=NUM_NEIGHBORS)

      val_auc, val_acc, cm  = my_eval_node_classification(logger,
                                            tgn,
                                         decoder,
                                         val_data,
                                         # full_data.edge_idxs[end_train_idx:end_train_idx + VAL_BATCH_SIZE],
                                         # full_data.edge_idxs,
                                         # val_data.edge_idxs,
                                         VAL_BATCH_SIZE,
                                         selected_sources_to_label,
                                        n_neighbors=NUM_NEIGHBORS)



      # sources_batch = full_data.sources[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
      # destinations_batch  = full_data.destinations[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
      # timestamps_batch  = full_data.timestamps[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
      # edge_idxs_batch  = full_data.edge_idxs[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
      # labels_idxs_batch  = full_data.labels[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
      # source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
      #                                                                              destinations_batch,
      #                                                                              destinations_batch,
      #                                                                              timestamps_batch,
      #                                                                              edge_idxs_batch,
      #                                                                              NUM_NEIGHBORS)
      # pred_prob_batch = decoder(source_embedding).sigmoid()
      # pred_prob= pred_prob_batch.detach().cpu().numpy()
      # val_auc = roc_auc_score(labels_idxs_batch, pred_prob)

      if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()
        # Restore memory we had at the end of training to be used when validating on new nodes.
        # Also backup memory after validation so it can be used for testing (since test edges are
        # strictly later in time than validation edges)
        tgn.memory.restore_memory(train_memory_backup)

      total_epoch_time = time.time() - start_epoch
      total_epoch_times.append(total_epoch_time)

      logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
      logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
      logger.info(f'val acc: {val_acc}')
      logger.info(f'confusion matrix = \n{cm}')

      # logger.info(
      #   'val auc: {}'.format(val_auc))

    # left_num_batch -= 1
    # init_num_batch += 1
    init_num_ws += 1
    begin_ws_idx = end_ws_idx
    end_ws_idx = min(init_num_ws * num_instances_shift, full_data.edge_idxs.shape[0]-1)

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()

