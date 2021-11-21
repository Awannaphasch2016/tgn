#!/usr/bin/env python3

import math

import numpy as np
import time
import pickle
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.utils import EarlyStopMonitor
from utils.data_processing import Data
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def accuracy(labels,  pred):

  return (labels == pred).sum()/labels.shape[0]

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
  cm = confusion_matrix(data.labels, pred, labels=np.unique(data.labels))
  if data.n_unique_labels == 2:
    try:
      auc_roc = roc_auc_score(data.labels, pred_prob)
    except:
      raise "Something is wrong."

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

      if k == 5:
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


    val_auc, val_acc, cm = eval_node_classification(tgn,
                                                    decoder,
                                                    val_data,
                                                    # full_data.edge_idxs,
                                                    BATCH_SIZE,
                                       n_neighbors=NUM_NEIGHBORS)

    # val_auc, val_acc, cm = eval_node_classification(tgn, decoder, val_data, val_data.edge_idxs, BATCH_SIZE,
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

def sliding_window_evaluation_node_prediction(
    logger,
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
    decoder,
    decoder_optimizer,
    decoder_loss_criterion,
    # criterion,
    # optimizer
    ):

  num_instance = len(full_data.sources)

  epoch_times = []
  total_epoch_times = []

  init_train_data = max(BATCH_SIZE, math.ceil(num_instance * 0.001)) # :DEBUG:
  # init_train_data = math.ceil(num_instance * 0.001) # :DEBUG:
  # init_train_data = math.ceil(num_instance * 0.5)

  end_train_idx = None


  num_instances_shift = BATCH_SIZE * 1 # 100

  total_num_ws =  math.ceil(num_instance/num_instances_shift) # 6
  init_num_ws = math.ceil((init_train_data)/num_instances_shift) #6
  left_num_ws = total_num_ws - init_num_ws

  # print(total_num_ws, init_num_ws, left_num_ws) #  5716, 6, 5710
  # print(init_train_data, num_instances_shift) # 572, 100

  for ws in range(left_num_ws):

    num_batch = math.ceil((init_train_data)/BATCH_SIZE)

    logger.debug('-ws = {}'.format(ws))
    ws_idx = ws

    m_loss = []
    # for epoch in range(NUM_EPOCH):
    for epoch in range(5):
    # for epoch in range(100):
      logger.debug('--epoch = {}'.format(epoch))
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

          end_train_idx = min(init_train_data, start_train_idx + BATCH_SIZE)

          assert start_train_idx < end_train_idx, "number of batch to run for each epoch was not set correctly."

          sources_batch, destinations_batch = full_data.sources[start_train_idx:end_train_idx], \
                                              full_data.destinations[start_train_idx:end_train_idx]
          edge_idxs_batch = full_data.edge_idxs[start_train_idx: end_train_idx]
          timestamps_batch = full_data.timestamps[start_train_idx:end_train_idx]
          labels_batch = full_data.labels[start_train_idx:end_train_idx]

          size = len(sources_batch)

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
            pred = decoder(source_embedding).softmax(dim=1)
            labels_batch_torch = torch.from_numpy(labels_batch).long().to(device)
            decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)

          loss += decoder_loss.item()

        loss /= args.backprop_every

        decoder_loss.backward()
        decoder_optimizer.step()
        m_loss.append(decoder_loss.item())

        # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
        # the start of time
        if USE_MEMORY:
          tgn.memory.detach_memory()

      epoch_time = time.time() - start_epoch
      epoch_times.append(epoch_time)

      logger.info('start validation...')
      # Initialize validation and test neighbor finder to retrieve temporal graph

      ### Validation

      if USE_MEMORY:
        # Backup memory at the end of training, so later we can restore it and use it for the
        # validation on unseen nodes
        train_memory_backup = tgn.memory.backup_memory()

      VAL_BATCH_SIZE = BATCH_SIZE * 10

      # :DEBUG:
      time_before_end_of_next_batch = full_data.timestamps <= full_data.timestamps[end_train_idx + VAL_BATCH_SIZE]
      time_after_end_of_current_batch = full_data.timestamps > full_data.timestamps[end_train_idx]
      val_mask = np.logical_and(time_before_end_of_next_batch, time_after_end_of_current_batch)
      val_data = Data(full_data.sources[val_mask],
                      full_data.destinations[val_mask],
                      full_data.timestamps[val_mask],
                      full_data.edge_idxs[val_mask],
                      full_data.labels[val_mask],
                      )

      assert val_mask.sum() == VAL_BATCH_SIZE, "size of validation set must be the same as batch size."


      tgn.eval()
      decoder.eval(
)
      val_auc = eval_node_classification(tgn,
                                         decoder,
                                         val_data,
                                         # full_data.edge_idxs[end_train_idx:end_train_idx + VAL_BATCH_SIZE],
                                         # full_data.edge_idxs,
                                         # val_data.edge_idxs,
                                         VAL_BATCH_SIZE,
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
      logger.info(
        'val auc: {}'.format(val_auc))

    # left_num_batch -= 1
    # init_num_batch += 1
    init_num_ws += 1
    # init_train_data = init_num_batch * BATCH_SIZE
    init_train_data = init_num_ws * num_instances_shift

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()

