import math

import numpy as np
import time
import pickle
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.utils import EarlyStopMonitor, get_neighbor_finder
from utils.sampler import RandEdgeSampler
from utils.data_processing import Data
from tqdm import tqdm


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


def sliding_window_evaluation(tgn,
                            num_instance,
                            BATCH_SIZE,
                            logger,
                            USE_MEMORY,
                            MODEL_SAVE_PATH,
                            args,
                            optimizer,
                            criterion,
                            full_data,
                            device,
                            NUM_NEIGHBORS):

  epoch_times = []
  total_epoch_times = []

  # ADD: variables for sliding_window_evaluation
  # init_train_data = math.ceil(num_instance * 0.4)
  # init_train_data = math.ceil(num_instance * 0.05)
  init_train_data = math.ceil(num_instance * 0.01)

  end_train_idx = None


  num_instances_shift = BATCH_SIZE * 10

  # total_num_batch =  math.ceil(num_instance/BATCH_SIZE)
  # init_num_batch = math.ceil((init_train_data)/BATCH_SIZE)
  # left_num_batch = total_num_batch - init_num_batch

  total_num_ws =  math.ceil(num_instance/num_instances_shift)
  init_num_ws = math.ceil((init_train_data)/num_instances_shift)
  left_num_ws = total_num_ws - init_num_ws

  for ws in range(left_num_ws):

    num_batch = math.ceil((init_train_data)/BATCH_SIZE)

    # print(train_data.n_interactions, train_data.n_unique_nodes)
    logger.debug('-ws = {}'.format(ws))
    ws_idx = ws
    # logger.debug('-init_num_batch = {}'.format(init_num_batch))
    # logger.debug('-left_num_batch = {}'.format(left_num_batch))
    # logger.debug('--train_data per ws = {}'.format(train_data.sources[:init_train_data]))

    m_loss = []
    # for epoch in range(NUM_EPOCH):
    # epoch_ref_window_size =
    for epoch in range(5):
      logger.debug('--epoch = {}'.format(epoch))
      start_epoch = time.time()
      ### Training :DOC:

      # Reinitialize memory of the model at the start of each epoch
      if USE_MEMORY:
        tgn.memory.__init_memory__()

      # # Train using only training graph
      # tgn.set_neighbor_finder(train_ngh_finder)

      # logger.debug('start {} epoch'.format(epoch))
      # for k in range(0, init_num_batch, args.backprop_every):
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

          # print(num_batch, init_train_data, start_train_idx, end_train_idx)
          # print('what?')

          assert start_train_idx < end_train_idx, "number of batch to run for each epoch was not set correctly."

          # print(batch_idx)
          # print(start_train_idx)
          # print(end_train_idx)
          # print('omg')

          # logger.debug('----compute edges probabilitie')

          # sources_batch, destinations_batch = train_data.sources[start_train_idx:end_train_idx], \
          #                                     train_data.destinations[start_train_idx:end_train_idx]
          # edge_idxs_batch = train_data.edge_idxs[start_train_idx: end_train_idx]
          # timestamps_batch = train_data.timestamps[start_train_idx:end_train_idx]

          sources_batch, destinations_batch = full_data.sources[start_train_idx:end_train_idx], \
                                              full_data.destinations[start_train_idx:end_train_idx]
          edge_idxs_batch = full_data.edge_idxs[start_train_idx: end_train_idx]
          timestamps_batch = full_data.timestamps[start_train_idx:end_train_idx]

          edge_hard_samples_batch = full_data.edge_idxs[start_train_idx:end_train_hard_negative_idx]
          timestamps_hard_samples_batch = full_data.timestamps[start_train_idx:end_train_hard_negative_idx]


          # train_mask = timestamps <= val_time
          # train_mask = timestamps < full_data.edge_idx[:init_train_data]
          # train_mask = train_data.timestamps <= full_data.edge_idxs[end_train_idx]
          # train_mask = train_data.timestamps <= full_data.timestamps[end_train_idx]
          # train_mask = np.logical_and(full_data.timestamps <= full_data.timestamps[end_train_idx], observed_edges_mask)
          train_mask = full_data.timestamps < full_data.timestamps[end_train_idx]
          # train_mask = np.logical_and(full_data.timestamps[start_train_idx] <  full_data.timestamps , full_data.timestamps < full_data.timestamps[end_train_idx])
          # print(np.sum(train_mask))
          # print('yii')
          # exit()

          # Initialize training neighbor finder to retrieve temporal graph
          train_data = Data(full_data.sources[train_mask],
                            full_data.destinations[train_mask],
                            full_data.timestamps[train_mask],
                            full_data.edge_idxs[train_mask],
                            full_data.labels[train_mask])


          # print(train_data.n_interactions, train_data.n_unique_nodes)
          train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
          tgn.set_neighbor_finder(train_ngh_finder)

          size = len(sources_batch)
          if True:
            train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations, train_data.edge_idxs, batch_ref_window_size)
            # observed_rand_sampler = RandEdgeSampler(observed_data.sources, observed_data.destinations)

            _, negatives_batch = train_rand_sampler.sample(size)

          else:

            train_rand_sampler = RandEdgeSampler_v2(train_data.sources, train_data.destinations, train_data.edge_idxs, batch_ref_window_size)

            n_hard_negative = size


          with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

          # tgn = tgn.train() # :DOC:
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

      logger.info('start validation...')
      # Initialize validation and test neighbor finder to retrieve temporal graph
      full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

      ### Validation
      # Validation uses the full graph
      tgn.set_neighbor_finder(full_ngh_finder)

      if USE_MEMORY:
        # Backup memory at the end of training, so later we can restore it and use it for the
        # validation on unseen nodes
        train_memory_backup = tgn.memory.backup_memory()

      # val_mask = full_data.timestamps < full_data.timestamps[end_train_idx + BATCH_SIZE]
      # val_data = Data(full_data.sources[val_mask],
      #                 full_data.destinations[val_mask],
      #                 full_data.timestamps[val_mask],
      #                 full_data.edge_idxs[val_mask],
      #                 full_data.labels[val_mask],
      #                 )
      # val_rand_sampler = RandEdgeSampler(val_data.sources, val_data.destinations, seed=0)
      #

      # print('here')
      # val_ap, val_auc = eval_edge_prediction(model=tgn,
      #                                                         negative_edge_sampler=val_rand_sampler,
      #                                                         data=val_data,
      #                                                         n_neighbors=NUM_NEIGHBORS)
      # print('now')

      VAL_BATCH_SIZE = BATCH_SIZE * 10
      sources_batch = full_data.sources[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
      destinations_batch  = full_data.destinations[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
      timestamps_batch  = full_data.timestamps[end_train_idx:end_train_idx + VAL_BATCH_SIZE]
      edge_idxs_batch  = full_data.edge_idxs[end_train_idx:end_train_idx + VAL_BATCH_SIZE]


      size = len(sources_batch)
      _, negative_samples = train_rand_sampler.sample(size)

      pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, NUM_NEIGHBORS)


      pred_score = np.concatenate([pos_prob.cpu().data.detach().numpy(), neg_prob.cpu().data.detach().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      from sklearn.metrics import average_precision_score, roc_auc_score

      val_auc = average_precision_score(true_label, pred_score)
      val_ap = roc_auc_score(true_label, pred_score)

      if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()
        # Restore memory we had at the end of training to be used when validating on new nodes.
        # Also backup memory after validation so it can be used for testing (since test edges are
        # strictly later in time than validation edges)
        tgn.memory.restore_memory(train_memory_backup)

      # val_aps.append(val_ap)
      # train_losses.append(np.mean(m_loss))

      # # Save temporary results to disk
      # pickle.dump({
      #   "val_aps": val_aps,
      #   "new_nodes_val_aps": new_nodes_val_aps,
      #   "train_losses": train_losses,
      #   "epoch_times": epoch_times,
      #   "total_epoch_times": total_epoch_times
      # }, open(results_path, "wb"))

      total_epoch_time = time.time() - start_epoch
      total_epoch_times.append(total_epoch_time)

      logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
      logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
      logger.info(
        'val auc: {}'.format(val_auc))
      logger.info(
        'val ap: {}'.format(val_ap))

      # # Early stopping
      # if early_stopper.early_stop_check(val_ap):
      #   logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      #   logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      #   best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      #   tgn.load_state_dict(torch.load(best_model_path))
      #   logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      #   tgn.eval()
      #   break
      # else:
      #   torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

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


