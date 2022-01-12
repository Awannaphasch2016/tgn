#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import re
from utils.crawler import Crawler
from utils.utils import return_min_length_of_list_members

def draw_multiple_node_classification(losses, aucs, accs, epoches , config = {}, plot_path = None, savefig=False):
  # assert len(losses) == 2
  # assert len(aucs) == 2
  # assert len(accs) == 2
  # assert len(epoches) == 2

  ylim = None
  if len(config) > 0:
    raise NotImplemented
    if 'ylim' in config:
      assert len(config["ylim"]) == 2
      assert isinstance(config["ylim"][0], tuple)
      assert len(config["ylim"][0]) == 2
      ylim = iter(config["ylim"]) # list((min, max), (min,max))

  fig, axs = plt.subplots(1, 2, figsize=(9, 3))
  # axs[0].plot( loss[:,-1], label = 'loss')
  for i, loss in enumerate(losses):
    axs[0].plot( np.mean(loss, axis = 1), label = f'loss_{i}')

  # # auc
  # axs[1].plot( np.zeros(len(loss)))
  # # axs[1].plot( auc[:,-1], label = 'auc')
  # for i, auc in enumerate(aucs):
  #   axs[1].plot( np.mean(auc, axis = 1), label = f'auc_{i}')
  # if ylim is not None:
  #   axs[1].set_ylim(*next(ylim))
  # else:
  #   # axs[1].set_ylim(auc.min(), auc.max())
  #   axs[1].set_ylim(0, 1)

  # ap
  axs[1].plot( np.zeros(len(loss)))
  # axs[2].plot( ap[:,-1], label = 'ap')
  for i, acc in enumerate(accs):
    axs[1].plot( np.mean(acc, axis = 1), label = f'acc_{i}')
  if ylim is not None:
    axs[1].set_ylim(*next(ylim))
  else:
    # axs[2].set_ylim(ap.min(), ap.max())
    axs[1].set_ylim(0, 1)

  fig.suptitle('model performance')
  axs[0].legend()
  axs[1].legend()

  if savefig:
    assert plot_path is not None
    plt.savefig(plot_path)
  else:
    plt.show()

def draw_node_classification(loss, auc, ap, epoch , config = {}, plot_path = None, savefig=False):

  ylim = None
  if len(config) > 0:
    if 'ylim' in config:
      assert len(config["ylim"]) == 2
      assert isinstance(config["ylim"][0], tuple)
      assert len(config["ylim"][0]) == 2
      ylim = iter(config["ylim"]) # list((min, max), (min,max))

  fig, axs = plt.subplots(1, 3, figsize=(9, 3))
  # axs[0].plot( loss[:,-1], label = 'loss')
  axs[0].plot( np.mean(loss, axis = 1), label = 'loss')

  # auc
  axs[1].plot( np.zeros(len(loss)))
  # axs[1].plot( auc[:,-1], label = 'auc')
  axs[1].plot( np.mean(auc, axis = 1), label = 'auc')
  if ylim is not None:
    axs[1].set_ylim(*next(ylim))
  else:
    axs[1].set_ylim(auc.min(), auc.max())

  # ap
  axs[2].plot( np.zeros(len(loss)))
  # axs[2].plot( ap[:,-1], label = 'ap')
  axs[2].plot( np.mean(ap, axis = 1), label = 'ap')
  if ylim is not None:
    axs[2].set_ylim(*next(ylim))
  else:
    axs[2].set_ylim(ap.min(), ap.max())

  fig.suptitle('model performance')
  axs[0].legend()
  axs[1].legend()
  axs[2].legend()
  if savefig:
    assert plot_path is not None
    plt.savefig(plot_path)
  else:
    plt.show()

class NodeClassificationCrawler(Crawler):
  def __init__(self, log_file):
    base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
    self.log_path = str(base_path / f'log/{log_file}.log')
    self.nodes_and_edges_weight_log_path = str(base_path / f'log/nodes_and_edges_weight/{log_file}.log')
    self.plot_path = str(base_path / f'plot/{log_file}.png')

    self.n_classes = 4

    if self.n_classes != 4:
      raise NotADirectoryError()

    train_loss, val_acc, cm, ws, time_list, epochs, n_epoch = self.crawl_data()
    # self.crawl_data_v2(nodes_and_edges_weight_log_path)

    self.train_loss = train_loss
    self.val_acc = val_acc
    self.cm = cm
    self.ws = ws
    self.time_list = time_list
    self.epochs = epochs
    self.n_epoch = n_epoch

  def get_return(self):
    return self.train_loss, self.val_acc ,self.epochs

  def get_time_val(self, line):
    return float(line.split('took')[1].split('s')[0].strip(' \n'))

  def get_ws_val(self, line):
    return int(line.split('-ws')[-1].strip('= \n'))

  def get_epoch_val(self, line):
    return int(line.split('epoch')[1].split(':')[0].strip(' \n='))

  def get_loss_val(self, line):
    return float(line.split("loss")[1].split(':')[1].split(',')[0].strip(' '))

  def get_auc_val(self, line):
    return float(line.split("val auc")[1].split(':')[1].split(',')[1].strip(' '))

  def get_acc_val(self, line):
    return float(line.split(':')[-1])

  def get_acc_val_v1(self, line):
    return float(line.split('array')[0].strip(', ').split(',')[-1].strip(' '))

  def get_cm_val(self, line):
    if ',' not in line:
      cm_val = [int(i.strip('\n]')) for i in line.split(' ') if len(i) > 0 and re.findall('[0-9]+', i)]

    elif ']]' not in line:
      if 'val auc' in line and 'array' in line:
        cm_val = [int(i.strip(', ')) for i in line.split('[[')[-1].strip('],\n').split(',')]
      else:
        cm_val = [int(i.strip('[] ')) for i in line.strip('[])\n dtype)').strip(' ').split(',')[:-1]]

    else:
      if 'dtype' in line:
        cm_val = [int(i.strip('[ ,]')) for i in line.split(',')[:-1]]
      else:
        cm_val = [int(i.strip('[] ')) for i in line.strip('[])\n dtype)').strip(' ').split(',')]

    assert len(cm_val) == self.n_classes

    return cm_val

  def crawl_data(self):
    train_loss = []
    val_auc = []
    val_acc = []
    test_acc = []
    cm = []
    ws = []
    time_list = []
    epochs = []

    with open(self.log_path, 'r') as f:
      for i, line in enumerate(f.readlines()): # 9341 lines in total

        is_epoch_exist = True if '--epoch' in line else False
        is_train_loss_exist = True if 'loss' in line else False
        is_val_auc_exist = True if 'val auc' in line else False
        is_cm_matrix_exist = True if '[' in line and  'labels' not in line else False
        is_ws_exist = True if '-ws' in line else False
        is_time_exist = True if 'took' in line else False
        is_val_acc_exist = True if 'val acc' in line else False

        if is_time_exist:
          time_list.append(self.get_time_val(line))

        if is_ws_exist:
          ws.append(self.get_ws_val(line))

        if is_epoch_exist:
          epochs.append(self.get_epoch_val(line))

        if is_train_loss_exist:
          train_loss.append(self.get_loss_val(line))

        if is_val_auc_exist and self.n_classes == 2:
          val_auc.append(self.get_auc_val(line))

        if is_val_acc_exist:
          val_acc.append(self.get_acc_val(line))

        if is_val_auc_exist and 'array' in line:
          # :NOTE: this may cause error in the future. the implementation of this is not as robust.
          # 2022-01-11 15:03:12,420 - first_logger - INFO - val auc: (None, 0.7395833333333334, array([[ 1,  0, 24,  0],
          #       [ 0,  0,  0,  0],
          #       [ 1,  0, 70,  0],
          #       [ 0,  0,  0,  0]], dtype=int64))
          val_acc.append(self.get_acc_val_v1(line))

        if is_cm_matrix_exist:
          cm.append(self.get_cm_val(line))

    n_epoch = max(epochs) - min(epochs) + 1
    all_epoch = self.get_complete_epoch(ws, n_epoch, train_loss)

    cm = np.array(cm).reshape(-1).reshape(-1,self.n_classes,self.n_classes)
    cm = cm[:all_epoch,:,:]
    cm = cm.reshape(-1, n_epoch,self.n_classes,self.n_classes)

    train_loss = np.array(train_loss)
    train_loss = train_loss[:all_epoch].reshape(-1, n_epoch)

    time_list = time_list[:all_epoch]
    time_list = np.array(time_list).reshape(-1, n_epoch)

    # val_auc = np.array(val_auc)
    # val_auc = val_auc[:all_epoch].reshape(-1, n_epoch)

    val_acc = np.array(val_acc)
    val_acc = val_acc[:all_epoch].reshape(-1, n_epoch)

    return train_loss, val_acc, cm, ws, time_list, epochs, n_epoch


  def crawl_data_v2(self):
    """crawl to get weight variance of the all window of the last window iteration."""
    raise NotImplementedError()

  def get_n_ws_with_all_epoch(self, ws, n_epoch, train_loss):
    # ws = self.ws
    # n_epoch = self.n_epoch
    # train_loss = self.train_loss

    if len(train_loss) > ((max(ws) + 1) * n_epoch):
      n_ws_with_all_epoch = (max(ws) + 1) # number of ws that have complete epoch.
    else:
      n_ws_with_all_epoch = (len(train_loss)) - (len(train_loss) % n_epoch)
      n_ws_with_all_epoch = n_ws_with_all_epoch/n_epoch # number of ws that have complete epoch.
    return n_ws_with_all_epoch

  def get_complete_epoch(self, ws, n_epoch, train_loss):
    n_ws_with_all_epoch = self.get_n_ws_with_all_epoch(ws, n_epoch, train_loss)

    assert int(n_ws_with_all_epoch) == n_ws_with_all_epoch
    all_epoch = int(n_ws_with_all_epoch * n_epoch)
    return all_epoch

  def plot(self):
    raise NotImplementedError()

    train_loss = self.train_loss
    epochs = self.epochs
    ws = self.ws
    time_list = self.time_list
    val_auc = self.val_auc
    val_acc = self.val_acc
    n_ws = max(ws)
    n_epoch = self.n_epoch
    all_epoch = self.get_complete_epoch(ws, n_epoch, train_loss)

    # assert max(ws) == len(ws) - 1 # to test only the first run.

    # train_loss = np.array(train_loss)
    # train_loss = train_loss[:all_epoch]
    # time_list = time_list[:all_epoch]
    # time_list = np.array(time_list).reshape(-1, n_epoch)

    # val_auc = np.array(val_auc)
    # val_auc = val_auc[:all_epoch]

    title = 'loss and accuracy'
    plt.plot(train_loss, label = 'loss')
    plt.plot(val_auc, label = 'auc')
    plt.plot(val_acc, label = 'accuracy')
    plt.xlabel('step')
    plt.title(title)
    plt.legend()
    plt.show()
    # plt.savefig(plot_path/ f"{log_file}_{title}.png")

    # import pandas as pd
    # pd.DataFrame(cm.reshape(-1, 4)).to_csv('tmp.csv')

  def print_cm(self):

    # :NOTE: reshape here maybe wronge
    print(f'confusion matrix of the last window of the first run = \n{self.cm[-1,-1:,:]}')


if __name__ == "__main__":
  losses, aucs, accs, epoches, log_files = [], [], [], [], []

  # log_file = '1637548846.1373804' # (incorrect output from my_eval_node_classification) + originial
  log_file = '1641851230.354585' # (incorrect output from my_eval_node_classification) model with =use_random_weight_to_benchmark_nf_iwf= flag is True + weight range is 0 - 500
  # log_file = '1641934229.3281178' # (correct output from my_eval_node_classification) + random weight
  c1 = NodeClassificationCrawler(log_file)
  c1.print_cm()
  # c1.plot()
  losses.append(c1.get_return()[0])
  accs.append(c1.get_return()[1])
  epoches.append(c1.get_return()[2])
  log_files.append(log_file)
  # pos_edges_weight = c1.crawl_data_v2(c1.nodes_and_edges_weight_log_path)
  # col_1 = pos_edges_weight.reshape(-1)
  # col_2 = np.array([[i for _ in range(pos_edges_weight.shape[1])] for i in range(pos_edges_weight.shape[0])]).reshape(-1)
  # col_name = ['weight', 'batch_idx']
  # pos_edges_weight_pd = pd.DataFrame(np.array([col_1, col_2]).T, columns=col_name)
  # pos_edges_weight_pd['batch_idx'] = pos_edges_weight_pd['batch_idx'].astype(int)
  # base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  # plot_path = str(base_path / f'plot/edges_weight_{log_file}.png')
  # sns.boxplot(x="batch_idx", y="weight", data=pos_edges_weight_pd)
  # # plt.savefig(plot_path)
  # plt.show()

  log_file = '1641931341.737258' # model with =use_random_weight_to_benchmark_nf_iwf_1= flag is True + weight range is 0 - 500
  c2 = NodeClassificationCrawler(log_file)
  c2.print_cm()
  # c2.plot()
  losses.append(c2.get_return()[0])
  accs.append(c2.get_return()[1])
  epoches.append(c2.get_return()[2])
  log_files.append(log_file)
  # pos_edges_weight = c2.crawl_data_v2(c2.nodes_and_edges_weight_log_path)
  # col_1 = pos_edges_weight.reshape(-1)
  # col_2 = np.array([[i for _ in range(pos_edges_weight.shape[1])] for i in range(pos_edges_weight.shape[0])]).reshape(-1)
  # col_name = ['weight', 'batch_idx']
  # pos_edges_weight_pd = pd.DataFrame(np.array([col_1, col_2]).T, columns=col_name)
  # pos_edges_weight_pd['batch_idx'] = pos_edges_weight_pd['batch_idx'].astype(int)
  # base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  # plot_path = str(base_path / f'plot/edges_weight_{log_file}.png')
  # sns.boxplot(x="batch_idx", y="weight", data=pos_edges_weight_pd)
  # # plt.savefig(plot_path)
  # plt.show()

  # list_ = []
  # for i,j in zip(c1.get_return(),c2.get_return()):
  #   shortest_len = return_min_length_of_list_members([i,j])
  #   # shortest_len = min(i.shape[0], j.shape[0])
  #   i = np.array(i)[:shortest_len]
  #   j = np.array(j)[:shortest_len]
  #   # i = np.array(i)[:shortest_len].reshape(-1, 1)
  #   # j = np.array(j)[:shortest_len].reshape(-1,1)
  #   list_.append(i-j)

  # # base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  # # plot_path = str(base_path / f'plot/comparing_{log_files[0]}_and_{log_files[1]}.png')
  # # # draw(*list_, config=config)
  # # # draw(*list_, plot_path=plot_path, savefig=True)
  # draw_node_classification(*list_)

  base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  plot_path = str(base_path / f'plot/{log_files[0]}_vs_{log_files[1]}.png')
  # draw_multiple(losses, aucs, aps, epoches, plot_path=plot_path,savefig=True)
  draw_multiple_node_classification(losses, aucs, accs, epoches, plot_path=plot_path,savefig=False)
