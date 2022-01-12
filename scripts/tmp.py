import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.crawler import Crawler


def draw_multiple_node_classification(losses, aucs, aps, epoches , config = {}, plot_path = None, savefig=False):
  # assert len(losses) == 2
  # assert len(aucs) == 2
  # assert len(aps) == 2
  # assert len(epoches) == 2
  raise NotImplementedError()

  ylim = None
  if len(config) > 0:
    raise NotImplemented
    if 'ylim' in config:
      assert len(config["ylim"]) == 2
      assert isinstance(config["ylim"][0], tuple)
      assert len(config["ylim"][0]) == 2
      ylim = iter(config["ylim"]) # list((min, max), (min,max))

  fig, axs = plt.subplots(1, 3, figsize=(9, 3))
  # axs[0].plot( loss[:,-1], label = 'loss')
  for i, loss in enumerate(losses):
    axs[0].plot( np.mean(loss, axis = 1), label = f'loss_{i}')

  # auc
  axs[1].plot( np.zeros(len(loss)))
  # axs[1].plot( auc[:,-1], label = 'auc')
  for i, auc in enumerate(aucs):
    axs[1].plot( np.mean(auc, axis = 1), label = f'auc_{i}')

  if ylim is not None:
    axs[1].set_ylim(*next(ylim))
  else:
    # axs[1].set_ylim(auc.min(), auc.max())
    axs[1].set_ylim(0, 1)

  # ap
  axs[2].plot( np.zeros(len(loss)))
  # axs[2].plot( ap[:,-1], label = 'ap')
  for i, ap in enumerate(aps):
    axs[2].plot( np.mean(ap, axis = 1), label = f'ap_{i}')
  if ylim is not None:
    axs[2].set_ylim(*next(ylim))
  else:
    # axs[2].set_ylim(ap.min(), ap.max())
    axs[2].set_ylim(0, 1)

  fig.suptitle('model performance')
  axs[0].legend()
  axs[1].legend()
  axs[2].legend(
)
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

    loss, auc, ap, epoch, ws, complete_ws_len = self.crawl_data(self.log_path)
    # self.crawl_data_v2(nodes_and_edges_weight_log_path)

    self.loss = loss
    self.auc = auc
    self.ap = ap
    self.epoch = epoch
    self.ws = ws
    self.complete_ws_len = complete_ws_len

  def plot(self, savefig=False):
    draw_node_classification(self.loss, self.auc, self.ap, self.epoch , self.plot_path, savefig=savefig)

  def get_return(self):
    return self.loss, self.auc, self.ap, self.epoch

  def get_ws_val(self, line):
    ws_val = line.split(" ")[-1].rstrip()
    return int(ws_val)

  def get_epoch_val(self, line):
    epoch_val = line.split(" ")[-1].rstrip()
    return int(epoch_val)

  def get_loss_val(self, line):
    loss_val = line.split(" ")[-1].rstrip()
    return float(loss_val)

  def get_auc_val(self, line):
    auc_val = line.split(" ")[-1].rstrip()
    return float(auc_val)

  def get_ap_val(self, line):
    ap_val = line.split(" ")[-1].rstrip()
    return float(ap_val)

  def crawl_data(self,log_path):
    loss = []
    auc = []
    ap = []
    epoch = []
    ws = []
    i = 1
    with open(log_path, 'r') as f:
      for i, line in enumerate(f.readlines()): # 9341 lines in total
        # if i == 100:
        #   exit()

        is_ws_exist = True if 'ws' in line and 'DEBUG' in line else False
        is_epoch_exist = True if 'epoch' in line and 'DEBUG' in line else False
        is_loss_exist = True if 'mean loss' in line and 'INFO' in line else False
        is_auc_exist = True if 'val auc' in line and 'INFO' in line else False
        is_ap_exist = True if 'val ap' in line and 'INFO' in line else False

        if is_ws_exist:
            ws.append(self.get_ws_val(line))
        if is_epoch_exist:
            epoch.append(self.get_epoch_val(line))
        if is_loss_exist:
            loss.append(self.get_loss_val(line))
        if is_auc_exist:
            auc.append(self.get_auc_val(line))
        if is_ap_exist:
            ap.append(self.get_ap_val(line))

    # non_missing_len = min(len(loss), len(auc), len(ap), len(epoch))
    complete_ws_len = min(max(ws) * 5, len(epoch))

    if complete_ws_len == max(ws) * 5:
      ws = ws[:-1]

    loss = np.array(loss[:complete_ws_len]).reshape(-1, 5)
    auc = np.array(auc[:complete_ws_len]).reshape(-1,5)
    ap = np.array(ap[:complete_ws_len]).reshape(-1,5)
    epoch = np.array(epoch[:complete_ws_len]).reshape(-1,5)
    return loss, auc, ap, epoch, ws, complete_ws_len

  def crawl_data_v2(self, log_path):

    pos_edges_weight = []

    with open(log_path, 'r') as f:
      for i, line in enumerate(f.readlines()): # 9341 lines in total
        # is_val_auc_exist = True if 'val auc' in line else False
        # is_cm_matrix_exist = True if '[' in line else False
        # is_pos_edges_weight = True if 'pos_edges_weight' in line else False
        is_neg_edges_weight = True if 'neg_edges_weight' in line else False
        is_epoch = True if 'epoch' in line else False
        is_ws = True if 'ws' in line else False
        is_comma_exist = True if ',' in line else False

        if (is_comma_exist and not is_neg_edges_weight and not is_epoch and not is_ws):
          pos_edges_weight_val = [float(i) for i in line.strip('\n,').split(']')[0].split('[')[-1].split(',')]
          pos_edges_weight.extend(pos_edges_weight_val)

      total_window_for_1_run = 0
      for i in range(max(self.ws)):
        total_window_for_1_run += i

      pos_edges_weight = pos_edges_weight[:total_window_for_1_run * 200 * 5]
      # pos_edges_weight = np.array(pos_edges_weight).reshape(5, -1, 200) # this will not work because each window iteration increase number of total window to iterate by 1.
      pos_edges_weight = np.array(pos_edges_weight).reshape(-1,200)
      pos_edges_weight = pos_edges_weight[-max(self.ws)+1:, :]

      return pos_edges_weight


if __name__ == "__main__":

  losses, aucs, aps, epoches = [], [], [] ,[]
  log_files = []

  # log_file = '1641588473.0998917' # model where =weighted_loss_method= is =share_selected_random_weight_per_window= + weight range is 0 - 500
  c11 = NodeClassificationCrawler(log_file)
  c11.plot()
  losses.append(c11.get_return()[0])
  aucs.append(c11.get_return()[1])
  aps.append(c11.get_return()[2])
  epoches.append(c11.get_return()[3])
  log_files.append(log_file)
  pos_edges_weight = c11.crawl_data_v2(c11.nodes_and_edges_weight_log_path)
  col_1 = pos_edges_weight.reshape(-1)
  col_2 = np.array([[i for _ in range(pos_edges_weight.shape[1])] for i in range(pos_edges_weight.shape[0])]).reshape(-1)
  col_name = ['weight', 'batch_idx']
  pos_edges_weight_pd = pd.DataFrame(np.array([col_1, col_2]).T, columns=col_name)
  pos_edges_weight_pd['batch_idx'] = pos_edges_weight_pd['batch_idx'].astype(int)
  base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  plot_path = str(base_path / f'plot/edges_weight_{log_file}.png')
  sns.boxplot(x="batch_idx", y="weight", data=pos_edges_weight_pd)
  # plt.savefig(plot_path)
  plt.show()

  log_file = '1641597198.5721123' # model where =weighted_loss_method= is =share_selected_random_weight_per_window= + weight range is 0 - 500
  c12 = NodeClassificationCrawler(log_file)
  # c12.plot()
  losses.append(c12.get_return()[0])
  aucs.append(c12.get_return()[1])
  aps.append(c12.get_return()[2])
  epoches.append(c12.get_return()[3])
  log_files.append(log_file)
  # pos_edges_weight = c12.crawl_data_v2(c12.nodes_and_edges_weight_log_path)
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


  list_ = []
  for i,j in zip(c11.get_return(),c12.get_return()):
    shortest_len = min(i.shape[0], j.shape[0])
    i = i[:shortest_len]
    j = j[:shortest_len]
    list_.append(i-j)
    # asser


  # # config = {}
  # # auc_ylim_tuple = (-0.01, 0.01)
  # # ap_ylim_tuple = (-0.01, 0.01)
  # # config['ylim'] = [auc_ylim_tuple, ap_ylim_tuple]

  # base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  # plot_path = str(base_path / f'plot/comparing_{log_files[0]}_and_{log_files[1]}.png')
  # # draw_node_classification(*list_, config=config)
  # # draw_node_classification(*list_, plot_path=plot_path, savefig=True)
  draw_node_classification(*list_)

  base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  plot_path = str(base_path / f'plot/{log_files[0]}_vs_{log_files[1]}.png')
  # draw_multiple_node_classification(losses, aucs, aps, epoches, plot_path=plot_path,savefig=True)
  draw_multiple_node_classification(losses, aucs, aps, epoches, plot_path=plot_path,savefig=False)
