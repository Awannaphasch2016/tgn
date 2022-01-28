import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.crawler import Crawler
from utils.utils import return_min_length_of_list_members, apply_off_set_ind

def get_shape_of_val_in_list(val):
  return [i.shape for i in val]

def get_len_of_val_in_list(val):
  return [len(i) for i in val]

def get_mean_val_from_same_period(val, off_set_ind, begin_idx_list, end_idx_list):
  tmp = []
  val_np = np.array(val)
  begin_idx_list = apply_off_set_ind(begin_idx_list, off_set_ind)
  end_idx_list = apply_off_set_ind(end_idx_list, off_set_ind)
  for b,e in zip(begin_idx_list, end_idx_list):
    if e <= val_np.shape[0]:
      tmp.append(np.mean(val_np[b:e, :], axis=0))
  return np.array(tmp)

def return_rolling_sum_of_its_member(list_of_vars):
  tmp = []
  for i in list_of_vars:
    tmp.append(np.cumsum(i))
  return tmp

def return_list_with_its_members_flatten(list_of_vars):
  tmp = []
  for i in list_of_vars:
    tmp.append(np.array(i).reshape(-1, 1))
  return tmp

def get_xy_values(x, y):
  x_values = x
  y_values = np.mean(y, axis = 1)
  # y_values = np.mean(y.reshape(-1, 1), axis = 1)

  xy_values = []
  if use_time_as_x_axis:
    xy_values.append(x_values)

  xy_values.append(y_values)
  xy_values = tuple(xy_values)

  return xy_values

def draw_multiple(header_dicts, losses, aucs, aps, epoches, times, end_ind_of_init_windows,  config = {}, plot_path = None, savefig=False, use_min_length_of_list_members=False, use_time_as_x_axis=False, test_performance_on_the_same_period=False):


  assert test_performance_on_the_same_period + use_min_length_of_list_members < 2

  if test_performance_on_the_same_period:
      list_of_list_of_test_instances_end_idx = return_list_of_test_data_on_the_same_period(header_dicts)
      list_of_window_begin_idxs_with_the_same_period, list_of_window_end_idxs_with_the_same_period  = get_list_of_window_idx_with_same_period(vc.header_dicts, list_of_list_of_test_instances_end_idx)

  # assert len(losses) == 2
  # assert len(aucs) == 2
  # assert len(aps) == 2
  # assert len(epoches) == 2

  x_labels = "windows"

  if use_time_as_x_axis:
    aucs = return_list_with_its_members_flatten(aucs)
    aps = return_list_with_its_members_flatten(aps)
    epoches = return_list_with_its_members_flatten(epoches)
    losses = return_list_with_its_members_flatten(losses)
    times = return_rolling_sum_of_its_member(return_list_with_its_members_flatten(times))
    x_labels = "seconds"


  if use_min_length_of_list_members:
    assert return_min_length_of_list_members(aucs) == return_min_length_of_list_members(losses)
    min_length = return_min_length_of_list_members(losses)

  else:
    min_length = 999999


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

  for i, (loss, t) in enumerate(zip(losses, times)):
    if test_performance_on_the_same_period:
      # :NOTE: assume that len of auc is the same as total number of window
      loss = get_mean_val_from_same_period(loss, end_ind_of_init_windows[i], list_of_window_begin_idxs_with_the_same_period[i], list_of_window_end_idxs_with_the_same_period[i])
      t = get_mean_val_from_same_period(t, end_ind_of_init_windows[i], list_of_window_begin_idxs_with_the_same_period[i], list_of_window_end_idxs_with_the_same_period[i])

      # t = np.array(t)[list_of_window_begin_idxs_with_the_same_period[i]]

    if use_min_length_of_list_members:
      loss = loss[:min_length]
      t = t[:min_length]

    axs[0].plot(*get_xy_values(t, loss), label = f'loss_{i}')
    axs[0].set_xlabel(x_labels)
    # axs[0].plot( np.mean(loss, axis = 1), label = f'loss_{i}')


  # auc
  # axs[1].plot( np.zeros(min_length))
  # axs[1].plot( auc[:,-1], label = 'auc')
  for i, (auc, t) in enumerate(zip(aucs, times)):
    if test_performance_on_the_same_period:
      # :NOTE: assume that len of auc is the same as total number of window
      auc = get_mean_val_from_same_period(auc, end_ind_of_init_windows[i], list_of_window_begin_idxs_with_the_same_period[i], list_of_window_end_idxs_with_the_same_period[i])

      t = get_mean_val_from_same_period(t, end_ind_of_init_windows[i], list_of_window_begin_idxs_with_the_same_period[i], list_of_window_end_idxs_with_the_same_period[i])
      # t = np.array(t)[list_of_window_begin_idxs_with_the_same_period[i]]

    if use_min_length_of_list_members:
      auc = auc[:min_length]
      t = t[:min_length]

    axs[1].plot( *get_xy_values(t, auc), label = f'auc_{i}')
    axs[1].set_xlabel(x_labels)
    # axs[1].plot( np.mean(auc[:min_length], axis = 1), label = f'auc_{i}')

  if ylim is not None:
    axs[1].set_ylim(*next(ylim))
  else:
    # axs[1].set_ylim(auc.min(), auc.max())
    axs[1].set_ylim(0, 1)

  # ap
  # axs[2].plot( np.zeros(min_length))
  # axs[2].plot( ap[:,-1], label = 'ap')
  for i, (ap, t) in enumerate(zip(aps, times)):
    if test_performance_on_the_same_period:
      # :NOTE: assume that len of auc is the same as total number of window
      ap = get_mean_val_from_same_period(ap, end_ind_of_init_windows[i], list_of_window_begin_idxs_with_the_same_period[i], list_of_window_end_idxs_with_the_same_period[i])

      t = get_mean_val_from_same_period(t, end_ind_of_init_windows[i], list_of_window_begin_idxs_with_the_same_period[i], list_of_window_end_idxs_with_the_same_period[i])
      # t = np.array(t)[list_of_window_begin_idxs_with_the_same_period[i]]
    if use_min_length_of_list_members:
      ap = ap[:min_length]
      t = t[:min_length]
    axs[2].plot( *get_xy_values(t, ap), label = f'ap_{i}')
    axs[2].set_xlabel(x_labels)
    # axs[2].plot( np.mean(ap[:min_length], axis = 1), label = f'ap_{i}')
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

def draw(loss, auc, ap, epoch , config = {}, plot_path = None, savefig=False, is_axis_1_as_x=False, use_mean_val=False, is_ensemble=False, is_individual_model_in_ensemble=False):
  drawing = Draw(savefig=savefig,plot_path=plot_path)

  drawing.set_constraint(
    is_axis_1_as_x=is_axis_1_as_x,
    use_mean_val=use_mean_val,
    is_ensemble=is_ensemble,
    is_individual_model_in_ensemble=is_individual_model_in_ensemble,
  )
  drawing.prep_draw()
  drawing.draw(loss, auc,ap, epoch, config)

#   ylim = None
#   if len(config) > 0:
#     if 'ylim' in config:
#       assert len(config["ylim"]) == 2
#       assert isinstance(config["ylim"][0], tuple)
#       assert len(config["ylim"][0]) == 2
#       ylim = iter(config["ylim"]) # list((min, max), (min,max))

#   if is_individual_model_in_ensemble and is_ensemble:
#     raise NotImplementedError()

#   # plot_path
#   suffix = []
#   if is_axis_1_as_x:
#     suffix.append('is_axis_1_as_x')
#   if use_mean_val:
#     suffix.append('use_mean_val')
#   if is_ensemble:
#     suffix.append('ensemble')
#   if is_individual_model_in_ensemble:
#     suffix.append('all_individual_model_in_ensemble')

#   suffix = '_'.join(suffix)
#   plot_path = plot_path.split('.')[:-1]
#   plot_path = '.'.join(plot_path)
#   plot_path = '_'.join([plot_path,suffix]) + '.png'

#   x_len = np.array(loss).shape[1] if is_axis_1_as_x else np.array(loss).shape[0]
#   fig, axs = plt.subplots(1, 3, figsize=(9, 3))

#   # axs[0].plot( loss[:,-1], label = 'loss')
#   if is_ensemble:
#     if is_axis_1_as_x or use_mean_val:
#       raise NotImplementedError()
#     axs[0].plot( loss[:,-1], label = 'loss')
#   elif is_individual_model_in_ensemble:
#     if is_axis_1_as_x or use_mean_val:
#       raise NotImplementedError()
#     for ind, i in enumerate(np.array(loss)):
#       axs[0].plot( i, label = f'loss_{ind}')
#   else:
#     if not is_axis_1_as_x:
#       if use_mean_val:
#         axs[0].plot( np.mean(loss, axis = 1), label = 'loss')
#       else:
#         for ind, i in enumerate(np.array(loss)):
#           axs[0].plot( i, label = f'loss_{ind}')
#     else:
#       raise NotImplementedError
#       if use_mean_val:
#         raise NotImplementedError()
#       for ind, i in enumerate(np.array(loss).T):
#         axs[0].plot( i, label = f'loss_model_{ind}')
#   # axs[0].plot( loss, label = 'loss')

#   # auc
#   # axs[1].plot(np.zeros(len(loss.T)))
#   axs[1].plot(np.zeros(x_len))
#   # axs[1].plot( auc[:,-1], label = 'auc')
#   if is_ensemble:
#     if is_axis_1_as_x or use_mean_val:
#       raise NotImplementedError()
#     axs[1].plot( auc[:,-1], label = 'auc')
#   elif is_individual_model_in_ensemble:
#     if is_axis_1_as_x or use_mean_val:
#       raise NotImplementedError()
#     for ind, i in enumerate(np.array(auc)):
#       axs[1].plot( i, label = f'auc_{ind}')
#   else:
#     if not is_axis_1_as_x:
#       if use_mean_val:
#         axs[1].plot( np.mean(auc, axis = 1), label = 'auc')
#       else:
#         for ind, i in enumerate(np.array(auc)):
#           axs[1].plot( i, label = f'auc_{ind}')
#     else:
#       raise NotImplementedError
#       if use_mean_val:
#         raise NotImplementedError()
#       for ind, i in enumerate(np.array(auc).T):
#         axs[1].plot( i, label = f'auc_model_{ind}')

#   # if not is_axis_1_as_x:
#   #   axs[1].plot( np.mean(auc, axis = 1), label = 'auc')
#   # else:
#   #   for ind, i in enumerate(np.array(auc)):
#   #     axs[1].plot( i, label = f'auc_window_{ind}')

#   if ylim is not None:
#     axs[1].set_ylim(*next(ylim))
#   else:
#     axs[1].set_ylim(auc.min(), auc.max())

#   # ap
#   # axs[2].plot( np.zeros(len(loss)))
#   axs[2].plot( np.zeros(x_len))
#   # axs[2].plot( ap[:,-1], label = 'ap')

#   if is_ensemble:
#     if is_axis_1_as_x or use_mean_val:
#       raise NotImplementedError()
#     axs[2].plot( ap[:,-1], label = 'auc')
#   elif is_individual_model_in_ensemble:
#     if is_axis_1_as_x or use_mean_val:
#       raise NotImplementedError()
#     for ind, i in enumerate(np.array(ap)):
#       axs[2].plot( i, label = f'ap_{ind}')
#   else:
#     if not is_axis_1_as_x:
#       if use_mean_val:
#         axs[2].plot( np.mean(ap, axis = 1), label = 'auc')
#       else:
#         for ind, i in enumerate(np.array(ap)):
#           axs[2].plot( i, label = f'ap_{ind}')
#     else:
#       raise NotImplementedError
#       if use_mean_val:
#         raise NotImplementedError()
#       for ind, i in enumerate(np.array(ap).T):
#         axs[2].plot( i, label = f'ap_model_{ind}')

#   # if not is_axis_1_as_x:
#   #   axs[2].plot( np.mean(ap, axis = 1), label = 'ap')
#   # else:
#   #   for ind, i in enumerate(np.array(ap)):
#   #     axs[2].plot( i, label = f'ap_window_{ind}')

#   if ylim is not None:
#     axs[2].set_ylim(*next(ylim))
#   else:
#     axs[2].set_ylim(ap.min(), ap.max())

#   fig.suptitle('model performance')
#   axs[0].legend()
#   axs[1].legend()
#   axs[2].legend()

#   print(plot_path)
#   if savefig:
#     assert plot_path is not None
#     plt.savefig(plot_path)
#   else:
#     plt.show()


# class LinkPredictionCrawler(Crawler):
#   def __init__(self, log_file, suffix=''):
#     base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
#     self.log_path = str(base_path / f'log/{log_file}.log')
#     self.nodes_and_edges_weight_log_path = str(base_path / f'log/nodes_and_edges_weight/{log_file}.log')
#     if len(suffix) > 0:
#       self.plot_path = str(base_path / f'plot/{log_file}_{suffix}.png')
#     else:
#       self.plot_path = str(base_path / f'plot/{log_file}.png')

#     header_dict, loss, auc, ap, epoch, ws, times, complete_ws_len = self.crawl_data()
#     # self.crawl_data_v2(nodes_and_edges_weight_log_path)

#     self.loss = loss
#     self.auc = auc
#     self.ap = ap
#     self.epoch = epoch
#     self.ws = ws
#     self.times = times
#     self.complete_ws_len = complete_ws_len
#     self.log_file = log_file
#     self.header_dict = header_dict
#     self.end_idx_of_init_window = 1 # :NOTE: this only works until init_window_size is not the same as batch_size

#   def plot(self, savefig=False,is_axis_1_as_x=False, use_mean_val=False, is_ensemble=False,is_individual_model_in_ensemble=False ):
#     draw(self.loss, self.auc, self.ap, self.epoch , plot_path=self.plot_path, savefig=savefig, is_axis_1_as_x=is_axis_1_as_x, use_mean_val=use_mean_val, is_ensemble=is_ensemble, is_individual_model_in_ensemble=is_individual_model_in_ensemble)

#   def get_return(self):
#     raise DeprecationWarning()
#     raise NotImplementedError()
#     return self.header_dict,self.log_file, self.loss, self.auc, self.ap, self.epoch, self.times, self.end_idx_of_init_window

#   def get_ws_val(self, line):
#     ws_val = line.split(" ")[-1].rstrip()
#     return int(ws_val)

#   def get_epoch_val(self, line):
#     epoch_val = line.split(" ")[-1].rstrip()
#     return int(epoch_val)

#   def get_loss_val(self, line):
#     loss_val = line.split(" ")[-1].rstrip()
#     return float(loss_val)

#   def get_auc_val(self, line):
#     auc_val = line.split(" ")[-1].rstrip()
#     return float(auc_val)

#   def get_time_val(self, line):
#     time_val = line.split('took')[-1].split('s')[0].strip(' ')
#     return float(time_val)

#   def get_ap_val(self, line):
#     ap_val = line.split(" ")[-1].rstrip()
#     return float(ap_val)

#   def get_config_params(self, line):
#     tmp = {}
#     tmp["batch_size"] = int(line.split("bs")[-1].split(',')[0].strip("="))
#     tmp["data"] = line.split('data')[-1].split(',')[0].strip("=").strip("'")
#     return tmp.copy()

#   def get_ensemble_idx_val(self,line):
#     return int(line.split(" ")[-1].rstrip())

#   def crawl_data(self):
#     loss = []
#     auc = []
#     ap = []
#     epoch = []
#     ws = []
#     times = []
#     ensemble_idxs = []
#     header_dict = {}
#     i = 1
#     with open(self.log_path, 'r') as f:
#       for i, line in enumerate(f.readlines()): # 9341 lines in total
#         # if i == 100:
#         #   exit()

#         is_ws_exist = True if 'ws' in line and 'DEBUG' in line else False
#         is_ensemble_exist = True if 'ensemble_idx' in line else False
#         is_epoch_exist = True if 'epoch' in line and 'DEBUG' in line else False
#         is_loss_exist = True if 'mean loss' in line and 'INFO' in line else False
#         is_auc_exist = True if 'val auc' in line and 'INFO' in line else False
#         is_ap_exist = True if 'val ap' in line and 'INFO' in line else False
#         is_time = True if 'took' in line else False
#         is_header = True if "Namespace" in line else False


#         if is_ws_exist:
#             ws.append(self.get_ws_val(line))
#         if is_ensemble_exist:
#             ensemble_idxs.append(self.get_ensemble_idx_val(line))
#         if is_epoch_exist:
#             epoch.append(self.get_epoch_val(line))
#         if is_loss_exist:
#             loss.append(self.get_loss_val(line))
#         if is_auc_exist:
#             auc.append(self.get_auc_val(line))
#         if is_ap_exist:
#             ap.append(self.get_ap_val(line))
#         if is_time:
#             times.append(self.get_time_val(line))
#         if is_header:
#             # :NOTE: I didn't implement header to reflect correct parameter in the earilier logs.
#             header_dict = self.get_config_params(line)

#         # ws_val = self.get_ws_val(line)
#         # epoch_val = self.get_epoch_val(line)
#         # loss_val = self.get_loss_val(line)
#         # auc_val = self.get_auc_val(line)
#         # ap_val = self.get_ap_val(line)

#         # reset param to None to prevent side effect
#         # if loss_val is not None:
#         #   loss.append(loss_val)
#         # if auc_val is not None:
#         #   auc.append(auc_val)
#         # if ap_val is not None:
#         #   ap.append(ap_val)
#         # if epoch_val is not None:
#         #   epoch.append(epoch_val)
#         # if ws_val is not None:
#         #   ws.append(ws_val)

#     # non_missing_len = min(len(loss), len(auc), len(ap), len(epoch))
#     # max_epoch = max(epoch)
#     max_epoch = max(epoch) + 1 # NOTE: haven't test it much
#     if len(ws) > 0:
#       complete_ws_len = min((max(ws)+1) * max_epoch, len(epoch))
#       if complete_ws_len == (max(ws)+1) * max_epoch:
#         ws = ws[:-1]
#     else:
#       complete_ws_len = len(epoch)
#       # complete_ws_len = max_epoch


#     loss = np.array(loss[:complete_ws_len]).reshape(-1, max_epoch)
#     auc = np.array(auc[:complete_ws_len]).reshape(-1,max_epoch)
#     ap = np.array(ap[:complete_ws_len]).reshape(-1,max_epoch)
#     epoch = np.array(epoch[:complete_ws_len]).reshape(-1,max_epoch)
#     times = np.array(times[:complete_ws_len]).reshape(-1, max_epoch)
#     return header_dict, loss, auc, ap, epoch, ws, times, complete_ws_len

def plot_tmp(loss, auc, ap, epoch , config = {}, plot_path = None, savefig=False):
  ylim = None
  if len(config) > 0:
    if 'ylim' in config:
      assert len(config["ylim"]) == 2
      assert isinstance(config["ylim"][0], tuple)
      assert len(config["ylim"][0]) == 2
      ylim = iter(config["ylim"]) # list((min, max), (min,max))

  fig, axs = plt.subplots(1, 3, figsize=(9, 3))
  # axs[0].plot( loss[:,-1], label = 'loss')
  for ind,i in enumerate(loss):
    axs[0].plot( i, label = f'loss_{ind}')

  # auc
  axs[1].plot(np.zeros(len(loss)))
  # axs[1].plot( auc[:,-1], label = 'auc')
  for ind,i in enumerate(auc):
    axs[1].plot( i, label = f'auc_{ind}')
  # axs[1].plot( auc, label = 'auc')
  if ylim is not None:
    axs[1].set_ylim(*next(ylim))
  else:
    axs[1].set_ylim(auc.min(), auc.max())

  # ap
  axs[2].plot( np.zeros(len(loss)))
  # axs[2].plot( ap[:,-1], label = 'ap')

  for ind,i in enumerate(ap):
    axs[2].plot( i, label = f'ap_{ind}')
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

class Draw:
  def __init__(self, savefig=False, plot_path=None):
    self.savefig = savefig
    self.plot_path = plot_path
    self.config = {}

  def prep_draw(self):
    self.config = self.get_config()
    self.plot_path =  self.get_plot_path()
    self.fig, self.axs = self.set_axis()
    self.constraints = self.get_draw_constraint()

  def set_constraint(self, is_axis_1_as_x=False, use_mean_val=False, is_ensemble=False, is_individual_model_in_ensemble=False):

    self.is_axis_1_as_x= is_axis_1_as_x
    self.use_mean_val= use_mean_val
    self.is_ensemble= is_ensemble
    self.is_individual_model_in_ensemble= is_individual_model_in_ensemble

  def get_plot_path(self):
    plot_path = self.plot_path
    suffix = self.get_suffix()

    plot_path = plot_path.split('.')[:-1]
    plot_path = '.'.join(plot_path)
    plot_path = '_'.join([plot_path,suffix]) + '.png'
    return plot_path

  def get_suffix(self):
    is_axis_1_as_x = self.is_axis_1_as_x
    use_mean_val = self.use_mean_val
    is_ensemble = self.is_ensemble
    is_individual_model_in_ensemble = self.is_individual_model_in_ensemble

    suffix = []
    if is_axis_1_as_x:
      suffix.append('is_axis_1_as_x')
    if use_mean_val:
      suffix.append('use_mean_val')
    if is_ensemble:
      suffix.append('ensemble')
    if is_individual_model_in_ensemble:
      suffix.append('all_individual_model_in_ensemble')

    suffix = '_'.join(suffix)
    return suffix


  def set_axis(self):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    return fig, axs

  def get_config(self):
    config = self.config

    if len(config) > 0:
      if 'ylim' in config:
        assert len(config["ylim"]) == 2
        assert isinstance(config["ylim"][0], tuple)
        assert len(config["ylim"][0]) == 2
        ylim = iter(config["ylim"]) # list((min, max), (min,max))
        config['ylim'] = ylim

    return config

  def draw_val_on_axis(self, idx, val, label=None):
    assert label is not None
    axs = self.axs

    constraints = self.constraints

    if constraints['y_axis'] == 'mean_performance':
      val = np.mean(val, axis=1)

    # axs[0].plot( loss[:,-1], label = 'loss')
    if constraints['drawing_style'] == 'individual_models_in_ensemble':
      for ind, i in enumerate(np.array(val)):
        axs[idx].plot( i, label = f'{label}_{ind}')
    elif constraints['drawing_style'] == 'voting_ensemble':
      raise NotImplementedError
    elif constraints['drawing_style'] == 'non_ensemble':
      if constraints['is_tracking_performance_as_window_move_forward']:
        axs[idx].plot( val, label = f'{label}')
    else:
      raise NotImplementedError

    return axs

  def get_draw_constraint(self):
    """
    start using it when condition is getting outof control
    """
    is_axis_1_as_x  = self.is_axis_1_as_x
    use_mean_val  = self.use_mean_val
    is_ensemble  = self.is_ensemble
    is_individual_model_in_ensemble  = self.is_individual_model_in_ensemble

    constraints = {}

    if is_individual_model_in_ensemble and is_ensemble:
      raise NotImplementedError()

    if is_axis_1_as_x:
      raise NotImplementedError()

    # axs[0].plot( loss[:,-1], label = 'loss')
    if is_ensemble:
      if is_axis_1_as_x or use_mean_val:
        raise NotImplementedError()
      constraints['x_axis'] = 'window'
      constraints['y_axis'] = 'performance'
      constraints['drawing_style'] = 'voting_ensemble'
      constraints['is_tracking_performance_as_window_move_forward'] = False
      # axs[0].plot( loss[:,-1], label = 'loss')
    elif is_individual_model_in_ensemble:
      if is_axis_1_as_x or use_mean_val:
        raise NotImplementedError()
      constraints['x_axis'] = 'epoch'
      constraints['y_axis'] = 'performance'
      constraints['drawing_style'] = 'individual_models_in_ensemble'
      constraints['is_tracking_performance_as_window_move_forward'] = False
    else:
      constraints['drawing_style'] = 'non_ensemble'
      if not is_axis_1_as_x:
        if use_mean_val:
          constraints['x_axis'] = 'window'
          constraints['y_axis'] = 'mean_performance'
          constraints['is_tracking_performance_as_window_move_forward'] = True
        else:
          constraints['x_axis'] = 'window'
          constraints['y_axis'] = 'mean_performance'
          constraints['is_tracking_performance_as_window_move_forward'] = True
      else:
        raise NotImplementedError
        constraints['x_axis'] = 'val_on_axis_1'
        constraints['y_axis'] = 'mean_performance'
        constraints['is_tracking_performance_as_window_move_forward'] = True
    return constraints

  def get_x_len(self, val):
    x_len = np.array(val).shape[1] if self.is_axis_1_as_x else np.array(val).shape[0]
    return x_len

  def draw(self, loss, auc, ap, epoch , config = {}):

    axs = self.axs
    fig = self.fig
    savefig = self.savefig
    plot_path = self.plot_path

    ylim = None

    x_len = self.get_x_len(loss)

    if len(config) > 0:
      raise NotImplementedError()
      config = get_config()
      ylim = config['ylim']

    ## loss
    axs = self.draw_val_on_axis(0, loss, label = 'loss')

    ## auc
    idx = 1
    axs = self.draw_val_on_axis(idx, auc, label = 'auc')
    axs[idx].plot(np.zeros(x_len))
    if ylim is not None:
      axs[idx].set_ylim(*next(ylim))
    else:
      axs[idx].set_ylim(auc.min(), auc.max())

    ## ap
    idx = 2
    axs = self.draw_val_on_axis(idx, ap, label = 'ap')
    axs[idx].plot( np.zeros(x_len))
    if ylim is not None:
      axs[idx].set_ylim(*next(ylim))
    else:
      axs[idx].set_ylim(ap.min(), ap.max())

    fig.suptitle('model performance')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    print(plot_path)
    if savefig:
      assert plot_path is not None
      plt.savefig(plot_path)
    else:
      plt.show()


class LinkPredictionCrawler(Crawler):
  def __init__(self, log_file, suffix=''):
    base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
    self.log_path = str(base_path / f'log/{log_file}.log')
    self.nodes_and_edges_weight_log_path = str(base_path / f'log/nodes_and_edges_weight/{log_file}.log')
    if len(suffix) > 0:
      self.plot_path = str(base_path / f'plot/{log_file}_{suffix}.png')
    else:
      self.plot_path = str(base_path / f'plot/{log_file}.png')

    header_dict, loss, auc, ap, epoch, ws, times, complete_ws_len = self.crawl_data()
    # self.crawl_data_v2(nodes_and_edges_weight_log_path)

    self.loss = loss
    self.auc = auc
    self.ap = ap
    self.epoch = epoch
    self.ws = ws
    self.times = times
    self.complete_ws_len = complete_ws_len
    self.log_file = log_file
    self.header_dict = header_dict
    self.end_idx_of_init_window = 1 # :NOTE: this only works until init_window_size is not the same as batch_size

  def plot(self, savefig=False,is_axis_1_as_x=False, use_mean_val=False, is_ensemble=False,is_individual_model_in_ensemble=False ):
    draw(self.loss, self.auc, self.ap, self.epoch , plot_path=self.plot_path, savefig=savefig, is_axis_1_as_x=is_axis_1_as_x, use_mean_val=use_mean_val, is_ensemble=is_ensemble, is_individual_model_in_ensemble=is_individual_model_in_ensemble)

  def get_return(self):
    raise DeprecationWarning()
    raise NotImplementedError()
    return self.header_dict,self.log_file, self.loss, self.auc, self.ap, self.epoch, self.times, self.end_idx_of_init_window

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

  def get_time_val(self, line):
    time_val = line.split('took')[-1].split('s')[0].strip(' ')
    return float(time_val)

  def get_ap_val(self, line):
    ap_val = line.split(" ")[-1].rstrip()
    return float(ap_val)

  def get_config_params(self, line):
    tmp = {}
    tmp["batch_size"] = int(line.split("bs")[-1].split(',')[0].strip("="))
    tmp["data"] = line.split('data')[-1].split(',')[0].strip("=").strip("'")
    return tmp.copy()

  def get_ensemble_idx_val(self,line):
    return int(line.split(" ")[-1].rstrip())

  def crawl_data(self):
    loss = []
    auc = []
    ap = []
    epoch = []
    ws = []
    times = []
    ensemble_idxs = []
    header_dict = {}
    i = 1
    with open(self.log_path, 'r') as f:
      for i, line in enumerate(f.readlines()): # 9341 lines in total
        # if i == 100:
        #   exit()

        is_ws_exist = True if 'ws' in line and 'DEBUG' in line else False
        is_ensemble_exist = True if 'ensemble_idx' in line else False
        is_epoch_exist = True if 'epoch' in line and 'DEBUG' in line else False
        is_loss_exist = True if 'mean loss' in line and 'INFO' in line else False
        is_auc_exist = True if 'val auc' in line and 'INFO' in line else False
        is_ap_exist = True if 'val ap' in line and 'INFO' in line else False
        is_time = True if 'took' in line else False
        is_header = True if "Namespace" in line else False


        if is_ws_exist:
            ws.append(self.get_ws_val(line))
        if is_ensemble_exist:
            ensemble_idxs.append(self.get_ensemble_idx_val(line))
        if is_epoch_exist:
            epoch.append(self.get_epoch_val(line))
        if is_loss_exist:
            loss.append(self.get_loss_val(line))
        if is_auc_exist:
            auc.append(self.get_auc_val(line))
        if is_ap_exist:
            ap.append(self.get_ap_val(line))
        if is_time:
            times.append(self.get_time_val(line))
        if is_header:
            # :NOTE: I didn't implement header to reflect correct parameter in the earilier logs.
            header_dict = self.get_config_params(line)

        # ws_val = self.get_ws_val(line)
        # epoch_val = self.get_epoch_val(line)
        # loss_val = self.get_loss_val(line)
        # auc_val = self.get_auc_val(line)
        # ap_val = self.get_ap_val(line)

        # reset param to None to prevent side effect
        # if loss_val is not None:
        #   loss.append(loss_val)
        # if auc_val is not None:
        #   auc.append(auc_val)
        # if ap_val is not None:
        #   ap.append(ap_val)
        # if epoch_val is not None:
        #   epoch.append(epoch_val)
        # if ws_val is not None:
        #   ws.append(ws_val)

    # non_missing_len = min(len(loss), len(auc), len(ap), len(epoch))
    # max_epoch = max(epoch)
    max_epoch = max(epoch) + 1 # NOTE: haven't test it much
    if len(ws) > 0:
      complete_ws_len = min((max(ws)+1) * max_epoch, len(epoch))
      if complete_ws_len == (max(ws)+1) * max_epoch:
        ws = ws[:-1]
    else:
      complete_ws_len = len(epoch)
      # complete_ws_len = max_epoch


    loss = np.array(loss[:complete_ws_len]).reshape(-1, max_epoch)
    auc = np.array(auc[:complete_ws_len]).reshape(-1,max_epoch)
    ap = np.array(ap[:complete_ws_len]).reshape(-1,max_epoch)
    epoch = np.array(epoch[:complete_ws_len]).reshape(-1,max_epoch)
    times = np.array(times[:complete_ws_len]).reshape(-1, max_epoch)
    return header_dict, loss, auc, ap, epoch, ws, times, complete_ws_len

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


class ValueCollection():
  def __init__(self):
    self.losses = []
    self.aucs = []
    self.aps = []
    self.epoches = []
    self.times = []
    self.log_files = []
    self.header_dicts = []
    self.end_ind_of_init_windows  = []

  def append_value(self, crawler):

    self.header_dicts.append(crawler.header_dict)
    self.log_files.append(crawler.log_file)
    self.losses.append(crawler.loss)
    self.aucs.append(crawler.auc)
    self.aps.append(crawler.ap)
    self.epoches.append(crawler.epoch)
    self.times.append(crawler.times)
    self.end_ind_of_init_windows.append(crawler.end_idx_of_init_window)


def get_list_of_batch_size(header_dicts):
  return [i["batch_size"] for i in header_dicts]

def get_largest_batch_size_from_list(header_dicts):
  max_bs = max(get_list_of_batch_size(header_dicts))
  for i in header_dicts:
    assert max_bs/i['batch_size'] == int(max_bs/i['batch_size'])
  return max_bs

def get_data_size(header_dicts):
  tmp = None
  for idx,i in enumerate(header_dicts):
    if i['data'] == 'reddit_10000':
      data_size = 10000
    elif i['data'] == 'wikipedia_10000':
      data_size = 10000
    elif i['data'] == 'mooc_10000':
      data_size = 10000
    elif i['data'] == 'lastfm_10000':
      data_size = 10000
    else:
      raise NotImplementedError()
    if idx > 0:
      assert tmp == data_size
    tmp = data_size

  return data_size

def get_number_of_window(data_size, bs):
  # :NOTE: this will need to be change when I implement window size to be independent of batch_size (by then I should be able to get window size from header.)
  tmp = data_size/bs
  assert tmp == int(tmp)
  return int(tmp)

def list_of_test_instances_end_idx(bs, total_number_of_window):
  return [(bs * (i+1)) for i in range(total_number_of_window)] # assume that idx start at 1

def return_list_of_instance_end_idx_with_max_batch_size(header_dicts):
  max_bs = get_largest_batch_size_from_list(header_dicts)
  data_size = get_data_size(header_dicts)
  total_number_of_window = get_number_of_window(data_size, max_bs)
  return list_of_test_instances_end_idx(max_bs, total_number_of_window)

# def get_test_data_on_the_same_period(header_dicts):
def return_list_of_test_data_on_the_same_period(header_dicts):
  return return_list_of_instance_end_idx_with_max_batch_size(header_dicts)

def get_list_of_window_idx_with_same_period(header_dicts, list_of_test_end_idx_on_the_same_period):
  """
  :NOTE: I should consider starting to refactor this function when I find it difficult to conceptualize/extend/modify the function.
  """
  data_size = get_data_size(header_dicts)
  max_bs = get_largest_batch_size_from_list(header_dicts)
  # list_of_window_idx_with_same_period = []
  list_of_window_begin_idx_with_same_period = []
  list_of_window_end_idx_with_same_period = []

  for idx, i in enumerate(header_dicts):
    bs = i['batch_size']
    multiple_bs = int(max_bs/bs)
    # multiple_bs
    total_number_of_window = get_number_of_window(data_size,bs)
    test_instances_end_idx_list = list_of_test_instances_end_idx(bs, total_number_of_window)
    window_idx_list = list(range(len(test_instances_end_idx_list)))

    window_begin_idx_with_same_period = []
    window_end_idx_with_same_period = []

    for j in list_of_test_end_idx_on_the_same_period:
      arange_idx_list = list(range(len(test_instances_end_idx_list)))
      t_ = np.array((test_instances_end_idx_list, list(map(lambda x: x+ 1, arange_idx_list)))).T
      # t_ = np.array((test_instances_end_idx_list, list(map(lambda x: x+ 1, range(len(test_instances_end_idx_list)))))).T
      tmp = list(filter(lambda x: j == x[0], t_ ))[0][1]

      if tmp != 0 and (tmp + multiple_bs) * bs <= data_size:
      # if (tmp + multiple_bs) * bs <= data_size:
        window_begin_idx_with_same_period.append(tmp)
        window_end_idx_with_same_period.append(tmp + multiple_bs)

      # for idxx, jj in enumerate(test_instances_end_idx_list):
        # if jj > j:
        #   assert idxx - 1 >= 0
        #   list_of_window_idx_with_same_period.append(idxx)
        #   break

    assert len(window_begin_idx_with_same_period) == len(window_end_idx_with_same_period)
    # :NOTE: I patched the error by excluding last element because reddit_10000 has 9999 instances not 10000.
    # list_of_window_begin_idx_with_same_period.append(window_begin_idx_with_same_period[:-1])
    # list_of_window_end_idx_with_same_period.append(window_end_idx_with_same_period[:-1])
    list_of_window_begin_idx_with_same_period.append(window_begin_idx_with_same_period)
    list_of_window_end_idx_with_same_period.append(window_end_idx_with_same_period)


  for x,y in zip(list_of_window_begin_idx_with_same_period, list_of_window_end_idx_with_same_period):
    # :NOTE: I patched the error by excluding last element because reddit_10000 has 9999 instances not 10000.
    assert len(x) == len(y)

    assert len(list_of_test_end_idx_on_the_same_period) in [len(x) + 1 , len(x)]


  return list_of_window_begin_idx_with_same_period, list_of_window_end_idx_with_same_period



if __name__ == "__main__":

  losses, aucs, aps, epoches, times= [], [], [] ,[], []
  log_files = []
  vc = ValueCollection()


  # log_file = '1640813396.4925542' # use_weight = True + sigmoid
  # c = LinkPredictionCrawler(log_file)
  # # c.plot()
  # losses.append(c.get_return()[0])
  # aucs.append(c.get_return()[1])
  # aps.append(c.get_return()[2])
  # epoches.append(c.get_return()[3])
  # pos_edges_weight = c.crawl_data_v2(c.nodes_and_edges_weight_log_path)
  # col_1 = pos_edges_weight.reshape(-1)
  # col_2 = np.array([[i for _ in range(pos_edges_weight.shape[1])] for i in range(pos_edges_weight.shape[0])]).reshape(-1)
  # col_name = ['weight', 'batch_idx']
  # pos_edges_weight_pd = pd.DataFrame(np.array([col_1, col_2]).T, columns=col_name)
  # pos_edges_weight_pd['batch_idx'] = pos_edges_weight_pd['batch_idx'].astype(int)
  # sns.boxplot(x="batch_idx", y="weight", data=pos_edges_weight_pd)
  # plt.show()


  # # log_file = '1640232708.151559'
  # log_file = '1640243549.1885495' # original; use_weight = False ; batch size = 200
  # c2 = LinkPredictionCrawler(log_file)
  # # c2.plot()
  # vc.append_value(c2)

  # # log_file = '1640232708.151559'
  # # log_file = '1640298667.0700464' # use_weight = True
  # log_file = '1640371085.9001596' # use_weight = True + correct ef-iwf implementation. + no sigmoid
  # # log_file = '1640381726.2764423' # use_weight = True + correct ef-iwf implementation. + sigmoid
  # c3 = LinkPredictionCrawler(log_file)
  # # c3.plot()
  # losses.append(c3.get_return()[0])
  # aucs.append(c3.get_return()[1])
  # aps.append(c3.get_return()[2])
  # epoches.append(c3.get_return()[3])

  # # log_file = '1640898429.909531' # ef-iwf where ef term is computed as raw_count. (run on 10k instances, run 1 time)
  # log_file = '1640905537.015449' # ef-iwf where ef term is computed as raw_count. (run on 10k instances; run 5 times)
  # c4 = LinkPredictionCrawler(log_file)
  # # c4.plot()
  # losses.append(c4.get_return()[0])
  # aucs.append(c4.get_return()[1])
  # aps.append(c4.get_return()[2])
  # epoches.append(c4.get_return()[3])
  # # pos_edges_weight = c4.crawl_data_v2(c4.nodes_and_edges_weight_log_path)
  # # col_1 = pos_edges_weight.reshape(-1)
  # # col_2 = np.array([[i for _ in range(pos_edges_weight.shape[1])] for i in range(pos_edges_weight.shape[0])]).reshape(-1)
  # # col_name = ['weight', 'batch_idx']
  # # pos_edges_weight_pd = pd.DataFrame(np.array([col_1, col_2]).T, columns=col_name)
  # # pos_edges_weight_pd['batch_idx'] = pos_edges_weight_pd['batch_idx'].astype(int)
  # # base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  # # plot_path = str(base_path / 'plot/edges_weight_1640898429-909531.png')
  # # sns.boxplot(x="batch_idx", y="weight", data=pos_edges_weight_pd)
  # # plt.savefig(plot_path)
  # # # plt.show()

  # log_file = '1640814430.1087918' # original ef-iwf (run on 100k)
  # c5 = LinkPredictionCrawler(log_file)
  # # c4.plot()
  # losses.append(c5.get_return()[0])
  # aucs.append(c5.get_return()[1])
  # aps.append(c5.get_return()[2])
  # epoches.append(c5.get_return()[3])

  # log_file = '1640977751.9673245' # ef-iwf where ef term is computed as raw_count. (run on 100k instances)
  # c6 = LinkPredictionCrawler(log_file)
  # # c6.plot()
  # losses.append(c6.get_return()[0])
  # aucs.append(c6.get_return()[1])
  # aps.append(c6.get_return()[2])
  # epoches.append(c6.get_return()[3])
  # log_files.append(log_file)

  # log_file = '1641232336.596783' # original (run on 100k reddit instances)
  # c7 = LinkPredictionCrawler(log_file)
  # # c7.plot()
  # losses.append(c7.get_return()[0])
  # aucs.append(c7.get_return()[1])
  # aps.append(c7.get_return()[2])
  # epoches.append(c7.get_return()[3])
  # log_files.append(log_file)

  # log_file = '1641246007.212651' # negative sampling using nf_iwf where nf is raw_count (run on 10k instances)
  # c8 = LinkPredictionCrawler(log_file)
  # # c7.plot()
  # losses.append(c8.get_return()[0])
  # aucs.append(c8.get_return()[1])
  # aps.append(c8.get_return()[2])
  # epoches.append(c8.get_return()[3])
  # log_files.append(log_file)

  # log_file = '1641252200.886314' # negative sampling using nf_iwf where nf is raw_count (run on 100k instances)
  # c9 = LinkPredictionCrawler(log_file)
  # # c7.plot()
  # losses.append(c9.get_return()[0])
  # aucs.append(c9.get_return()[1])
  # aps.append(c9.get_return()[2])
  # epoches.append(c9.get_return()[3])
  # log_files.append(log_file)
  # pos_edges_weight = c9.crawl_data_v2(c9.nodes_and_edges_weight_log_path)
  # col_1 = pos_edges_weight.reshape(-1)
  # col_2 = np.array([[i for _ in range(pos_edges_weight.shape[1])] for i in range(pos_edges_weight.shape[0])]).reshape(-1)
  # col_name = ['weight', 'batch_idx']
  # pos_edges_weight_pd = pd.DataFrame(np.array([col_1, col_2]).T, columns=col_name)
  # pos_edges_weight_pd['batch_idx'] = pos_edges_weight_pd['batch_idx'].astype(int)
  # base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  # plot_path = str(base_path / 'plot/edges_weight_1640898429-909531.png')
  # sns.boxplot(x="batch_idx", y="weight", data=pos_edges_weight_pd)
  # # plt.savefig(plot_path)
  # plt.show()

  # log_file = '1641426436.864333' # model with random weight + weight range from 0 to 5
  # c10 = LinkPredictionCrawler(log_file)
  # # c10.plot()
  # losses.append(c10.get_return()[0])
  # aucs.append(c10.get_return()[1])
  # aps.append(c10.get_return()[2])
  # epoches.append(c10.get_return()[3])
  # log_files.append(log_file)
  # # pos_edges_weight = c10.crawl_data_v2(c10.nodes_and_edges_weight_log_path)
  # # col_1 = pos_edges_weight.reshape(-1)
  # # col_2 = np.array([[i for _ in range(pos_edges_weight.shape[1])] for i in range(pos_edges_weight.shape[0])]).reshape(-1)
  # # col_name = ['weight', 'batch_idx']
  # # pos_edges_weight_pd = pd.DataFrame(np.array([col_1, col_2]).T, columns=col_name)
  # # pos_edges_weight_pd['batch_idx'] = pos_edges_weight_pd['batch_idx'].astype(int)
  # # base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  # # plot_path = str(base_path / 'plot/edges_weight_1640898429-909531.png')
  # # sns.boxplot(x="batch_idx", y="weight", data=pos_edges_weight_pd)
  # # # plt.savefig(plot_path)
  # # plt.show()


  # log_file = '1641588473.0998917' # model where =weighted_loss_method= is =share_selected_random_weight_per_window= + weight range is 0 - 500
  # c11 = LinkPredictionCrawler(log_file)
  # # c11.plot()
  # losses.append(c11.get_return()[0])
  # aucs.append(c11.get_return()[1])
  # aps.append(c11.get_return()[2])
  # epoches.append(c11.get_return()[3])
  # log_files.append(log_file)
  # # pos_edges_weight = c11.crawl_data_v2(c11.nodes_and_edges_weight_log_path)
  # # col_1 = pos_edges_weight.reshape(-1)
  # # col_2 = np.array([[i for _ in range(pos_edges_weight.shape[1])] for i in range(pos_edges_weight.shape[0])]).reshape(-1)
  # # col_name = ['weight', 'batch_idx']
  # # pos_edges_weight_pd = pd.DataFrame(np.array([col_1, col_2]).T, columns=col_name)
  # # pos_edges_weight_pd['batch_idx'] = pos_edges_weight_pd['batch_idx'].astype(int)
  # # base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  # # plot_path = str(base_path / f'plot/edges_weight_{log_file}.png')
  # # sns.boxplot(x="batch_idx", y="weight", data=pos_edges_weight_pd)
  # # # plt.savefig(plot_path)
  # # plt.show()

  # log_file = '1641597198.5721123' # model where =weighted_loss_method= is =share_selected_random_weight_per_window= + weight range is 0 - 500
  # c12 = LinkPredictionCrawler(log_file)
  # # c12.plot()
  # losses.append(c12.get_return()[0])
  # aucs.append(c12.get_return()[1])
  # aps.append(c12.get_return()[2])
  # epoches.append(c12.get_return()[3])
  # log_files.append(log_file)
  # # pos_edges_weight = c12.crawl_data_v2(c12.nodes_and_edges_weight_log_path)
  # # col_1 = pos_edges_weight.reshape(-1)
  # # col_2 = np.array([[i for _ in range(pos_edges_weight.shape[1])] for i in range(pos_edges_weight.shape[0])]).reshape(-1)
  # # col_name = ['weight', 'batch_idx']
  # # pos_edges_weight_pd = pd.DataFrame(np.array([col_1, col_2]).T, columns=col_name)
  # # pos_edges_weight_pd['batch_idx'] = pos_edges_weight_pd['batch_idx'].astype(int)
  # # base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  # # plot_path = str(base_path / f'plot/edges_weight_{log_file}.png')
  # # sns.boxplot(x="batch_idx", y="weight", data=pos_edges_weight_pd)
  # # # plt.savefig(plot_path)
  # # plt.show()

  # log_file = '1641977944.7344809' # model with =use_random_weight_to_benchmark_ef_iwf_1= flag is True + weight range is 0 - 5
  # c13 = LinkPredictionCrawler(log_file)
  # # c10.plot()
  # losses.append(c13.get_return()[0])
  # aucs.append(c13.get_return()[1])
  # aps.append(c13.get_return()[2])
  # epoches.append(c13.get_return()[3])
  # log_files.append(log_file)
  # # pos_edges_weight = c10.crawl_data_v2(c10.nodes_and_edges_weight_log_path)
  # # col_1 = pos_edges_weight.reshape(-1)
  # # col_2 = np.array([[i for _ in range(pos_edges_weight.shape[1])] for i in range(pos_edges_weight.shape[0])]).reshape(-1)
  # # col_name = ['weight', 'batch_idx']
  # # pos_edges_weight_pd = pd.DataFrame(np.array([col_1, col_2]).T, columns=col_name)
  # # pos_edges_weight_pd['batch_idx'] = pos_edges_weight_pd['batch_idx'].astype(int)
  # # base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  # # plot_path = str(base_path / 'plot/edges_weight_1640898429-909531.png')
  # # sns.boxplot(x="batch_idx", y="weight", data=pos_edges_weight_pd)
  # # # plt.savefig(plot_path)
  # # plt.show()

  # log_file = '1642416998.4788427' # original + epoch 50 + batch size 1000
  # c14 = LinkPredictionCrawler(log_file)
  # # c14.plot()
  # losses.append(c14.get_return()[0])
  # aucs.append(c14.get_return()[1])
  # aps.append(c14.get_return()[2])
  # epoches.append(c14.get_return()[3])
  # times.append(c14.get_return()[4])
  # log_files.append(log_file)

  # log_file = '1642421136.658543' # original + epoch 20 + batch size 1000
  # c15 = LinkPredictionCrawler(log_file)
  # # c15.plot()
  # losses.append(c15.get_return()[0])
  # aucs.append(c15.get_return()[1])
  # aps.append(c15.get_return()[2])
  # epoches.append(c15.get_return()[3])
  # times.append(c15.get_return()[4])
  # log_files.append(log_file)

  # log_file = '1642416734.1726859' # original + epoch 5 + batch size 1000
  # c16 = LinkPredictionCrawler(log_file)
  # # c16.plot()
  # vc.append_value(c16)

  # log_file = '1642416474.9904623' # original + epoch 5 + batch size 2000
  # c17 = LinkPredictionCrawler(log_file)
  # # c17.plot()
  # vc.append_value(c17)


  # log_file = '1642432356.5350947' # original + epoch 3 + batch size 1000
  # c18 = LinkPredictionCrawler(log_file)
  # # c15.plot()
  # losses.append(c18.get_return()[0])
  # aucs.append(c18.get_return()[1])
  # aps.append(c18.get_return()[2])
  # epoches.append(c18.get_return()[3])
  # times.append(c18.get_return()[4])
  # log_files.append(log_file)

  # log_file = '1642599016.4373028' # original + epoch 5 + batch size 1000 + ef_iwf * 1
  # c19 = LinkPredictionCrawler(log_file)
  # # c17.plot()
  # vc.append_value(c19)

  # log_file = '1642598633.3073912' # original + epoch 5 + batch size 1000 + ef_iwf * 50
  # c20 = LinkPredictionCrawler(log_file)
  # # c17.plot()
  # vc.append_value(c20)

  # log_file = '1642599792.0887227' # original + epoch 5 + batch size 1000 + ef_iwf * 500
  # c21 = LinkPredictionCrawler(log_file)
  # # c17.plot()
  # vc.append_value(c21)

  # log_file = '1642600251.3233526' # original + epoch 5 + batch size 1000 + ef_iwf * 0.1
  # c22 = LinkPredictionCrawler(log_file)
  # # c17.plot()
  # vc.append_value(c22)

  # log_file = '1642763194.1611047' # inverse_ef-iwf * 1 + epoch 5 + batch size 1000
  # c23 = LinkPredictionCrawler(log_file)
  # # c17.plot()
  # vc.append_value(c23)

  # log_file = '1642763750.673035' # inverse_ef-iwf * 50 + epoch 5 + batch size 1000
  # c24 = LinkPredictionCrawler(log_file)
  # # c17.plot()
  # vc.append_value(c24)

  # log_file = '1642764868.0464556' # inverse_ef-iwf * 500 + epoch 5 + batch size 1000
  # c25 = LinkPredictionCrawler(log_file)
  # # c17.plot()
  # vc.append_value(c25)

  # log_file = '1642766439.4182317' # inverse_ef-iwf * 0.1 + epoch 5 + batch size 1000
  # c26 = LinkPredictionCrawler(log_file)
  # # c17.plot()
  # vc.append_value(c26)


  # log_file = '1643032949.2657177' # ensemble incomplete
  # c27 = LinkPredictionCrawler(log_file)
  # # c27.plot()
  # plot_tmp(c27.loss, c27.auc, c27.ap, c27.epoch , c27.plot_path, savefig=False)
  # vc.append_value(c27)

  # log_file = '1643034547.8263297' # ensemble incomplete
  # c28 = LinkPredictionCrawler(log_file)
  # # c27.plot()
  # plot_tmp(c28.loss, c28.auc, c28.ap, c28.epoch , c28.plot_path, savefig=False)
  # vc.append_value(c28)

  # log_file = '1643034979.244234' # ensemble incomplete + 50 epoch
  # c29 = LinkPredictionCrawler(log_file)
  # # c27.plot()
  # plot_tmp(c29.loss, c29.auc, c29.ap, c29.epoch , c29.plot_path, savefig=False)
  # vc.append_value(c29)


  # log_file = '1643206992.5413334' #
  # c30 = LinkPredictionCrawler(log_file)
  # c30.plot()
  # # plot_tmp(c30.loss, c30.auc, c30.ap, c30.epoch , c30.plot_path, savefig=False)
  # vc.append_value(c30)

  # log_file = '1643207628.7793047' #
  # c31 = LinkPredictionCrawler(log_file)
  # c31.plot()
  # # plot_tmp(c30.loss, c30.auc, c30.ap, c30.epoch , c30.plot_path, savefig=False)
  # vc.append_value(c31)

  # log_file = '1643207872.3252301' #
  # c32 = LinkPredictionCrawler(log_file)
  # c32.plot()
  # # plot_tmp(c30.loss, c30.auc, c30.ap, c30.epoch , c30.plot_path, savefig=False)
  # vc.append_value(c32)

  # wikipedia_10000 + forward + ws_multiplier = 1 + batch_size = 1000
  # log_file = '1643207628.7793047' #
  # c33 = LinkPredictionCrawler(log_file)
  # # c33.plot()
  # # plot_tmp(c30.loss, c30.auc, c30.ap, c30.epoch , c30.plot_path, savefig=False)
  # vc.append_value(c33)

  # ## lastfm_10000 + forward + ws_multiplier = 1 + batch_size = 1000
  # log_file = '1643208084.6326253' #
  # c34 = LinkPredictionCrawler(log_file)
  # # c34.plot()
  # # plot_tmp(c30.loss, c30.auc, c30.ap, c30.epoch , c30.plot_path, savefig=False)
  # vc.append_value(c34)

  # ## mooc_10000 + forward + ws_multiplier = 1 + batch_size = 1000
  # log_file = '1643207872.3252301' #
  # c35 = LinkPredictionCrawler(log_file)
  # # c35.plot()
  # # plot_tmp(c30.loss, c30.auc, c30.ap, c30.epoch , c30.plot_path, savefig=False)
  # vc.append_value(c35)

  # ## lastfm_10000 + ensemble + ws_multiplier = 1 + batch_size = 1000
  # log_file = '1643208229.8289232' #
  # c36 = LinkPredictionCrawler(log_file)
  # # c36.plot()
  # # plot_tmp(c30.loss, c30.auc, c30.ap, c30.epoch , c30.plot_path, savefig=False)
  # vc.append_value(c36)

  # ## mooc_10000 + ensemble + ws_multiplier = 1 + batch_size = 1000
  # log_file = '1643208291.8184156' #
  # c37 = LinkPredictionCrawler(log_file)
  # # c37.plot()
  # # plot_tmp(c30.loss, c30.auc, c30.ap, c30.epoch , c30.plot_path, savefig=False)
  # vc.append_value(c37)

  # ## wikipedia_10000 + ensemble + ws_multiplier = 1 + batch_size = 1000
  # log_file = '1643208291.8184156' #
  # c38 = LinkPredictionCrawler(log_file)
  # # c38.plot()
  # # plot_tmp(c30.loss, c30.auc, c30.ap, c30.epoch , c30.plot_path, savefig=False)
  # vc.append_value(c38)

  ## reddit_10000 + ensemble + ws_multiplier = 1 + batch_size = 1000 + epoch= 100
  log_file = '1643210480.2940898' #
  # suffix = 'epoch_wise'
  c39 = LinkPredictionCrawler(log_file)
  # c39 = LinkPredictionCrawler(log_file, suffix=suffix)
  # plot_path = str(base_path / f'plot/edges_weight_{log_file}.png')
  # c39.plot(savefig=True, is_ensemble=True)
  # c39.plot(savefig=False, is_individual_model_in_ensemble=True)
  # c39.plot(savefig=False, is_individual_model_in_ensemble=True)
  c39.plot(savefig=False)
  # plot_tmp(c39.loss, c39.auc, c39.ap, c39.epoch , c39.plot_path, savefig=False)
  vc.append_value(c39)
  exit()

  # # reddit_10000 + forward + ws_multiplier = 1 + batch_size = 1000 + epoch = 5
  # log_file = '1643215872.0304363' #
  # c40 = LinkPredictionCrawler(log_file)
  # c40.plot(savefig=False, is_axis_1_as_x=True)
  # # plot_tmp(c40.loss, c40.auc, c40.ap, c40.epoch , c40.plot_path, savefig=False)
  # vc.append_value(c40)

  ## reddit_10000 + ensemble + ws_multiplier = 1 + batch_size = 1000 + epoch= 5
  log_file = '1643208424.0300257' #
  # suffix = 'epoch_wise'
  c41 = LinkPredictionCrawler(log_file)
  # c41 = LinkPredictionCrawler(log_file, suffix=suffix)
  # plot_path = str(base_path / f'plot/edges_weight_{log_file}.png')
  # c41.plot(savefig=True, is_ensemble=True)
  c41.plot(savefig=True, is_individual_model_in_ensemble=True)
  # c41.plot(savefig=True)
  # c41.plot(savefig=True, is_axis_1_as_x=True)
  # plot_tmp(c41.loss, c41.auc, c41.ap, c41.epoch , c41.plot_path, savefig=False)
  vc.append_value(c41)
  exit()


  # list_ = []
  # for i,j in zip(c10.get_return(),c13.get_return()):
  #   shortest_len = return_min_length_of_list_members([i,j])
  #   # shortest_len = min(i.shape[0], j.shape[0])
  #   i = i[:shortest_len]

    # list_.append(i-j)
    # asser


  # config = {}
  # auc_ylim_tuple = (-0.01, 0.01)
  # ap_ylim_tuple = (-0.01, 0.01)
  # config['ylim'] = [auc_ylim_tuple, ap_ylim_tuple]

  # # base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  # # plot_path = str(base_path / f'plot/comparing_{log_files[0]}_and_{log_files[1]}.png')
  # # # draw(*list_, config=config)
  # # # draw(*list_, plot_path=plot_path, savefig=True)
  # draw(*list_)


  ## args for draw_multiple function
  test_performance_on_the_same_period = True
  # test_performance_on_the_same_period = False

  # use_min_length_of_list_members = True
  use_min_length_of_list_members = False

  # savefig = True
  savefig = False

  # use_time_as_x_axis = True
  use_time_as_x_axis = False

  base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
  vs_log_files = ""
  for f in vc.log_files:
    vs_log_files += f.replace('.','-') + '_'

  if use_min_length_of_list_members:
    # plot_path = str(base_path / f'plot/{log_files[0]}_vs_{log_files[1]}_min_length=T.png')
    # plot_path = str(base_path / f'plot/{vs_log_files}min_length=T.png')
    is_min_length =  "T"
  else:
    is_min_length =  "F"
    # plot_path = str(base_path / f'plot/{log_files[0]}_vs_{log_files[1]}_min_length=F.png')
    # plot_path = str(base_path / f'plot/{vs_log_files}min_length=F.png')

  if use_time_as_x_axis:
    is_time_as_x = "T"
  else:
    is_time_as_x = "F"
  plot_path = str(base_path / f'plot/{vs_log_files}min_length={is_min_length}_use_time_as_x={is_time_as_x}_test_performance_on_the_same_period={test_performance_on_the_same_period}.png')

  # draw_multiple(losses, aucs, aps, epoches, plot_path=plot_path,savefig=True)
  # draw_multiple(losses, aucs, aps, epoches, plot_path=plot_path,savefig=False, return_min_length_of_list_members=False)
  draw_multiple(vc.header_dicts,vc.losses, vc.aucs, vc.aps, vc.epoches, vc.times, vc.end_ind_of_init_windows ,plot_path=plot_path,savefig=savefig, use_min_length_of_list_members=use_min_length_of_list_members, use_time_as_x_axis=use_time_as_x_axis, test_performance_on_the_same_period=test_performance_on_the_same_period)
