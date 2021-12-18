from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

log_file = '1638569964.7565887' # 1000 bs on reddit with expert + 10 percent budget + 10 epochs + bug fix + log distibution count diregarding frequency of unique nodes
# log_file = '1638376034.93718'
# log_file = '1638211341.688131' # 1000 bs on reddit with expert + 10 percent budget + 10 epochs + bug fix ; true_train_label　is incorrectly logged.
# log_file = '1638194173.5304792' # 1000 bs on reddit with expert + 10 percent budget + 50 epochs
# log_file = '1638193757.1753845' # 1000 bs on reddit with expert + 90 percent budget + 10 epochs
# log_file = '1638147023.6811063' # 1000 bs on reddit with_expert + 10 percent budget + 10 epochs
# log_file = '1638146355.8566914' # 5000 bs on reddit_with_expert + 10 percent budget + 10 epochs
# log_file = '1638134540.2439132' # 1000 bs on reddit_with_expert_10000.
# log_file = '1637162600.6482246'
base_path = Path.cwd()
plot_path = base_path / "plot/"
log_path = str(base_path / f'log/{log_file}.log')

train_loss = []
val_auc = []
test_acc = []
cm = []
ws = []
time_list = []
true_test_label = []
true_train_label = []
pred_test_label = []
pred_train_label = []

pred_train_label_exclude_freq_count = []
pred_test_label_exclude_freq_count = []
true_train_label_exclude_freq_count = []
true_test_label_exclude_freq_count = []

total_number_of_labelled_unique_nodes = []
total_number_of_labelled_nodes = []

BATCH_SIZE = 5000
i = 1

previous_epoch = None
epochs = []

def get_label_dist(line):
  tmp = [[jj.strip(' ') for jj in ii.strip(',) ').split(',')] for ii in
     line.split('[')[1].split(']')[0].split('(')[1:]]

  return pprint_label_dist(tmp), convert_ind_value_tuple_to_np(tmp)

def convert_ind_value_tuple_to_np(tmp):
    max_label = 4
    tmp1 = np.zeros(4)
    assert tmp1.shape[0] == max_label
    for i,j in tmp:
        tmp1[int(i)] = int(j)
    return tmp1

def pprint_label_dist(tmp):
    tmp1 = []
    for i in tmp:
        s = '->'.join(i)
        tmp1.append(s)
    return ','.join(tmp1)


def plot_labels_dist(labels_np, total_number_of_labelled_unique_nodes, title, path_to_dir, log_file, savefig=False):
    plt.plot(labels_np, label=[0, 1, 2, 3])
    plt.plot(labels_np.sum(axis=1), label='total labels')
    plt.plot(total_number_of_labelled_unique_nodes, label=['total labelled unique nodes.'])
    plt.xlabel('step')
    plt.title(title)
    plt.legend()
    if savefig:
        plt.savefig(path_to_dir/ f"{log_file}_{title}.png")
    plt.show()

with open(log_path, 'r') as f:
  for i, line in enumerate(f.readlines()): # 9341 lines in total
    is_epoch_exist = True if '--epoch' in line else False
    is_train_loss_exist = True if 'loss' in line else False
    is_val_auc_exist = True if 'val auc' in line else False
    is_cm_matrix_exist = True if '[' in line and 'labels' not in line else False
    is_train_label_distribution_exist = True if 'train labels epoch distribution' in line else False
    is_train_label_distribution_exist_exclude_freq_count = True if 'train labels epoch distribution (disregard frequency of unique node)' in line else False
    is_test_label_distribution_exist = True if 'test labels distribution' in line else False
    is_test_label_distribution_exist_exclude_freq_count = True if 'test labels epoch distribution (disregard frequency of unique node)' in line else False
    is_predicted = True if 'predicted' in line else False
    is_ws_exist = True if '-ws' in line else False
    is_time_exist = True if 'took' in line else False
    is_total_number_of_labelled_unique_nodes = True if 'total number of labelled uniqued nodes' in line else False
    is_total_number_of_labelled_nodes = True if 'total labels batch epoch distribution' in line else False

    # 2021-11-21 18:49:41,876 - root - INFO - epoch: 9 took 16.97s
    if is_time_exist:
      time_val = float(line.split('took')[1].split('s')[0].strip(' \n'))
      time_list.append(time_val)

    if is_ws_exist:
       ws_val = int(line.split('-ws')[-1].strip('= \n'))
       ws.append(ws_val)

    if is_epoch_exist:
       epoch = int(line.split('epoch')[1].split(':')[0].strip(' \n='))
       epochs.append(epoch)

    if is_train_loss_exist:
       train_loss_val =  float(line.split("Epoch mean loss")[1].split(':')[1].split(',')[0].strip(' '))
       train_loss.append(train_loss_val)


       # 2021-11-21 16:59:20,137 - root - INFO - val auc: (None, 0.61, array([[ 0,  0, 39,  0],

    if is_val_auc_exist:
       val_auc_val =  float(line.split("val auc")[1].split(':')[1].split(',')[1].strip(' '))
       cm_val =  [int(i.strip(' ')) for i in line.split("val auc")[1].split(':')[1].split('[')[-1].split(']')[-2].strip(' ').split(',') ]
       val_auc.append(val_auc_val)
       cm.append(cm_val)

    if is_cm_matrix_exist and not is_val_auc_exist:
      cm_val = [int(i.strip(' ')) for i in line.strip('[]),\n ').strip(' ').split(',')]
      cm.append(cm_val)

    if is_predicted and is_train_label_distribution_exist and not is_train_label_distribution_exist_exclude_freq_count:
      pred_train_label.append(get_label_dist(line))
    if is_predicted and is_test_label_distribution_exist and not is_test_label_distribution_exist_exclude_freq_count:
      pred_test_label.append(get_label_dist(line))
    if not is_predicted and is_train_label_distribution_exist and not is_train_label_distribution_exist_exclude_freq_count:
      true_train_label.append(get_label_dist(line))
    if not is_predicted and is_test_label_distribution_exist and not is_test_label_distribution_exist_exclude_freq_count:
      true_test_label.append(get_label_dist(line))

    if is_predicted and is_train_label_distribution_exist_exclude_freq_count:
        pred_train_label_exclude_freq_count.append(get_label_dist(line))
    if is_predicted and is_test_label_distribution_exist_exclude_freq_count:
        pred_test_label_exclude_freq_count.append(get_label_dist(line))
    if not is_predicted and is_train_label_distribution_exist_exclude_freq_count:
        true_train_label_exclude_freq_count.append(get_label_dist(line))
    if not is_predicted and is_test_label_distribution_exist_exclude_freq_count:
        true_test_label_exclude_freq_count.append(get_label_dist(line))

    if is_total_number_of_labelled_unique_nodes:
        total_number_of_labelled_unique_nodes_val = int(line.split('=')[1].strip('\n '))
        total_number_of_labelled_unique_nodes.append(total_number_of_labelled_unique_nodes_val)

    if is_total_number_of_labelled_nodes:
        total_number_of_labelled_nodes.append(get_label_dist(line))


n_epoch = max(epochs) - min(epochs) + 1
n_ws = max(ws)

assert max(ws) == len(ws) - 1 # to test only the first run.


train_loss = np.array(train_loss)
n_ws_with_all_epoch = train_loss.shape[0] - (train_loss.shape[0] % n_epoch)
total_number_of_labelled_unique_nodes = total_number_of_labelled_unique_nodes[:n_ws_with_all_epoch]
train_loss = train_loss[:n_ws_with_all_epoch]
time_list = time_list[:n_ws_with_all_epoch]
time_list = np.array(time_list).reshape(-1, n_epoch)

val_auc = np.array(val_auc)
val_auc = val_auc[:n_ws_with_all_epoch]
title = 'loss and accuracy'
plt.plot(train_loss, label = 'loss')
plt.plot(val_auc, label = 'accuracy')
plt.xlabel('step')
plt.title(title)
plt.legend()
plt.show()
plt.savefig(plot_path/ f"{log_file}_{title}.png")

# total_number_of_labelled_nodes = total_number_of_labelled_nodes[:n_ws_with_all_epoch]
# total_number_of_labelled_nodes_np = np.array([j for i,j in total_number_of_labelled_nodes])
# title = 'distribution of total true train labels on each step'
# plot_labels_dist(total_number_of_labelled_nodes_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)
#
true_train_label = true_train_label[:n_ws_with_all_epoch]
true_train_label_np = np.array([j for i,j in true_train_label])
# title = 'distribution of true train labels on each step'
# plot_labels_dist(true_train_label_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)
#
pred_train_label = pred_train_label[:n_ws_with_all_epoch]
pred_train_label_np = np.array([j for i,j in pred_train_label])
# title = 'distribution of predicted train labels on each step'
# plot_labels_dist(pred_train_label_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)
# exit()

true_test_label = true_test_label[:n_ws_with_all_epoch]
true_test_label_np = np.array([j for i,j in true_test_label])
# title = 'distribution of true test labels on each step'
# plot_labels_dist(true_test_label_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)
#
pred_test_label = pred_test_label[:n_ws_with_all_epoch]
pred_test_label_np = np.array([j for i,j in pred_test_label])
# title = 'distribution of predicted test labels on each step'
# plot_labels_dist(pred_test_label_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)





true_train_label_exclude_freq_count = true_train_label_exclude_freq_count[:n_ws_with_all_epoch]
train_label_exclude_freq_count_np = np.array([j for i,j in true_train_label_exclude_freq_count])
# title = 'distribution of true train labels on each step'
# plot_labels_dist(train_label_exclude_freq_count_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)
#
pred_train_label_exclude_freq_count = pred_train_label_exclude_freq_count[:n_ws_with_all_epoch]
pred_train_label_exclude_freq_count_np = np.array([j for i,j in pred_train_label_exclude_freq_count])
# title = 'distribution of predicted train labels on each step'
# plot_labels_dist(pred_train_label_exclude_freq_count_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)
#
true_test_label_exclude_freq_count = true_test_label_exclude_freq_count[:n_ws_with_all_epoch]
test_label_exclude_freq_count_np = np.array([j for i,j in true_test_label_exclude_freq_count])
# title = 'distribution of true test labels on each step'
# plot_labels_dist(test_label_exclude_freq_count_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)

pred_test_label_exclude_freq_count = pred_test_label_exclude_freq_count[:n_ws_with_all_epoch]
pred_test_label_exclude_freq_count_np = np.array([j for i,j in pred_test_label_exclude_freq_count])
# title = 'distribution of predicted test labels on each step'
# plot_labels_dist(pred_test_label_exclude_freq_count_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)

def plot_labels_dist_with_and_without_freq_count_np(labels_np, labels_np_exclude_freq_count_np, total_number_of_labelled_unique_nodes, title, path_to_dir, log_file, savefig=False):
    plt.plot(labels_np_exclude_freq_count_np, label= ["0 w/o freq","1 w/o freq","2 w/o freq","3 w/o freq"])
    plt.plot(labels_np, label=[0, 1, 2, 3])
    # plt.plot(labels_np.sum(axis=1), label='total labels')
    plt.plot(total_number_of_labelled_unique_nodes, label=['total labelled unique nodes.'])
    plt.xlabel('step')
    plt.title(title)
    plt.legend()
    if savefig:
        plt.savefig(path_to_dir/ f"{log_file}_{title}.png")
    plt.show()


# true_train_label_exclude_freq_count = true_train_label_exclude_freq_count[:n_ws_with_all_epoch]
# train_label_exclude_freq_count_np = np.array([j for i,j in true_train_label_exclude_freq_count])
title = 'distribution of true train labels with and without frequency count on each step'
# plot_labels_dist_with_and_without_freq_count_np(true_train_label_np[:100],train_label_exclude_freq_count_np[:100], total_number_of_labelled_unique_nodes[:100], title, plot_path, log_file, savefig=True)
plot_labels_dist_with_and_without_freq_count_np(true_test_label_np,train_label_exclude_freq_count_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)

# pred_train_label_exclude_freq_count = pred_train_label_exclude_freq_count[:n_ws_with_all_epoch]
# pred_train_label_exclude_freq_count_np = np.array([j for i,j in pred_train_label_exclude_freq_count])
title = 'distribution of predicted train labels with and without frequency count on each step'
plot_labels_dist_with_and_without_freq_count_np(pred_train_label_np,pred_train_label_exclude_freq_count_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)

# true_test_label_exclude_freq_count = true_test_label_exclude_freq_count[:n_ws_with_all_epoch]
# test_label_exclude_freq_count_np = np.array([j for i,j in true_test_label_exclude_freq_count])
title = 'distribution of true test labels with and without frequency count on each step'
plot_labels_dist_with_and_without_freq_count_np(true_test_label_np,test_label_exclude_freq_count_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)

# pred_test_label_exclude_freq_count = pred_test_label_exclude_freq_count[:n_ws_with_all_epoch]
# pred_test_label_exclude_freq_count_np = np.array([j for i,j in pred_test_label_exclude_freq_count])
title = 'distribution of predicted test labels with and without frequency count on each step'
plot_labels_dist_with_and_without_freq_count_np(pred_test_label_np,pred_test_label_exclude_freq_count_np, total_number_of_labelled_unique_nodes, title, plot_path, log_file, savefig=True)