#!/usr/bin/env python3
from utils.crawler import Crawler
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ValueCollection():
  def __init__(self):
    self.dfs = []

  def append_value(self, df):

    self.dfs.append(df)

class LinkPredictionCrawler_1(Crawler):
  def __init__(self, log_file, suffix=''):
    base_path = Path('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/')
    self.log_path = str(base_path / f'log/nodes_and_edges_weight/{log_file}/performance_observer.pickle')
    # self.crawl_data_v2(nodes_and_edges_weight_log_path)

  def crawl_data(self):
    self.df = pd.read_pickle(self.log_path)

def draw(df, x, y, hue):
    sns.boxplot(x=x, y=y, data=df, hue="Metrics")
    plt.yscale('log')
    plt.show()

# draw_multiple()

vc =  ValueCollection()

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 3 --window_stride_multiplier 1 --use_nf_iwf_weight
# log_file = '1644012732.606389'
# c73 = LinkPredictionCrawler_1(log_file)
# c73.crawl_data()
# c73.df['Name'] = log_file
# vc.append_value(c73.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 10000 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 1 --window_stride_multiplier 1
# log_file = '1644034660.7790797'
# c74 = LinkPredictionCrawler_1(log_file)
# c74.crawl_data()
# c74.df['Name'] = log_file
# vc.append_value(c74.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 50 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 1 --window_stride_multiplier 1
# # log_file = '1643944198.2235723' # 1 run
# log_file = '1643950077.9129927'
# c59 = LinkPredictionCrawler_1(log_file)
# c59.crawl_data()
# c59.df['Name'] = log_file
# vc.append_value(c59.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 500 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 1 --window_stride_multiplier 1
# log_file = '1644083686.5323043'
# c74 = LinkPredictionCrawler_1(log_file)
# c74.crawl_data()
# c74.df['Name'] = log_file
# vc.append_value(c74.df)

# # (list "-d" "reddit_10000" "--use_memory" "--n_runs" "1" "--n_epoch" "2" "--bs" "1000" "--ws_multiplier" "1" "--custom_prefix" "tmp" "--ws_framework" "ensemble")
# log_file = '1644083686.5323043'
# c75 = LinkPredictionCrawler_1(log_file)
# c75.crawl_data()
# c75.df['Name'] = log_file
# vc.append_value(c75.df)

# (list "-d" "reddit_10000" "--use_memory" "--n_runs" "1" "--n_epoch" "5" "--bs" "200" "--ws_multiplier" "1" "--custom_prefix" "tmp" "--ws_framework" "ensemble" "--disable_cuda")
# log_file = '1645467382.394281'
# log_file = '1645468621.269857'
log_file = '1645468840.7249002'
c76 = LinkPredictionCrawler_1(log_file)
c76.crawl_data()
c76.df['Name'] = log_file
vc.append_value(c76.df)

# (list "-d" "reddit_10000" "--use_memory" "--n_runs" "1" "--n_epoch" "5" "--bs" "200" "--ws_multiplier" "1" "--custom_prefix" "tmp" "--ws_framework" "ensemble" "--disable_cuda" "--init_n_instances_as_multiple_of_ws" "6" "--fix_begin_data_ind_of_models_in_ensemble")
# log_file = '1645559677.4656665'
log_file = '1645560100.8301754'
c77 = LinkPredictionCrawler_1(log_file)
c77.crawl_data()
c77.df['Name'] = log_file
vc.append_value(c77.df)

df = pd.concat(vc.dfs)
df.reset_index(inplace=True)
# df = pd.melt(df, id_vars=['index', 'ws_idx', 'batch_idx', 'epoch_idx', 'Name', 'Mean Loss'], value_vars=['AUC','Absolute Precision'], var_name='Metrics', value_name='Metrics Values')
df = pd.melt(df, id_vars=['index', 'ws_idx', 'batch_idx', 'ensemble_idx','epoch_idx', 'Name', 'Mean Loss'], value_vars=['AUC','Absolute Precision'], var_name='Metrics', value_name='Metrics Values')


# # sns.relplot(x='batch_idx', y='Metrics Values', data=df, col="Metrics", kind='line', hue='Name')
# # sns.relplot(x='batch_idx', y='Mean Loss', data=df, col="Metrics", kind='line', hue='Name')
# sns.relplot(x='epoch_idx', y='Mean Loss', data=df[df['batch_idx'] == 0], col="Metrics", kind='line', hue='Name')
# plt.yscale('log')
# plt.show()


# sns.relplot(x='epoch_idx', y='Metrics Values', data=df[ np.logical_and(df['batch_idx'] == 0, df['epoch_idx'] < 500) ], col="Metrics",kind='line')
# sns.relplot(x='epoch_idx', y='Metrics Values', data=df[df['epoch_idx'] < 1000], hue='batch_idx', col="Metrics",kind='line')
# sns.relplot(x='epoch_idx', y='Metrics Values', data=df[df['epoch_idx'] < 1000], hue='batch_idx', col="Metrics",kind='line')
# sns.relplot(x='epoch_idx', y='Mean Loss', data=df[df['epoch_idx'] < 1000], hue='batch_idx', col="Metrics",kind='line')
# sns.relplot(x='epoch_idx', y='Mean Loss', data=df[df['epoch_idx'] < 1000], hue='batch_idx',kind='line')
# sns.relplot(x='ws_idx', y='Mean Loss', data=df, hue='ensemble_idx',kind='line', col='Metrics')
sns.relplot(x='ws_idx', y='Metrics Values', data=df, col="Metrics", kind='line', hue='Name')
# sns.relplot(x='ws_idx', y='Metrics Values', data=df, col="Metrics", kind='line')
plt.yscale('log') # plt.savefig('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/img/overfitting_model_loss.png')
plt.show()
