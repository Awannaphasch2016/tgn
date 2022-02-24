#!/usr/bin/env python3
from utils.crawler import Crawler
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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

# # reddit_10000 + forward + ws_multiplier = 1 + batch_size = 1000 + epoch= 5 + keep 1 window
# log_file = '1643947632.5513124' # 1 run
# c54 = LinkPredictionCrawler_1(log_file)
# c54.crawl_data()
# c54.df['Name'] = log_file
# vc.append_value(c54.df)


# # reddit_10000 + forward + ws_multiplier = 1 + batch_size = 1000 + epoch= 5 + keep 2 window
# log_file = '1643947679.1715186' # 1 run
# c55 = LinkPredictionCrawler_1(log_file)
# c55.crawl_data()
# c55.df['Name'] = log_file
# vc.append_value(c55.df)

# # reddit_10000 + forward + ws_multiplier = 1 + batch_size = 1000 + epoch= 5 + keep 3 window
# log_file = '1643947722.1774318' # 1 run
# c56 = LinkPredictionCrawler_1(log_file)
# c56.crawl_data()
# c56.df['Name'] = log_file
# vc.append_value(c56.df)

# # reddit_10000 + forward + ws_multiplier = 1 + batch_size = 1000 + epoch= 5 + keep all window
# log_file = '1643947455.2605557' # 1 run
# c57 = LinkPredictionCrawler_1(log_file)
# c57.crawl_data()
# c57.df['Name'] = log_file
# vc.append_value(c57.df)

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 50 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --window_stride_multiplier 1
# reddit_10000 + forward + ws_multiplier = 1 + batch_size = 1000 + epoch= 5 + keep all window
# log_file = '1643943757.5759256' # 1 run
log_file = '1643948503.9221833'
c58 = LinkPredictionCrawler_1(log_file)
c58.crawl_data()
c58.df['Name'] = log_file
vc.append_value(c58.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 50 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 1 --window_stride_multiplier 1
# # log_file = '1643944198.2235723' # 1 run
# log_file = '1643950077.9129927'
# c59 = LinkPredictionCrawler_1(log_file)
# c59.crawl_data()
# c59.df['Name'] = log_file
# vc.append_value(c59.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 50 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 2 --window_stride_multiplier 1
# # log_file = '1643944633.5943987' # 1 run
# log_file = '1643950472.025392'
# c60 = LinkPredictionCrawler_1(log_file)
# c60.crawl_data()
# c60.df['Name'] = log_file
# vc.append_value(c60.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 50 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 3 --window_stride_multiplier 1
# # log_file = '1643945049.3255963' # 1 run
# log_file = '1643950910.5746765'
# c61 = LinkPredictionCrawler_1(log_file)
# c61.crawl_data()
# c61.df['Name'] = log_file
# vc.append_value(c61.df)


# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --window_stride_multiplier 1 --use_time_decay_multiplier
log_file = '1644007650.528038'
c62 = LinkPredictionCrawler_1(log_file)
c62.crawl_data()
c62.df['Name'] = log_file
vc.append_value(c62.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 1 --window_stride_multiplier 1 --use_time_decay_multiplier
# log_file = '1644007827.2711818'
# c63 = LinkPredictionCrawler_1(log_file)
# c63.crawl_data()
# c63.df['Name'] = log_file
# vc.append_value(c63.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 2 --window_stride_multiplier 1 --use_time_decay_multiplier
# log_file = '1644007895.1632655'
# c64 = LinkPredictionCrawler_1(log_file)
# c64.crawl_data()


# vc.append_value(c64.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 3 --window_stride_multiplier 1 --use_time_decay_multiplier
# log_file = '1644007951.7181628'
# c65 = LinkPredictionCrawler_1(log_file)
# c65.crawl_data()
# c65.df['Name'] = log_file
# vc.append_value(c65.df)

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --window_stride_multiplier 1 --use_ef_iwf_weight
log_file = '1644008759.0023758'
c65 = LinkPredictionCrawler_1(log_file)
c65.crawl_data()
c65.df['Name'] = log_file
vc.append_value(c65.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 1 --window_stride_multiplier 1 --use_ef_iwf_weight
# log_file = '1644008947.1709862'
# c66 = LinkPredictionCrawler_1(log_file)
# c66.crawl_data()
# c66.df['Name'] = log_file
# vc.append_value(c66.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 2 --window_stride_multiplier 1 --use_ef_iwf_weight
# log_file = '1644008995.451304'
# c67 = LinkPredictionCrawler_1(log_file)

# c67.df['Name'] = log_file
# vc.append_value(c67.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 3 --window_stride_multiplier 1 --use_ef_iwf_weight
# log_file = '1644009040.641835'
# c68 = LinkPredictionCrawler_1(log_file)
# c68.crawl_data()
# c68.df['Name'] = log_file
# vc.append_value(c68.df)


# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --window_stride_multiplier 1 --use_nf_iwf_weight
log_file = '1644010709.4537165'
c69 = LinkPredictionCrawler_1(log_file)
c69.crawl_data()
c69.df['Name'] = log_file
vc.append_value(c69.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 1 --window_stride_multiplier 1 --use_nf_iwf_weight
# log_file = '1644011812.3486102'
# c70 = LinkPredictionCrawler_1(log_file)
# c70.crawl_data()
# c70.df['Name'] = log_file
# vc.append_value(c70.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 2 --window_stride_multiplier 1 --use_nf_iwf_weight
# log_file = '1644011861.1766088'
# c71 = LinkPredictionCrawler_1(log_file)
# c71.crawl_data()
# c71.df['Name'] = log_file
# vc.append_value(c71.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 3 --window_stride_multiplier 1 --use_nf_iwf_weight
# log_file = '1644011912.423623'
# c72 = LinkPredictionCrawler_1(log_file)
# c72.crawl_data()
# c72.df['Name'] = log_file
# vc.append_value(c72.df)

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --window_stride_multiplier 1 --use_nf_weight
# here is use data.sources to compute pos_edges_weight
# log_file = '1644242216.1149487'
log_file = '1644242594.7757893'
c73 = LinkPredictionCrawler_1(log_file)
c73.crawl_data()
c73.df['Name'] = log_file
vc.append_value(c73.df)

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --window_stride_multiplier 1 --use_nf_weight
# here is use data.destination to compute pos_edges_weight
log_file = '1644242782.074008'
c74 = LinkPredictionCrawler_1(log_file)
c74.crawl_data()
c74.df['Name'] = log_file
vc.append_value(c74.df)

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --window_stride_multiplier 1 --use_ef_weight
log_file = '1644243005.7744277'
c75 = LinkPredictionCrawler_1(log_file)
c75.crawl_data()
c75.df['Name'] = log_file
vc.append_value(c75.df)

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --window_stride_multiplier 1 --use_ef_weight --use_time_decay
log_file = '1644243253.977707'
c76 = LinkPredictionCrawler_1(log_file)
c76.crawl_data()
c76.df['Name'] = log_file
vc.append_value(c76.df)

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --window_stride_multiplier 1 --use_ef_iwf_weight --use_time_decay
log_file = '1644244151.472011'
c77 = LinkPredictionCrawler_1(log_file)
c77.crawl_data()
c77.df['Name'] = log_file
vc.append_value(c77.df)

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --window_stride_multiplier 1 --use_nf_iwf_weight --use_time_decay
log_file = '1644244814.7723272'
c78 = LinkPredictionCrawler_1(log_file)
c78.crawl_data()
c78.df['Name'] = log_file
vc.append_value(c78.df)

# python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 5 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --window_stride_multiplier 1 --use_nf_weight --use_time_decay
log_file = '1644245068.9889233'
c79 = LinkPredictionCrawler_1(log_file)
c79.crawl_data()
c79.df['Name'] = log_file
vc.append_value(c79.df)


# df = pd.concat( [c54.df, c55.df, c56.df, c57.df])
df = pd.concat(vc.dfs)
df = pd.melt(df, id_vars=['ws_idx', 'batch_idx', 'epoch_idx', 'Name'], value_vars=['AUC','Absolute Precision'], var_name='Metrics', value_name='Metrics Values')
sns.relplot(x='batch_idx', y='Metrics Values', data=df, col="Metrics", kind='line', hue='Name')
plt.yscale('log')
plt.show()
