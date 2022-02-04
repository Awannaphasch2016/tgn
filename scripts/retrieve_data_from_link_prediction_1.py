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

# reddit_10000 + forward + ws_multiplier = 1 + batch_size = 1000 + epoch= 5 + keep 1 window
log_file = '1643947632.5513124' # 1 run
c54 = LinkPredictionCrawler_1(log_file)
c54.crawl_data()
c54.df['Name'] = log_file
vc.append_value(c54.df)


# reddit_10000 + forward + ws_multiplier = 1 + batch_size = 1000 + epoch= 5 + keep 2 window
log_file = '1643947679.1715186' # 1 run
c55 = LinkPredictionCrawler_1(log_file)
c55.crawl_data()
c55.df['Name'] = log_file
vc.append_value(c55.df)

# reddit_10000 + forward + ws_multiplier = 1 + batch_size = 1000 + epoch= 5 + keep 3 window
log_file = '1643947722.1774318' # 1 run
c56 = LinkPredictionCrawler_1(log_file)
c56.crawl_data()
c56.df['Name'] = log_file
vc.append_value(c56.df)

# reddit_10000 + forward + ws_multiplier = 1 + batch_size = 1000 + epoch= 5 + keep all window
log_file = '1643947455.2605557' # 1 run
c57 = LinkPredictionCrawler_1(log_file)
c57.crawl_data()
c57.df['Name'] = log_file
vc.append_value(c57.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 50 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --window_stride_multiplier 1
# # reddit_10000 + forward + ws_multiplier = 1 + batch_size = 1000 + epoch= 5 + keep all window
# log_file = '1643943757.5759256' # 1 run
# c58 = LinkPredictionCrawler_1(log_file)
# c58.crawl_data()
# c58.df['Name'] = log_file
# vc.append_value(c58.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 50 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 1 --window_stride_multiplier 1
# log_file = '1643944198.2235723' # 1 run
# c59 = LinkPredictionCrawler_1(log_file)
# c59.crawl_data()
# c59.df['Name'] = log_file
# vc.append_value(c59.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 50 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 2 --window_stride_multiplier 1
# log_file = '1643944633.5943987' # 1 run
# c60 = LinkPredictionCrawler_1(log_file)
# c60.crawl_data()
# c60.df['Name'] = log_file
# vc.append_value(c60.df)

# # python train_self_supervised.py -d reddit_10000 --use_memory --n_runs 1 --n_epoch 50 --bs 1000  --ws_framework forward --custom_prefix tmp --ws_multiplier 1 --keep_last_n_window_as_window_slides 3 --window_stride_multiplier 1
# log_file = '1643945049.3255963' # 1 run
# c61 = LinkPredictionCrawler_1(log_file)
# c61.crawl_data()
# c61.df['Name'] = log_file
# vc.append_value(c61.df)

# df = pd.concat( [c54.df, c55.df, c56.df, c57.df])
df = pd.concat(vc.dfs)
df = pd.melt(df, id_vars=['ws_idx', 'batch_idx', 'epoch_idx', 'Name'], value_vars=['AUC','Absolute Precision'], var_name='Metrics', value_name='Metrics Values')
sns.relplot(x='batch_idx', y='Metrics Values', data=df, col="Metrics", kind='line', hue='Name')
plt.yscale('log')
plt.show()
