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

vc =  ValueCollection()


# (list "-d" "reddit_10000" "--use_memory" "--n_runs" "1" "--n_epoch" "2" "--bs" "1000" "--ws_multiplier" "1" "--custom_prefix" "tmp" "--ws_framework" "ensemble"
# log_file = '1644512354.6743796'
log_file = '1644514533.8118901'
c75 = LinkPredictionCrawler_1(log_file)
c75.crawl_data()
c75.df['Name'] = log_file
vc.append_value(c75.df)

# orignal
log_file = '1642416734.1726859'
c76 = LinkPredictionCrawler_1(log_file)
c76.crawl_data()
# c76.df = c75.sdomething
# c76.df = c76.df
c76.df['Name'] = log_file
c76.df.rename(columns={'ws_idx':'model_idx'})
vc.append_value(c76.df)
df = pd.concat(vc.dfs)
df.reset_index(inplace=True)
df = pd.melt(df, id_vars=['index', 'ensemble_idx', 'batch_idx', 'epoch_idx', 'Name', 'Mean Loss'], value_vars=['AUC','Absolute Precision'], var_name='Metrics', value_name='Metrics Values')

# sns.relplot(x='epoch_idx', y='Metrics Values', data=df, hue='ensemble_idx', col="Metrics",kind='line')
sns.catplot(x='epoch_idx', y='Metrics Values', data=df, hue='ensemble_idx', col="Metrics",kind='bar')
plt.yscale('log')
# plt.savefig('/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/img/overfitting_model_loss.png')
plt.show()
