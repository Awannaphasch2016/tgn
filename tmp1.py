#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ast import literal_eval

path_list = []
# n_ws = range(8)
dict_ = {
    "weight": None,
    "batch_idx": None,
    }
np_list = []
# n_ws = range(9)
n_ws = range(3)
# n_epoch = range(5)
n_epoch = range(1)
ws_multiplier = 1
weight_list = []
ws_list = []
epoch_list = []
batch_idx_list = []
# log_file = '1643846302.98939' # use_time_decay_multiplier + convert_seconds_to_days
# log_file = '1643847864.361433' # use_time_decay_multiplier + convert_seconds_to_hour
# log_file = '1643848253.9848332' # use_time_decay_multiplier + convert_seconds_to_mins
# log_file = '1643849250.0097542'# use_time_decay_multiplier + convert_seconds_to_hours + ef_iwf_weight
# log_file = '1643924628.2793286'
log_file = '1643926404.516756' # use_time_decay_multiplier + convert_seconds_to_hours + ef_iwf_weight + keep 2 windows

# for i in n_ws:
#     for j in n_epoch:
#         for k in range((i + 1) * ws_multiplier):
#             # path =f"/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/log/nodes_and_edges_weight/{log_file}/pos_edges_weight_run=0_ws={i}_epoch={j}_batch_{k}.csv"
#             path =f"/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/log/nodes_and_edges_weight/{log_file}/pos_edges_weight_run=0_ws={i}_epoch={j}_batch={k}.csv"
#             path_list.append(path)

#             df = pd.read_csv(path)

#             weight_list.extend(df.mean(axis=1).to_numpy().tolist())
#             ws_list.extend([ i ])
#             epoch_list.extend([ j ])
#             batch_idx_list.extend([ k ])

#             # dict_['weight'] = df.mean(axis=1).to_numpy()
#             # dict_['batch_idx'] = i
#             # dict_['epoch_idx'] = j
#             # dict_['idx'] = np.arange(df.shape[0])

#             # df = pd.DataFrame.from_dict(dict_)
#             # sns.boxplot(x="batch_idx", y="weight", data=df)
#             # sns.boxplot(x="batch_idx", y="weight", data=df)
#             # sns.boxplot(x="idx", y="weight", data=df)
#             # plt.show()

#             # np_list.append(df.to_numpy().tolist())

path =f"/mnt/c/Users/terng/OneDrive/Documents/Working/tgn/log/nodes_and_edges_weight/{log_file}/observer.pickle"
df = pd.read_pickle(path)
# val = np.array([weight_list, ws_list, epoch_list, batch_idx_list]).T
# df = pd.DataFrame(val, columns=['weight', 'ws_idx', 'epoch_idx', 'batch_idx'])
# val = np.array([ i for i in df['pos_edges_weight'] ]).mean(axis=1)


tmp = []
# TODO: optimize with broadcasting
for ind, i in enumerate(df['pos_edges_weight']):
    for j in i:
        tmp1 = df[["ws_idx", "epoch_idx", "batch_idx"]].to_numpy().tolist()[ind]
        tmp1.extend([ j ])
        tmp.append(tmp1)


df = pd.DataFrame(tmp, columns=df.columns)
sns.boxplot(x="ws_idx", y="pos_edges_weight", data=df)
plt.yscale('log')
plt.show()

sns.boxplot(x="epoch_idx", y="pos_edges_weight", data=df)
plt.yscale('log')
plt.show()

sns.boxplot(x="batch_idx", y="pos_edges_weight", data=df)
plt.yscale('log')
plt.show()
