# This file is created during the time my laptop charger is broken and I am using labs computers to write code.

import pandas as pd
# y = pd.read_csv('data/reddit_with_expert_labels.csv')
y = []
y.append("user_id,item_id,timestamp,state_label,comma_separated_list_of_features")
x = pd.read_csv('data/reddit_no_header.csv')
x['0'] = x['0'].astype(int)
x['1'] = x['1'].astype(int)
x['3'] = x['3'].astype(int)

with open('data/reddit_with_expert_labels.csv', 'w') as f:
    f.write(y[0] + '\n')
    for d in x.values:
        # d = i.split(',')
        u_id = int(d[0])
        i_id = int(d[1])
        state_label = int(d[3])
        line = str(u_id)+','+str(i_id)+','+str(d[2])+','+str(state_label)+','+','.join([str(g) for g in d[4:]])
        # print(line)
        f.write(line + '\n')