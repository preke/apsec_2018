
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import preprocessing


pos = pd.read_csv('pos.csv')
neg = pd.read_csv('neg.csv')
sim = pd.read_csv('456_sim.csv')
sim = sim[['sim', 'label', 'pair_id']]


# pos['label'] = '1'
# neg['label'] = '0'
neg['idx'] = neg.index
neg['pair_id'] = neg.idx
pos['idx'] = pos.index
pos['pair_id'] = pos.idx.apply(lambda x: x + len(neg))


total = pd.concat([neg, pos])
# print(total['pair_id'].describe())
# print(sim['pair_id'].describe())
# print(sim.shape)
# print(total.head())

sim = pd.merge(sim, total, on='pair_id', how='left')
sim = sim[['sim', 'label', 'pair_id', 'same_tw', 'same_prio', 'same_comp', 'Title_1', 'Title_2', 'Issue_id_1', 'Issue_id_2']]
# print(sim.shape)

# print(sim.head())

train_set = sim.head(int(0.8*len(sim)))[['sim', 'same_comp', 'same_prio', 'same_tw']].fillna(0)
train_label = sim.head(int(0.8*len(sim)))['label'].apply(lambda x: int(x))

test_set = sim.tail(int(0.2*len(sim)))[['sim', 'same_comp', 'same_prio', 'same_tw']].fillna(0)
test_label = sim.tail(int(0.2*len(sim)))['label'].apply(lambda x: int(x))
# test_set = sim.head(int(0.2*len(sim)))[['sim', 'same_comp', 'same_prio', 'same_tw']].fillna(0)
# test_label = sim.head(int(0.2*len(sim)))['label'].apply(lambda x: int(x))

validation_label = test_label
model = LogisticRegression()
model.fit(train_set, train_label)
res = model.predict(test_set)

# recall = float(sum(list(res)))/sum(list(validation_label))
precision = 0.0

cnt = 0
for i in range(len(res)):
    if res[i] == list(validation_label)[i]:
        cnt += 1
        if res[i] == 1:
            precision += 1
t = precision
precision = t / float(float(sum(list(res))))
recall = t /sum(list(validation_label))
f1 = 2*precision*recall / (precision + recall)

print('acc:{:.6f}'.format(float(cnt)/len(res)))
print('f1:{:.6f}'.format(f1))

cnn_cnt = 0
test_sim = list(test_set['sim'])
for i in range(len(test_sim)):
    if (test_sim[i] >= 0.5) & (list(validation_label)[i] == 1):
        cnn_cnt += 1
    elif (test_sim[i] < 0.5) & (list(validation_label)[i] == 0):
        cnn_cnt += 1

print('acc:{:.6f}'.format(float(cnn_cnt)/len(res)))

# cnn_test = sim.tail(int(0.2*len(sim)))[['Title_1', 'Title_2', 'label']]
# print('===')
# print(len(train_set)/8*7)
# print(len(train_set)/8*1)
# print(len(test_set))
# print(len(sim))

cnn_train = sim.head(int(0.8*len(sim)))[['Title_1', 'Title_2', 'label']]
cnn_train.to_csv('cnn_train_new.csv')

cnn_test = sim.tail(int(0.2*len(sim)))[['Title_1', 'Title_2', 'label']]
cnn_test.to_csv('cnn_test_new.csv')


