# coding = utf-8
import os
import pandas as pd
import numpy as np
# from preprocess import Preprocess

import re
import random
import tarfile
import urllib
from torchtext import data
from datetime import datetime
import pickle
from gensim.models import Word2Vec
import jieba
import traceback



data_path = '../spark/spark.csv'
neg_path = '../../lr_pair_feature_data/spark/neg.csv'
pos_path = '../../lr_pair_feature_data/spark/pos.csv'

def times_window(t1, t2):
    t1 = pd.to_datetime(t1)
    t2 = pd.to_datetime(t2)
    delta = t2 - t1 if t2 > t1 else t1 - t2
    if delta.days < 90:
        return 1
    else:
        return 0

def train_word2vec_model(df):
    '''
    basic w2v model trained by sentences
    '''
    corpus = []
    for i, r in df.iterrows():
        try:
            corpus.append(jieba.lcut(r['Title']))
            # print jieba.lcut(r['ques1'])
            corpus.append(jieba.lcut(r['Description']))
        except:
            pass
            # print('Exception: ', r['ques1']
    word2vec_model = Word2Vec(corpus, size=300, window=3, min_count=1, sg=0, iter=100)
    return word2vec_model

def load_data(data_path):
    #
    # df = pd.read_csv(open(data_path, 'rU'))
    df = pd.read_csv(data_path, encoding = 'gb18030')
    
    df['Duplicate_null'] = df['Duplicated_issue'].apply(lambda x : pd.isnull(x))
    
    # prep = Preprocess()
    # df['Desc_list'] = df['Title'].apply(lambda x : prep.stem_and_stop_removal(x))
    
    # Positive samples
    df_data = df[df['Duplicate_null'] == False]


    df_field = df_data[['Issue_id', 'Title', 'Duplicated_issue', 'Resolution']]
    df_field['dup_list'] = df_field['Duplicated_issue'].apply(lambda x: x.split(';'))
    Dup_list = []
    for i,r in df_field.iterrows():
        for dup in r['dup_list']:
            # print(dup)
            if int(r['Issue_id'].split('-')[1]) < int(dup.split('-')[1]):
                if dup.startswith('MAP'):
                    Dup_list.append([r['Issue_id'], dup, r['Resolution']])
    df_pairs_pos = pd.DataFrame(Dup_list, columns = ['Issue_id_1', 'Issue_id_2', 'Resolution'])

    # Negative samples
    neg_dup_list = []
    cnt = 0
    for i,r in df.iterrows():
        if r['Duplicate_null'] == True:
            j = 1
            try:
                while not df.ix[i+j]['Issue_id'].startswith('MAP'):
                    j += 1
                neg_dup_list.append([r['Issue_id'], df.ix[i+j]['Issue_id'], r['Resolution']])
                cnt += 1
            except:
                print(traceback.print_exc()) 
            
        if cnt > len(Dup_list):
            break

    df_pairs_neg = pd.DataFrame(neg_dup_list, columns = ['Issue_id_1', 'Issue_id_2', 'Resolution'])

    df_pairs_neg['Title_1'] = df_pairs_neg['Issue_id_1'].apply(lambda x: list(df[df['Issue_id'] == x]['Title'])[0])
    df_pairs_neg['Title_2'] = df_pairs_neg['Issue_id_2'].apply(lambda x: list(df[df['Issue_id'] == x]['Title'])[0])
    
    df_pairs_pos['Title_1'] = df_pairs_pos['Issue_id_1'].apply(lambda x: list(df[df['Issue_id'] == x]['Title'])[0])
    df_pairs_pos['Title_2'] = df_pairs_pos['Issue_id_2'].apply(lambda x: list(df[df['Issue_id'] == x]['Title'])[0] if len(list(df[df['Issue_id'] == x]['Title'])) > 0 else '')
    # df_pairs_pos['Title_2'] = df_pairs_pos['Issue_id_2'].apply(lambda x: list(df[df['Issue_id'] == x]['Title'])[0])
    
    df_pairs_pos['same_comp'] = df_pairs_pos.apply(lambda r: 1 if list(df[df['Issue_id'] == r['Issue_id_1']]['Component']) == list(df[df['Issue_id'] == r['Issue_id_2']]['Component']) else 0, axis = 1)
    df_pairs_neg['same_comp'] = df_pairs_neg.apply(lambda r: 1 if list(df[df['Issue_id'] == r['Issue_id_1']]['Component']) == list(df[df['Issue_id'] == r['Issue_id_2']]['Component']) else 0, axis = 1)

    df_pairs_pos['same_prio'] = df_pairs_pos.apply(lambda r: 1 if list(df[df['Issue_id'] == r['Issue_id_1']]['Priority']) == list(df[df['Issue_id'] == r['Issue_id_2']]['Priority']) else 0, axis = 1)
    df_pairs_neg['same_prio'] = df_pairs_neg.apply(lambda r: 1 if list(df[df['Issue_id'] == r['Issue_id_1']]['Priority']) == list(df[df['Issue_id'] == r['Issue_id_2']]['Priority']) else 0, axis = 1)

    df_pairs_pos['same_tw'] = df_pairs_pos.apply(lambda r: times_window(list(df[df['Issue_id'] == r['Issue_id_1']]['Created_time'])[0], list(df[df['Issue_id'] == r['Issue_id_1']]['Created_time'])[0]),axis = 1)
    df_pairs_neg['same_tw'] = df_pairs_neg.apply(lambda r: times_window(list(df[df['Issue_id'] == r['Issue_id_1']]['Created_time'])[0], list(df[df['Issue_id'] == r['Issue_id_1']]['Created_time'])[0]),axis = 1)
    '''
    df_pairs_pos = df_pairs_pos[['Title_1','Title_2']]  
    df_pairs_neg = df_pairs_neg[['Title_1','Title_2']]  
    
    '''
    df_pairs_neg['Title_1'].apply(lambda x: str(' '.join(x)))
    df_pairs_neg['Title_2'].apply(lambda x: str(' '.join(x)))
    df_pairs_pos['Title_1'].apply(lambda x: str(' '.join(x)))
    df_pairs_pos['Title_2'].apply(lambda x: str(' '.join(x)))
    '''
    '''
    df_pairs_neg.to_csv(neg_path)#, index=False, header=False)
    df_pairs_pos.to_csv(pos_path)#, index=False, header=False)
    '''
    '''
    ratios = [0.7, 0.1, 0.2]
    train_set = pd.concat([df_pairs_neg.iloc[range(int(ratios[0]*len(df_pairs_neg)))],df_pairs_pos.iloc[range(int(ratios[0]*len(df_pairs_pos)))]])
    test_set = pd.concat([df_pairs_neg.iloc[range(int(ratios[0]*len(df_pairs_neg)), int((ratios[1] + ratios[0])*len(df_pairs_neg)))],df_pairs_pos.iloc[range(int(ratios[0]*len(df_pairs_pos)), int((ratios[1] + ratios[0])*len(df_pairs_pos)))]])
    vali_set = pd.concat([df_pairs_neg.iloc[range(int((ratios[1] + ratios[0])*len(df_pairs_neg)), len(df_pairs_neg))],df_pairs_pos.iloc[range(int((ratios[1] + ratios[0])*len(df_pairs_pos)), len(df_pairs_pos))]])
    return train_set, test_set, vali_set


def load_glove_as_dict(filepath):
    word_vec = {}
    with open(filepath) as fr:
        for line in fr:
            line = line.split()
            word = line[0]
            vec = line[1:]
            word_vec[word] = vec
    return word_vec

if __name__ == '__main__':
     load_data(data_path)    
