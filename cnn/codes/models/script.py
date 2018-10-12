
import pandas as pd


spark = pd.read_csv('spark/cnn_train_new.csv')
hadoop = pd.read_csv('hadoop/cnn_train_new.csv')
hdfs = pd.read_csv('hdfs/cnn_train_new.csv')
mapreduce = pd.read_csv('mapreduce/cnn_train_new.csv')

total = pd.concat([spark, hadoop, hdfs, mapreduce])
total.to_csv('cnn_train_new.csv', index= False)
