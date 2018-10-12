import pandas as pd


spark_pos = pd.read_csv('spark/pos.csv', names = ['a','b'])
spark_neg = pd.read_csv('spark/neg.csv', names = ['a','b'])
hdfs_pos = pd.read_csv('hdfs/pos.csv', names = ['a','b'])
hdfs_neg = pd.read_csv('hdfs/neg.csv', names = ['a','b'])
mapreduce_pos = pd.read_csv('mapreduce/pos.csv', names = ['a','b'])
mapreduce_neg = pd.read_csv('mapreduce/neg.csv', names = ['a','b'])
hadoop_pos = pd.read_csv('hadoop/pos.csv', names = ['a','b'])
hadoop_neg = pd.read_csv('hadoop/neg.csv', names = ['a','b'])
# pmes = ['a','b']rint(spark_pos.head())
total_pos = pd.concat([spark_pos, hdfs_pos, mapreduce_pos, hadoop_pos], axis = 0)
total_neg = pd.concat([spark_neg, hdfs_neg, mapreduce_neg, hadoop_neg], axis = 0)

total_pos.to_csv('total_pos.csv', header = None, index=False)
total_neg.to_csv('total_neg.csv', header = None, index=False)

print(total_pos.shape)

