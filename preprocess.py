import pandas as pd
import json
import time
from sklearn.model_selection import train_test_split
import random

type_class = {'A': 1, 'B': 2, 'C': 3}


# split train.csv 90% to training data, 10% to testing data
def split_dataset():
    train_data = pd.read_csv("./trip_data/train.csv")
    train, test = train_test_split(train_data, test_size=0.1, random_state=42)

    test.to_csv('test_split.csv', encoding='utf-8', index=False, na_rep='NA')
    train.to_csv('train_split.csv', encoding='utf-8', index=False, na_rep='NA')


# convert train_split.csv to my feature (read trip_data/data_mean.txt for more detail)
def pre_process(file, save_name):
    cols = ['TRIP_ID', 'CALL_TYPE', 'TIMESTAMP', 'DAY_TYPE', 'POLYLINE', 'MISSING_DATA']
    for iteration, train_data in enumerate(pd.read_csv("./trip_data/"+file+".csv", usecols=cols, iterator=True, chunksize=1000)):

        for tid, tr in enumerate(train_data['MISSING_DATA']):
            if tr:
                # print(train_data)
                train_data = train_data.drop([tid + iteration*1000])

        gps = train_data['POLYLINE'].apply(json.loads)  # amazing json
        COST = [(len(i)-1) * 15 for i in gps]

        CALL_TYPE = [type_class[i] for i in train_data['CALL_TYPE']]
        TIME_TYPE = [time.localtime(t)[3] for t in train_data['TIMESTAMP']]
        DAY_TYPE = [type_class[i] for i in train_data['DAY_TYPE']]

        write_dict = {'TRIP_ID': train_data['TRIP_ID'], 'CALL_TYPE': CALL_TYPE, 'TIME_TYPE': TIME_TYPE, 'DAY_TYPE': DAY_TYPE,
                      'ANSWER': COST, 'POLYLINE': gps}

        df = pd.DataFrame(write_dict)

        if iteration == 0:
            df.to_csv(save_name + '.csv', encoding='utf-8', index=False, header=True)
        else:
            df.to_csv(save_name+'.csv', encoding='utf-8', index=False,  header=False, mode='a')


# pre_process + remove the back few gps data(let testing data like kaggle test.csv)
def test_remove(file, save_name):
    cols = ['TRIP_ID', 'CALL_TYPE', 'TIMESTAMP', 'DAY_TYPE', 'POLYLINE', 'MISSING_DATA']
    for iteration, train_data in enumerate(pd.read_csv("./trip_data/"+file+".csv", usecols=cols, iterator=True, chunksize=1000)):

        for tid, tr in enumerate(train_data['MISSING_DATA']):
            if tr:
                # print(train_data)
                train_data = train_data.drop([tid + iteration*1000])

        gps = train_data['POLYLINE'].apply(json.loads)  # amazing json
        COST = [(len(i)-1) * 15 for i in gps]

        CALL_TYPE = [type_class[i] for i in train_data['CALL_TYPE']]
        TIME_TYPE = [time.localtime(t)[3] for t in train_data['TIMESTAMP']]
        DAY_TYPE = [type_class[i] for i in train_data['DAY_TYPE']]

        remove_gps = []
        for gid, gdata in enumerate(gps):
            k = random.randint(5, 45)
            if len(gdata) < k:
                remove_gps.append(gdata)
            else:
                remove_gps.append(gdata[0:k])

        write_dict = {'TRIP_ID': train_data['TRIP_ID'], 'CALL_TYPE': CALL_TYPE, 'TIME_TYPE': TIME_TYPE, 'DAY_TYPE': DAY_TYPE,
                      'ANSWER': COST, 'POLYLINE': remove_gps}

        df = pd.DataFrame(write_dict)

        if iteration == 0:
            df.to_csv(save_name + '.csv', encoding='utf-8', index=False, header=True)
        else:
            df.to_csv(save_name+'.csv', encoding='utf-8', index=False,  header=False, mode='a')


split_dataset()
test_remove("test_split", "pre_test_remove")
pre_process("train_split", "pre_train")

#
