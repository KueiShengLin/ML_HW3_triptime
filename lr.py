import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import numpy as np
from math import radians, cos, sin, asin, sqrt
import json


def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r


def pre_train(data):

    feature_list = [] #time, call, day, first, last, distance
    ans = data['ANSWER'].values.tolist()

    # row[0:ID 1:ANSWER, 2:CALL, 3:DAY, 4:POLY, 5:TIME, 6:Trip ID]
    for rid, row in enumerate(data.itertuples()):
        feature = []
        if 7 <= row[5] <= 10:
            feature = [1, 0, 0, 0]

        elif 17 <= row[5] <= 20:
            feature = [0, 1, 0, 0]

        elif 23 <= row[5] <= 24 or 0 <= row[5] <= 6:
            feature = [0, 0, 1, 0]

        else:
            feature = [0, 0, 0, 1]

        call = [0]*3
        call[row[2]-1] = 1
        feature += call

        day = [0]*3
        day[row[3]-1] = 1
        feature += day

        gps = json.loads(row[4])
        if len(gps) == 0:
            f_long, f_loti = 0, 0
            l_long, l_loti = 0, 0
            c_long, c_loti = 0, 0
            distance = 0
            cut_dis = 0
        else:
            f_long, f_loti = gps[0][0], gps[0][1]
            l_long, l_loti = gps[-1][0], gps[-1][1]
            if len(gps) < 5:
                c_long, c_loti = l_long, l_loti
            else:
                c_long, c_loti = gps[4][0], gps[4][1]
            distance = haversine(f_long, f_loti, l_long, l_loti)
            cut_dis = haversine(f_long, f_loti, c_long, c_loti)

        # feature.append(f_long)
        # feature.append(f_loti)
        # feature.append(l_long)
        # feature.append(l_loti)
        feature.append(l_long - f_long)
        feature.append(l_loti - f_loti)
        feature.append(c_long - f_long)
        feature.append(c_loti - f_loti)
        feature.append(cut_dis)
        feature.append(distance)

        feature_list.append(feature)

    return feature_list, ans


def score(prediction, y_hat, tol):
    acc = 0
    for pid,y in enumerate(prediction):
        if y_hat[pid] in range(y-tol, y+tol, 1):
            acc += 1
    return acc / len(y_hat)

clf = LogisticRegression( multi_class='multinomial', solver='sag', tol=0.1)
# clf = SGDRegressor() # LogisticRegression
# train_data = pd.read_csv("./dataset2/pre_train.csv")
# x_train, y_train = pre_train(train_data)
# clf.fit(x_train, y_train)

classes = []
# for iteration, train_data in enumerate(pd.read_csv("./dataset2/pre_train.csv", iterator=True, chunksize=1000)):
#     _, y_train = pre_train(train_data)
#     classes += np.unique(y_train).tolist()

classes = list(set(classes))
for iteration, train_data in enumerate(pd.read_csv("./dataset2/pre_train.csv", iterator=True, chunksize=1000000)):
    x_train, y_train = pre_train(train_data)
    clf.fit(x_train, y_train)

    if iteration % 1 == 0:
        print('train:', iteration)
        break
print("train_down")

test = pd.read_csv("./dataset2/pre_test.csv")
x_test, y_test = pre_train(test)
# score = clf.score(x_test, y_test)
prediction = clf.predict(x_test)

score_list = []

score_list.append(score(prediction, y_test, 0))
score_list.append(score(prediction, y_test, 30))
score_list.append(score(prediction, y_test, 60))
score_list.append(score(prediction, y_test, 100))
score_list.append(score(prediction, y_test, 150))
score_list.append(score(prediction, y_test, 200))
score_list.append(score(prediction, y_test, 300))
print(score_list)
np.savetxt('./training/lr_acc100' + '.txt', score_list, delimiter=',')
# print(score)
# for test in pd.read_csv("./dataset2/pre_test.csv", iterator=True, chunksize=1000):
#     x_test, y_test = pre_train(test)
#     score = clf.score(x_test, y_test)
#     print(score)
#
