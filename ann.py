import json
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    ans = [[i] for i in ans]

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


def add_layer(inputs, input_tensors, output_tensors, activation_function=None):
    with tf.name_scope('Weights'):
        W = tf.Variable(tf.random_normal([input_tensors, output_tensors]))
    with tf.name_scope('Biases'):
        b = tf.Variable(tf.random_normal([output_tensors]))
    with tf.name_scope('Formula'):
        formula = tf.add(tf.matmul(inputs, W), b)
    if activation_function is None:
        outputs = formula
    else:
        outputs = activation_function(formula)
    return outputs


def train():
    for iteration, train_data in enumerate(pd.read_csv("./dataset2/pre_train.csv", iterator=True, chunksize=1)):
        x_train, y_train = pre_train(train_data)
        _, cost, prediction = sess.run([train_op, loss, output_layer], feed_dict={x_feeds: x_train, y_feeds: y_train})
        if iteration % 1000 == 0:
            print('data point:', iteration)
            print('cost', cost)
            print('prediction:', prediction[0])
            print("Y:", y_train[0])

        if iteration % 100000 == 0:
            saver = tf.train.Saver()
            save_path = saver.save(sess, "/work/ML/HW3/ML_HW3_triptime/save/save.ckpt")
            print("Model saved in file: %s" % save_path)


def test():
    saver = tf.train.Saver()
    saver.restore(sess, "/work/ML/HW3/ML_HW3_triptime/save/save.ckpt")
    test_data = pd.read_csv("./dataset2/pre_test.csv")
    x_test, y_test = pre_train(test_data)
    prediction = sess.run([output_layer], feed_dict={x_feeds: x_test, y_feeds: y_test})
    score_list = []
    score_list.append(score(prediction, y_test, 0))
    score_list.append(score(prediction, y_test, 30))
    score_list.append(score(prediction, y_test, 60))
    score_list.append(score(prediction, y_test, 100))
    score_list.append(score(prediction, y_test, 150))
    score_list.append(score(prediction, y_test, 200))
    score_list.append(score(prediction, y_test, 300))
    print(score_list)
    np.savetxt('./training/ann_acc' + '.txt', score_list, delimiter=',')

    # for iteration, test_data in enumerate(pd.read_csv("./dataset2/pre_test.csv", iterator=True, chunksize=1)):
    #     x_test, y_test = pre_train(test_data)
    #     prediction = sess.run([output_layer], feed_dict={x_feeds: x_test, y_feeds: y_test})
    #     prediction = round(prediction[0][0][0])


def score(prediction, y_hat, tol):
    acc = 0
    for pid, y in enumerate(prediction):
        if y_hat[pid] in range(y-tol, y+tol, 1):
            acc += 1
    return acc / len(y_hat)


INPUT = 16
OUTPUT = 1
HIDDEN = 5

with tf.name_scope('Input'):
    x_feeds = tf.placeholder(tf.float32, shape=[None, INPUT], name='input')

with tf.name_scope('Label'):
    y_feeds = tf.placeholder(tf.float32, shape=[None, OUTPUT], name='label')

with tf.name_scope('Layer1'):
    w1 = tf.Variable(tf.random_normal([INPUT, HIDDEN]), name='w1')
    b1 = tf.Variable(tf.random_normal([HIDDEN]), name='b1')
    formula1 = tf.add(tf.matmul(x_feeds, w1), b1)
    hidden_layer1 = tf.nn.relu(formula1)

    w2 = tf.Variable(tf.random_normal([HIDDEN, HIDDEN]), name='w2')
    b2 = tf.Variable(tf.random_normal([HIDDEN]), name='b2')
    formula2 = tf.add(tf.matmul(hidden_layer1, w2), b2)
    hidden_layer2 = tf.nn.relu(formula2)

with tf.name_scope('Output'):
    wo = tf.Variable(tf.random_normal([HIDDEN, OUTPUT]), name='wo')
    bo = tf.Variable(tf.random_normal([OUTPUT]), name='bo')
    formula_o = tf.add(tf.matmul(hidden_layer2, wo), bo)
    output_layer = tf.nn.relu(formula_o)
    # output_layer = add_layer(inputs=hidden_layer2, input_tensors=HIDDEN, output_tensors=OUTPUT, activation_function=tf.nn.relu)

with tf.name_scope('Loss'):
    loss = tf.losses.mean_squared_error(labels=y_feeds, predictions=output_layer)
    tf.summary.scalar('loss', loss)

with tf.name_scope('Training'):
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))

sess.run(init)

train()
# test()
#
