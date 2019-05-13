# -*- coding: utf8 -*-
import argparse
import csv
import random
import numpy as np
import tensorflow as tf
import pickle
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class PairWise:
    def __init__(self, argv):
        self.batch_size = argv.batch_size
        self.data_dimension = argv.data_dimension
        self.layernum = argv.layernum
        self.aerfa = argv.aerfa
        self.dim_y = argv.dim_y
        self.init_dim = int(pow(2, self.layernum-1)*self.aerfa*self.data_dimension)
        # (data_dimension, 1) => (init_dim, 1) => (init_dim/2, 1)
        self.W = [tf.Variable(tf.random_uniform([self.data_dimension, self.init_dim], -1, 1))]
        self.B = [tf.Variable(tf.zeros([self.init_dim]))]
        for i in range(self.layernum-1):
            self.W.append(tf.Variable(tf.random_uniform(
                [int(self.init_dim/pow(2, i)), int(self.init_dim/pow(2, i+1))], -1, 1)))
            self.B.append(tf.Variable(tf.zeros([int(self.init_dim/pow(2, i+1))])))

        self.W_ = []
        self.B_ = []
        for i in range(self.layernum-1):
            self.W_.append(tf.Variable(tf.random_uniform(
                [int(self.init_dim/pow(2, self.layernum-i-1)), int(self.init_dim/pow(2, self.layernum-i-2))], -1, 1)))
            self.B_.append(tf.Variable(tf.zeros([int(self.init_dim/pow(2, self.layernum-i-2))])))

        self.W_.append(tf.Variable(tf.random_uniform([self.init_dim, self.data_dimension], -1, 1)))
        self.B_.append(tf.Variable(tf.zeros([self.data_dimension])))

        self.Wc = tf.Variable(tf.random_uniform([int(self.aerfa*self.data_dimension), self.dim_y], -1, 1))
        self.Bc = tf.Variable(tf.zeros([self.dim_y]))

    def encoder(self, x):
        y = tf.matmul(x, self.W[0]) + self.B[0]  #
        y = tf.nn.relu(y)
        w_mean, w_var = tf.nn.moments(y, 0)
        scale = tf.Variable(tf.ones([y.shape[1]]))
        offset = tf.Variable(tf.zeros([y.shape[1]]))
        variance_epsilon = 0.001
        y = tf.nn.batch_normalization(y, w_mean, w_var, offset, scale, variance_epsilon)
        print(y.shape)

        for w, b, in zip(self.W[1:], self.B[1:]):
            # print(w.shape, b.shape)
            y = tf.matmul(y, w) + b
            y = tf.nn.relu(y)
            w_mean, w_var = tf.nn.moments(y, 0)
            scale = tf.Variable(tf.ones([y.shape[1]]))
            offset = tf.Variable(tf.zeros([y.shape[1]]))
            variance_epsilon = 0.001
            y = tf.nn.batch_normalization(y, w_mean, w_var, offset, scale, variance_epsilon)

            print(y.shape)
        return y

    def decoder(self, y):
        y1 = y
        # print(y1.shape)

        for w_, b_, in zip(self.W_, self.B_):
            # print(w_.shape, b_.shape)
            y1 = tf.matmul(y1, w_) + b_
            y1 = tf.nn.relu(y1)
            w_mean, w_var = tf.nn.moments(y1, 0)
            scale = tf.Variable(tf.ones([y1.shape[1]]))
            offset = tf.Variable(tf.zeros([y1.shape[1]]))
            variance_epsilon = 0.001
            y1 = tf.nn.batch_normalization(y1, w_mean, w_var, offset, scale, variance_epsilon)
            # print("???")
            print(y1.shape)

        return y1  # y是压缩后的，y1是经过多层变换后还原的

    def train1(self, x):
        z = self.encoder(x)
        y1 = self.decoder(z)
        print(y1.shape)

        return y1                # z是压缩后的，y1是经过多层变换后还原的

    def train2(self, x, y):
        z = self.encoder(x)
        pre_y = tf.nn.softmax(tf.matmul(z, self.Wc) + self.Bc)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=pre_y, labels=y)
        loss_mean = tf.reduce_mean(loss)
        return loss_mean

    def train3(self, x1, x2, y):
        def cosine(q, a):
            pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
            pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
            pooled_mul_12 = tf.reduce_sum(q * a, 1)
            score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
            return score

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        # res = tf.abs(cosine(z1, z2))
        res = cosine(z1, z2)
        # res = tf.Print(res, [res], message=" lalalala")
        aa = tf.constant(0.8, shape=[self.batch_size, ])    # （0.9, 0.3)
        bb = tf.constant(0.2, shape=[self.batch_size, ])
        ones = tf.ones(shape=[self.batch_size])
        cross_batch = y*tf.maximum(0., aa-res) + (ones-y)*tf.maximum(0., res-bb)
        cross_mean = tf.reduce_mean(cross_batch)
        return cross_mean

    def get_y(self, x):
        y = tf.matmul(x, self.W[0]) + self.B[0]  #
        print(y.shape)

        for w, b, in zip(self.W[1:], self.B[1:]):
            # print(w.shape, b.shape)
            y = tf.matmul(y, w) + b
            print(y.shape)
        y1 = y

        return y1  # y是压缩后的，y1是经过多层变换后还原的


def get_data(data_file):
    textFiles = csv.reader(open(data_file))
    data = list(textFiles)
    length = len(data[0])
    for d in data:
        # print(d)
        for i in range(len(d)):
            if d[i] is "" or d[i] is None:
                d[i] = 0
                print("0000000")
    # print(data[0])
    train_data = [d[1:-1] for d in data]
    train_label = [d[-1] for d in data]
    print(length, len(train_data[0]), len(train_label[0]))
    return train_data, train_label


def generate_data(train_data, train_label):
    def get_idx(min, max, sample):
        return np.random.randint(min, max, [sample])

    def split(label, train_data):
        idx1 = [i for i in range(len(label)) if int(label[i]) == 1]
        idx0 = list(set(range(len(train_label)))-set(idx1))
        # print("len(idx0)    :", len(idx0))
        # print("len(idx1)    :", len(idx1))
        data0 = train_data[idx0]
        data1 = train_data[idx1]
        return data0, data1

    data0, data1 = split(train_label, train_data)
    # print("type(data0): ", type(data0), data0.shape)
    # print("type(data1): ", type(data1), data1.shape)
    d0, d1 = [], []
    for d in data1:
        # print(d)
        idx = get_idx(0, len(data0), 4)
        d1 += [d] * 4
        # print(data0[idx])
        d0 += list(data0[idx])
    # print(" 10 ")
    # print(np.array(d0).shape)
    # print(np.array(d1).shape)
    data10 = [d0, d1]

    doo, d0 = [], []
    idxoo = get_idx(0, len(data0), len(data1))
    for i in idxoo:
        idx = get_idx(0, len(data0), 2)
        doo += [list(data0[i])] * 2
        d0 += list(data0[idx])
    # print(" 00 ")
    # print(np.array(d0).shape)
    # print(np.array(doo).shape)
    data00 = [doo, d0]

    dll, d1 = [], []
    for d in data1:
        idx = get_idx(0, len(data1), 2)
        dll += [d] * 2
        d1 += list(data1[idx])
    # print(" 11 ")
    # print(np.array(dll).shape)
    # print(np.array(d1).shape)
    data11 = [dll, d1]

    new_label = [1] * len(data00[0]) + [0] * len(data10[0]) + [1] * len(data11[0])
    new_data = [data00[0] + data10[0] + data11[0], data00[1] + data10[1] + data11[1]]
    # print("len(new_label);  ", len(new_label))
    return new_data, new_label


def trans_train_label(train_label):
    label = []
    for i in train_label:
        tem = [0, 0]
        # ys.append(y)
        tem[int(i)] = 1
        label.append(tem)
    return np.array(label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameter')
    parser.add_argument("--input_pt", type=str, default="../data/UCI.csv", help="dataset name")  # 数据集
    parser.add_argument("--output_pt", type=str, default="", help="dataset name")  # 数据集
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--layernum", type=int, default=3, help="layer number")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--aerfa", type=float, default=0.5, help="aerfa")
    parser.add_argument("--data_dimension", type=int, default=24, help="data dimension")
    parser.add_argument("--dim_y", type=int, default=2, help="label dimension")
    parser.add_argument("--epochs", type=int, default=50, help="epoch num")

    argv = parser.parse_args()

    data_file = argv.input_pt
    train_data, train_label = get_data(data_file)
    # 减均值，除方差
    train_data = np.array(train_data, dtype=np.float)
    train_data = (train_data-np.mean(train_data, axis=0))/np.std(train_data, axis=0)
    # 1->[0, 1]
    train_label2 = trans_train_label(train_label)

    argv.data_dimension = len(train_data[0])

    x = tf.placeholder(tf.float32, [argv.batch_size, len(train_data[0])])
    x1 = tf.placeholder(tf.float32, [None, len(train_data[0])])
    y = tf.placeholder(tf.float32, [argv.batch_size, argv.dim_y])

    x2 = tf.placeholder(tf.float32, [argv.batch_size, len(train_data[0])])
    y2 = tf.placeholder(tf.float32, [argv.batch_size, ])

    model = PairWise(argv)
    y1 = model.train1(x)
    cost1 = tf.reduce_mean(tf.square(x-y1))
    optimizer1 = tf.train.AdamOptimizer(learning_rate=argv.lr).minimize(cost1)

    loss2 = model.train2(x, y)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=argv.lr).minimize(loss2)

    loss3 = model.train3(x, x2, y2)
    optimizer3 = tf.train.AdamOptimizer(learning_rate=argv.lr).minimize(loss3)

    z = model.get_y(x1)

    print("ACTION!")
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # 变量初始化
        sess.run(init)
        res_y = None
        total_batch = int(len(train_data) / argv.batch_size)
        for epoch in range(argv.epochs):
            avg_cost = 0
            data = train_data
            for i in range(total_batch):
                batch_x = data[i*argv.batch_size:(i+1)*argv.batch_size]
                _, c = sess.run([optimizer1, cost1], feed_dict={x: batch_x})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
        print(" Auto-Encoder Successful!")
        print(" \n=============================================================================\n")

        for epoch in range(argv.epochs//2):
            avg_cost = 0
            data = train_data
            label = train_label2
            for i in range(total_batch):
                batch_x = data[i*argv.batch_size:(i+1)*argv.batch_size]
                batch_y = label[i*argv.batch_size:(i+1)*argv.batch_size]

                _, c = sess.run([optimizer2, loss2], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
        print(" Classification Over!")

        print(" \n=============================================================================\n")

        print(" Pair-Wise Action!")
        [data1, data2], label = generate_data(train_data, train_label)
        print("len(data1), len(data2), len(label):  ", len(data1), len(data2), len(label))
        # label = np.array(label, np.int32)
        a = list(range(len(data1)))
        total_batch = int(len(a) / argv.batch_size)
        for epoch in range(argv.epochs//2):
            avg_cost = 0
            random.shuffle(a)
            for i in range(total_batch):
                b = a[i * argv.batch_size:(i + 1) * argv.batch_size]
                batch_x1 = np.array([data1[i] for i in b])
                batch_x2 = np.array([data2[i] for i in b])
                batch_y = np.array([label[i] for i in b])

                _, c = sess.run([optimizer3, loss3], feed_dict={x: batch_x1, x2: batch_x2, y2: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
        print(" Pair-Wise Over!")

        res_y = np.array(sess.run([z], feed_dict={x1: train_data}))

        fin = open("../result/auto_encoder/ae_tc_pw_no_abs.pkl", 'wb')
        pickle.dump([res_y, label], fin)
        fin.close()
        print(" Pickle Successful!")


