# -*- coding: utf8 -*-

# -*- coding: utf8 -*-
import argparse
import csv
import numpy as np
import tensorflow as tf
# from pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark import SparkContext,SparkConf
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext


class Vae:
    def __init__(self, argv):
        self.batch_size = argv.batch_size
        self.data_dimension = argv.data_dimension
        self.n_hidden = argv.n_hidden
        self.dim_z = argv.dim_z

    def get_z(self, x, keep_prob):
        mu, sigma = self.gaussian_MLP_encoder(x, keep_prob, name="get_z")

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        return z

    def gaussian_MLP_encoder(self, x,  keep_prob, reuse=False, name="train"):

        with tf.variable_scope("gaussian_MLP_encoder_%s" %name, reuse=reuse):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # 1st hidden layer
            w0 = tf.get_variable('w0', [x.get_shape()[1], self.n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [self.n_hidden], initializer=b_init)
            h0 = tf.matmul(x, w0) + b0
            h0 = tf.nn.elu(h0)
            h0 = tf.nn.dropout(h0, keep_prob)

            # 2nd hidden layer
            w1 = tf.get_variable('w1', [h0.get_shape()[1], self.n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [self.n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.tanh(h1)
            h1 = tf.nn.dropout(h1, keep_prob)

            # output layer
            # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
            wo = tf.get_variable('wo', [h1.get_shape()[1], self.dim_z * 2], initializer=w_init)
            bo = tf.get_variable('bo', [self.dim_z * 2], initializer=b_init)
            gaussian_params = tf.matmul(h1, wo) + bo

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.dim_z]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.dim_z:])

        return mean, stddev

    # Bernoulli MLP as decoder
    def bernoulli_MLP_decoder(self, z, keep_prob, reuse=False):

        with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # 1st hidden layer
            w0 = tf.get_variable('w0', [z.get_shape()[1], self.n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [self.n_hidden], initializer=b_init)
            h0 = tf.matmul(z, w0) + b0
            h0 = tf.nn.tanh(h0)
            h0 = tf.nn.dropout(h0, keep_prob)

            # 2nd hidden layer
            w1 = tf.get_variable('w1', [h0.get_shape()[1], self.n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [self.n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.elu(h1)
            h1 = tf.nn.dropout(h1, keep_prob)

            # output layer-mean
            wo = tf.get_variable('wo', [h1.get_shape()[1], self.data_dimension], initializer=w_init)
            bo = tf.get_variable('bo', [self.data_dimension], initializer=b_init)
            y = tf.sigmoid(tf.matmul(h1, wo) + bo)

        return y

    # Gateway
    def autoencoder(self, x_hat, x, keep_prob):

        # encoding
        mu, sigma = self.gaussian_MLP_encoder(x_hat,  keep_prob)

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # decoding
        y = self.bernoulli_MLP_decoder(z,  keep_prob)
        y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

        # loss
        print("x.shape:   ", x.shape)
        print("y.shape:   ", y.shape)
        marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = marginal_likelihood - KL_divergence

        loss = -ELBO

        return y, z, loss, -marginal_likelihood, KL_divergence

    def decoder(self, z):

        y = self.bernoulli_MLP_decoder(z,  1.0, reuse=True)

        return y


def get_data(data_file):
    sc = SparkContext(conf=SparkConf().setAppName("read data"))
    sc.setLogLevel("ERROR")
    textFiles=sc.textFile(data_file).collect()
    # textFiles = sc.pickleFile(data_file)

    data = [row.split(",") for row in textFiles]
    length = len(data[0])
    data1 = np.array(data[0], np.float32)
    print(data[0])
    print(data1)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameter')
    parser.add_argument("--input_pt", type=str, default="data/UCI.csv", help="dataset name")  # 数据集
    parser.add_argument("--output_pt", type=str, default="", help="dataset name")  # 数据集
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--dim_z", type=int, default=20, help="layer number")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--aerfa", type=float, default=1, help="aerfa")
    parser.add_argument("--data_dimension", type=int, default=24, help="data dimension")
    parser.add_argument("--epochs", type=int, default=20, help="epoch num")

    argv = parser.parse_args()
    data_file = argv.input_pt
    # 数据归一化
    train_data, train_label = get_data(data_file)
    train_data = np.array(train_data, dtype=np.float)
    train_data = (train_data-np.mean(train_data, axis=0))/np.std(train_data, axis=0)

    argv.data_dimension = len(train_data[0])
    argv.dim_z = int(argv.data_dimension*argv.aerfa)
    argv.n_hidden = 2*argv.dim_z

    vae = Vae(argv)

    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x_hat = tf.placeholder(tf.float32, shape=[argv.batch_size, len(train_data[0])])
    x = tf.placeholder(tf.float32, shape=[None, len(train_data[0])])

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # input for PMLR
    z_in = tf.placeholder(tf.float32, shape=[None, len(train_data[0])], name='latent_variable')

    # network architecture
    y, z, loss, neg_marginal_likelihood, KL_divergence = vae.autoencoder(x_hat, x, keep_prob)

    # optimization
    train_op = tf.train.AdamOptimizer(argv.lr).minimize(loss)

    z = vae.get_z(x, keep_prob=0.9)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 变量初始化
        sess.run(init)
        res_y = None
        total_batch = int(len(train_data) / argv.batch_size)
        for epoch in range(argv.epochs):
            avg_cost = 0
            np.random.shuffle(train_data)
            data = train_data
            for i in range(total_batch):
                batch_x = data[i*argv.batch_size:(i+1)*argv.batch_size]
                _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                    (train_op, loss, neg_marginal_likelihood, KL_divergence),
                    feed_dict={x_hat: batch_x, x: batch_x, keep_prob: 0.9})

                avg_cost += tot_loss / total_batch
            print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                epoch, tot_loss, loss_likelihood, loss_divergence))

            print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))

        res_z = np.array(sess.run(z, feed_dict={x: train_data, keep_prob: 0.9}))
    # res_data = []
    # for f, t in zip(res_z, train_label):
    #     tem = []
    #     for ff in f:
    #         tem.append(float(ff))
    #     res_data.append([Vectors.dense(tem), float(t[0])])
    #
    # spark = SparkSession.builder.appName("DataFrame").getOrCreate()
    #
    # print(type(res_data))
    # print(np.array(res_data).shape)
    # spark_df = spark.createDataFrame(res_data, ["features", "label"])
    #
    # spark_df.write.parquet(argv.output_pt)





