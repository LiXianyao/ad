#-*- coding:utf-8 -*-#
import xgboost as xgb
import time
import csv
import sys
import ConfigParser
import datetime
import numpy as np
import re
import json

reload(sys)
sys.setdefaultencoding("utf-8")

class models(object):
    model_dir = "./model2/"
    ##缺省参数（用配置文件更新）
    ##缺省参数（用配置文件更新）
    boost_param = {'max_depth': 5, 'eta': 0.25, 'silent': 1, 'objective': 'binary:logistic',
                   'eval_metric': 'auc', 'subsample': 0.8}
    train_param = {'num_boost_round': 450, 'early_stopping_rounds': 30}

    def __init__(self, ConfigFile):
        ##读取配置文件，确定迅雷参数、训练/测试数据路径
        self.configFile = ConfigFile
        self.get_configure()

    def procedure(self):
        ##流水线：调用训练函数训练模型-》保存模型及对应的参数-》调用测试函数检测查全率
        self.data_train = xgb.DMatrix(self.task_dict["train_file"])
        self.data_test = xgb.DMatrix(self.task_dict["test_file"])
        self.bst = self.train_model()
        self.save_model(self.bst, self.task_dict["model_id"])
        print "训练效果自检测"
        self.test_model(self.bst, self.data_train)
        print "测试数据对比"
        self.test_model(self.bst, self.data_test)

    def test_only(self):
        self.data_test = xgb.DMatrix(self.task_dict["test_file"])
        self.bst = xgb.Booster()
        self.bst.load_model(self.task_dict["test_model"])
        print "测试数据对比"
        self.test_model(self.bst, self.data_test, test_only=True)

    def train_model(self):
        ##使用训练数据训练
        print "train_start"
        watchlist = [(self.data_test, "eval"), (self.data_train, "train")]
        bst = xgb.train(self.boost_param, self.data_train, num_boost_round=self.train_param["num_boost_round"],
                        evals=watchlist, early_stopping_rounds=self.train_param["early_stopping_rounds"])
        print "train_end"
        return bst #返回模型

    def save_model(self, bst, model_id = "M003"):
        timestamp = re.search("[0-9]+", self.task_dict["train_file"]).group()
        if timestamp == None or len(timestamp) == 0:
            timestamp = "amodel"
        self.model_name = timestamp + "_" + model_id + ".model"
        bst.save_model(self.model_dir + self.model_name)
        self.save_param(self.model_name)

    def save_param(self, model_name):
        param_file = model_name.replace(".model", ".param")
        with open(self.model_dir + param_file, "wb") as fparam:
            fparam.write(str(self.task_dict) + "\n")
            fparam.write(str(self.boost_param) + "\n")
            fparam.write(str(self.train_param) + "\n")

    def test_model(self, bst, data_test, test_only = False):
        ##读取测试数据

        test_size = data_test.num_row()

        ##测试数据测试
        start_time_s = datetime.datetime.now()
        y_hat = bst.predict(data_test)
        endtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
        end_time_s = datetime.datetime.now()
        cost_time_s = (end_time_s - start_time_s).microseconds

        y = data_test.get_label()##获取测试数据label

        ##检查命中率
        cnt_predict = {0: 0, 1: 0}
        cnt_origin = {0: 0, 1: 0}
        fit_cnt = {0: 0, 1: 0} ##预测与实际一致的计数
        #pre_true_cnt = {}
        #pre_false_cnt = {}
        pre_prob = []
        for i in range(test_size):
            #print y_hat[i]
            #print type(y_hat[i])
            #print len(y_hat[i])
            predict_res = int(y_hat[i] >= 0.8)
            pre_prob.append(y_hat[i])
            true_label = int(y[i])
            cnt_predict[predict_res] += 1
            cnt_origin[true_label] += 1
            if predict_res == true_label:
                fit_cnt[true_label] += 1
            """
            elif true_label == 1:
                loc = round(y_hat[i], 1)
                if loc not in pre_true_cnt.keys():
                    pre_true_cnt[loc] = 1
                else:
                    pre_true_cnt[loc] += 1
            elif true_label == 0:
                loc = round(y_hat[i], 1)
                if loc not in pre_false_cnt.keys():
                    pre_false_cnt[loc] = 1
                else:
                    pre_false_cnt[loc] += 1
            """

        ##计算AUC
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y, pre_prob, pos_label=1)
        sklearn_auc = auc(fpr, tpr)

        if cnt_predict[1] == 0:
            cnt_predict[1] = 1
        #print pre_true_cnt
        #print pre_false_cnt
        print "预测结果: ", cnt_predict
        print "实际标签: ", cnt_origin
        print "对比结果： ", fit_cnt, "准确率 = %f, %f" % (
        float(fit_cnt[0]) / cnt_predict[0], float(fit_cnt[1]) / cnt_predict[1])
        print "label=1查全率 = ", float(fit_cnt[1]) / cnt_origin[1]
        print "auc = ", sklearn_auc
        if test_only == False:
            record_file = self.model_name.replace(".model", ".record")
            with open(self.model_dir + record_file, "ab") as frecord:
                #frecord.write(str(pre_true_cnt) + "\n")
                frecord.write("预测结果: " + str(cnt_predict) + "\n")
                frecord.write("实际标签: "+ str(cnt_origin ) + "\n")
                frecord.write("对比结果： " + str(fit_cnt) + "准确率 = %f, %f"%(float(fit_cnt[0])/cnt_predict[0], float(fit_cnt[1])/cnt_predict[1]) + "\n")
                frecord.write("label=1查全率 =  " + str(float(fit_cnt[1]) / cnt_origin[1]) + "\n")
                frecord.write("auc = "+ str(sklearn_auc) + "\n")

    def get_configure(self): ##获取配置文件的内容
        self.cf = ConfigParser.ConfigParser()
        self.cf.read(self.configFile)
        self.boost_param.update(self.config_from_sec("boost_param"))
        self.train_param = self.config_from_sec("train_param")
        self.task_dict = self.config_from_sec("task")
        self.statistics_dict = self.config_from_sec("statistics_file")

    def config_from_sec(self, sec_name):
        sec = self.cf.options(sec_name)
        config_dict = {}
        for opt_name in sec:
            opt_value = self.cf.get(sec_name, opt_name)
            try:  ##尝试转为整型
                config_dict[opt_name] = int(opt_value)
            except ValueError: ##尝试转为浮点
                try:
                    config_dict[opt_name] = float(opt_value)
                except ValueError:
                    config_dict[opt_name] = opt_value
        print config_dict
        return config_dict

"""脚本的命令行输入提示"""
def printUsage():
    print "usage: train_10.py -f <configFileName>"

if __name__=="__main__":
    import getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:", ["file="])
    except getopt.GetoptError:
        printUsage()
        sys.exit(-1)

    configFile = "train.config"
    for opt, arg in opts:
        if opt in ("-f","--file"):
            configFile = arg
    print configFile
    new_model = models(configFile)
    new_model.procedure()
    #new_model.test_only()