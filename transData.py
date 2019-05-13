#-*-encoding:utf-8-*-#

import sys
import csv
import math
import numpy as np
import re
#import pandas as pd
reload(sys)
sys.setdefaultencoding("utf-8")
##数据字段转化为编码、分割测试数据、训练数据，保存为文件

class transData(object):
    tag_seg = []
    addition_data = {}
    trans_data_set = {}
    trans_title = {}
    maxcnt_file = "combine/maxcnt.txt"
    scaling_file = "combine/scaling.txt"
    app_bundle_file = "apart/appbundle.txt"
    maxcnt_dict = ""
    saving_tail = "_train.txt"
    country = "sau"
    file_tail = ".txt"
    delimiter = "\01"
    monitor_interval = 10000000
    digit_seg = 5
    rate = 50
    float_scal = 1
    cnt_true = 0
    cnt_false = 0
    roundf = 5
    testFile = False
    assume = 6000
    FalseStack = []
    TrueStack = []
    saving_dir = "trainn/expNobundleNoadh12/"
    #saving_dir = "trainn/expNoBundle/"

    def __init__(self):
        self.app_bundle = self.get_json(self.app_bundle_file)
        self.app_bundle_size = len(self.app_bundle)
        self.maxcnt_dict = self.get_json(self.maxcnt_file)
        self.scaling = self.get_json(self.scaling_file)
        self.scaling["mb_rate"] = self.scaling["app_rate"]
        self.scaling["mb_size"] = self.scaling["app_size"]

    def trans_txt(self, dir_name):
        filenames = self.get_file_names(dir_name)
        for i in range(len(filenames)):
            filename = filenames[i]
            if not self.testFile and i == len(filenames) - 1:
                continue
            elif self.testFile and i < len(filenames) - 1:
                continue
            fp = open(filename, "rU")
            data_ymd_stamp = re.search('[0-9]+', filename).group()
            train_data_name = data_ymd_stamp + "_" + self.country + "_d" + str(self.digit_seg) + "_s" + str(self.rate) + "_f" + str(self.float_scal)
            with open(self.saving_dir + train_data_name + ".data", "wb") as temp:
                with open(self.saving_dir + train_data_name + ".csv", "wb") as temp_csv:
                    csv_writer = csv.writer(temp_csv)
                    csv_writer.writerow(self.tag_seg)
                    row_cnt = 0
                    trans_cnt = 0
                    self.cnt_false = 0
                    self.cnt_true = 0
                    self.init_addition_data()
                    for buf in fp:
                        row_cnt += 1
                        if row_cnt == 1:
                            continue
                        partition = buf.split(self.delimiter)
                        if len(partition) != self.partLen:
                            print u"第%d行数据字段异常, content = %s"%(row_cnt, buf)
                            continue
                        data = dict(zip(self.tag_seg, partition))
                        is_need = self.trans_row(data, row_cnt, temp, csv_writer)
                        if is_need == True:
                            #break
                            trans_cnt += 1
                        if row_cnt % self.monitor_interval == 0:
                            print "row_cnt = %d" % (row_cnt / self.monitor_interval)
                            #break
                    print "proceed end, total row as %d, target row as %d" % (row_cnt, trans_cnt)
                    print "has true: false as %d :%d"%(self.cnt_true, self.cnt_false)

                    if not self.testFile:
                        print "now proceed False part"
                        self.cnt_false = 0
                        self.cnt_true = 0
                        row_cnt = 0
                        self.FalseStack.extend(self.TrueStack)
                        print len(self.FalseStack)
                        self.assume = len(self.TrueStack)
                        seed = 1
                        np.random.seed(seed)
                        #np.random.shuffle(self.FalseStack)
                        for data in self.FalseStack:
                            row_cnt += 1
                            is_need = self.trans_row(data, row_cnt, temp, csv_writer, trans_stack=True)
                            if row_cnt % self.monitor_interval == 0:
                                print "row_cnt = %d, cnt false=%d" % (row_cnt / self.monitor_interval, self.cnt_false)
                                # break
                        print "proceed end, total row as %d" % row_cnt
                        print "has true: false as %d :%d" % (self.cnt_true, self.cnt_false)
                        del self.TrueStack[:]
                        del self.FalseStack[:]
            fp.close()

    def get_file_names(self, dir_name):
        data_file_names = []
        import os
        for dir, subdir, files in os.walk(dir_name):
            for file in files:
                if file.find("_combine.csv")!= -1:
                    data_file_names.append(dir + "/" + file)
            return data_file_names

    def get_json(self, file_path):
        with open(file_path, "rb") as df:
            json_str = df.readline()
            import json
            data = json.loads(json_str)
        #print len(data)
        return data

    def init_addition_data(self):
        pass

    def trans_row(self, data, row_cnt, fw, fw_csv, trans_stack=False):
        return False

    def digit_apart(self, num, depth):
        if num < 10:
            return depth + int(num / self.digit_seg)
        last = int(bool(10 % self.digit_seg > 0))
        return self.digit_apart(num/10, depth + 10/self.digit_seg + last)

    def sigmoid(self, x):
        return 1.0 / (1 + math.exp(x * -1.0))

class tag_click_convert(transData):

    prev_seg = ["brand", "model", "advertiser_network_id", "advertiser_id", "app_id",
                "os_version", "media_bundle"]
    prev_seg_tail = ["_req", "_conv", "_conv_rate"]
    prev_mask = ["advertiser_network_id"]


    def __init__(self, data_dir):
        super(tag_click_convert, self).__init__()

        self.tag_seg = ["convert", "app_plat", "con_type", "h", "app_rank", "app_rate", "app_price",
                   "app_bundle", "app_size",
                   "mb_rank", "mb_rate", "mb_price", "mb_bundle", "mb_size", "did_last", "did_cnt", "did_conv_app",
                   "did_conv_bundle"]
        self.tag_seg_encode_len = [1, 1, 'c', 'c', 'c', 's', 1, 'm', 's',
                                   'm', 'm', 'm', 'm', 'm',
                                   #'c', 's', 1, self.app_bundle_size + 1, 's',
                                   1, 1, 1, 'm']
        for seg in self.prev_seg:
            ###每个统计指标有三个字段
            statis_seg = [seg + "_req", seg + "_conv", seg + "_conv_r", seg + "_req_f", seg + "_conv_f", seg + "_conv_r_f"]
            self.tag_seg.extend(statis_seg)
        self.partLen = len(self.tag_seg)
        self.con_type_dict = {0: 0, 2: 1, 6: 2}
        self.delimiter = ","
        self.file_tail = ".csv"
        self.monitor_interval = 1000000
        self.trans_maxcnt()
        print u"=========初始化统计信息导入完毕，开始筛选数据==========="
        self.trans_txt(data_dir)

    ##将字段取值根据设置的编码长度进行转换
    def trans_seg(self, data, loc, seg):
        encode_len = self.tag_seg_encode_len[loc]
        try:
            data = float(data)
        except:
            data = bool(data)

        if encode_len == 1: #1为编码， >0 则为1
            if data > 0:
                return [1], [1]
            return [0], [1]
        elif type(encode_len) == int: ##向下取整型编码
            encoding = [0] * encode_len
            mask = [0] * encode_len
            data_loc = int(data)
            encoding[data_loc] = 1
            return encoding, mask
        elif encode_len == 's': ##需要归一化的字段
            upb = self.scaling[seg]["max"]
            lowb = self.scaling[seg]["min"]
            miu = (upb + lowb) / 2.0
            scal_res = round((data - miu) / (upb - lowb) * 2 * self.float_scal, self.roundf)
            return [scal_res ], [1]
        elif encode_len == 'c':#标记为特判处理的字段
            if seg == 'h':
                data /= 2
                encoding = [0] * 12
                mask = [0] * 12
                data_loc = int(data)
                encoding[data_loc] = 1
                return encoding, mask
            if seg == "con_type":
                encoding = [0] * (len(self.con_type_dict.keys()) + 1)
                mask = [0] * (len(self.con_type_dict.keys()) + 1)
                try:
                    encoding[self.con_type_dict[int(data)]] = 1
                except:
                    encoding[len(self.con_type_dict.keys())] = 1
                return encoding, mask
            if seg.find("rank") != -1: ##排名转化为 [0, 1]内的函数值
                value = round(self.rank_value(data) * self.float_scal,  self.roundf)
                return [value], [1]
        elif encode_len == 'm':
            return [], []

    ###预处理各项统计字段的长度，确定位数
    def trans_maxcnt(self):
        for seg in self.prev_seg:
            seg_max = self.maxcnt_dict[seg]
            for i in range(len(seg_max)):
                if seg_max[i] > 1:##数量级拆分
                    #print "seg = %s, value = %d" % (seg, seg_max[i])
                    self.maxcnt_dict[seg][i] = self.digit_apart(seg_max[i], 0) + 1
                    #print "trans as digits %d" % self.maxcnt_dict[seg][i]

    def init_addition_data(self):
        self.addition_data["missingAppRank"] = set()
        self.addition_data["missingApp"] = set()
        self.maxcnt = {}
        for seg in self.prev_seg:
            self.maxcnt[seg] = [0, 0, 0]

    ###统计每个字段的取值个数，有没有NULL,空这样的特例，每种取值的样本数
    def trans_row(self, data, row_cnt, fw, fw_csv, trans_stack=False):
        #print data
        label = data["convert"]
        if not trans_stack:
            if label == "1":
                self.cnt_true += 1
                if not self.testFile:
                    self.TrueStack.append(data)
                    return True
            else:
                self.cnt_false += 1
                if not self.testFile:
                    if self.cnt_false < self.assume * self.rate:
                        self.FalseStack.append(data)
                    return True
        else:
            if label == "1":
                self.cnt_true += 1
            else:
                #print self.cnt_false, self.assume * self.rate, self.cnt_false >= self.assume * self.rate
                if self.cnt_false >= self.assume * self.rate:
                    return False
                else:
                    self.cnt_false += 1

        data_line = []
        mask = []
        encode_seg_len = len(self.tag_seg_encode_len)
        ##转换编码字段
        for idx in range(encode_seg_len):
            seg = self.tag_seg[idx]
            try:
                trans_res, trans_mask = self.trans_seg(data[seg], idx, seg)
                data_line.extend(trans_res)
                mask.extend(trans_mask)
            except:
                print "seg = %s, value = %s" % (seg, data[seg])
            #print "seg = %s, seglen = %d, value = %s" % (seg, len(trans_res), data[seg])

        ##转换统计字段
        for seg in self.prev_seg:
            trans_res = []
            statis_seg = [seg + "_req", seg + "_conv", seg + "_conv_r", seg + "_req_f", seg + "_conv_f", seg + "_conv_r_f"]
            if seg in self.prev_mask:
                continue
            ###编码处理的
            for i in []:
                part_res = [0] * self.maxcnt_dict[seg][i]  ###根据最大值的encoding分配长度
                value_idx = self.digit_apart(float(data[statis_seg[i]]), 0)
                #print "seg = %s, maxl = %d, valueidx= %d" % (seg, self.maxcnt_dict[seg][i], value_idx)
                part_res[value_idx] = 1
                mask.extend(part_res)
                trans_res.extend(part_res)

            ##直接放置的
            for i in [5]:
                trans_res.append(round(float(data[statis_seg[i]]) * self.float_scal, self.roundf))
                mask.append(1)
            data_line.extend(trans_res)
            #print "seg = %s, seglen = %d" % (seg, len(trans_res))

        #print data_line
        self.writeline(data_line, mask, fw, fw_csv)
        return True

    def writeline(self, data_line, mask, fw, fw_csv):
        out_s = str(data_line[0])
        fw_csv.writerow(data_line)
        for idx in range(1, len(data_line)):
            if mask[idx] == 0 and data_line[idx] == 0:
                continue
            out_s += " " + str(idx - 1) + ":" + str(data_line[idx])
        fw.write(out_s + "\n")

    def rank_value(self, x):
        return self.sigmoid((100 - x) / 50.0) / self.sigmoid(2.0)

if __name__ == "__main__":
    convert_dir = "./combine/expAndnot/"
    tag_click_convert(convert_dir)