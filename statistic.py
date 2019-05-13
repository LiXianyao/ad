#-*-encoding:utf-8-*-#

import sys
import csv
import datetime as dt
#import pandas as pd
reload(sys)
sys.setdefaultencoding("utf-8")
import json

class statisticsData(object):
    tag_seg = []
    statistics_data = {}
    statistics_data_set = {}
    statistics_title = {}
    delimiter = "\01"
    dir = ""
    file = ""
    file_tail = ".txt"
    null_to_zero = []
    null_to_empty = []
    monitor_interval = 100000000
    saving_tail = "_statistics.csv"


    def __init__(self):
        self.partLen = len(self.tag_seg)

    def statistics_txt(self, filename):
        ###不同平台的app、不同分类的app， 不同平台不同分类的app分别由多少
        fp = open(filename, "rU")
        with open(filename.replace(self.file_tail, self.saving_tail).replace(self.dir, "statistics/"), "ab") as temp:
            writer = csv.writer(temp)
            row_cnt = 0
            statistics_cnt = 0
            self.init_statistics_data()
            for buf in fp:
                row_cnt += 1
                buf = buf.strip().lower()
                partition = buf.split(self.delimiter)
                if len(partition) != self.partLen:
                    # print u"第%d行数据字段异常, content = %s"%(row_cnt, buf)
                    continue
                data = dict(zip(self.tag_seg, partition))
                self.trans_null(data)
                is_need = self.statistics_row(data, row_cnt)
                if is_need == True:
                    statistics_cnt += 1
                if row_cnt % self.monitor_interval== 0:
                    print "row_cnt = %d" % (row_cnt / self.monitor_interval)
                    #break
            print "proceed end, total row as %d, target row as %d" % (row_cnt, statistics_cnt)
            #self.save_statistics_data()
            self.write_res(writer)
        fp.close()

    def init_statistics_data(self):
        pass

    def statistics_row(self, data, row_cnt):
        return False

    def trans_null(self, data_seg):
        for seg in self.null_to_zero:
            if data_seg[seg] == "null":
                data_seg[seg] = "0"
        for seg in self.null_to_empty:
            if data_seg[seg] == "null":
                data_seg[seg] = ""

    ####将额外信息的字段分文件保存下来
    def save_statistics_data(self):
        pass

    def write_res(self, writer):
        global statistics_data
        output_order = ["tsegcnt"]
        output_order.extend(self.tag_seg)
        for key in output_order:
            writer.writerow(["table %s" % key])
            writer.writerow(self.statistics_title[key])
            self.output_dict(self.statistics_data[key], writer, 0, [])
            writer.writerow([])
            # for column in columns:

    def output_dict(self, data, writer, depth, prev):
        if type(data) != dict:
            if depth == 0:
                writer.writerow([data])
            elif type(data) == list:
                prev.extend(data)
                writer.writerow(prev)
            else:
                prev.append(data)
                writer.writerow(prev)
            return
        for key in data.keys():
            nextdata = prev[:]
            nextdata.append(key)
            self.output_dict(data[key], writer, depth + 1, nextdata)

class tag_click_convert(statisticsData):

    tag_seg = ["click_id", "source", "country", "sdk_version", "make", "brand", "model", "request_id", "bid_time",
               "advertiser_network_id", "advertiser_id", "campaign_id", "exchange_id", "app_platform", "app_id",
               "inner_subid", "position_id", "device_id", "os_version", "carrier", "ts", "media_bundle", "con_type",
               "revenue", "install_conversion", "event_conversion", "ymd", "h", "r"]

    keep_seg = ["country", "brand", "model", "advertiser_network_id", "advertiser_id", "app_platform", "app_id",
                "device_id", "os_version", "media_bundle", "con_type", "install_conversion", "event_conversion", "ymd",
                "h"]

    dir = "data/"
    file = "tag_click_convert.txt"
    check_seg_upb = 100
    count_seg_upb = 1000

    def __init__(self):
        super(tag_click_convert, self).__init__()
        self.null_to_zero = ["advertiser_network_id", "advertiser_id", "app_platform", "con_type","install_conversion", "event_conversion"]
        self.null_to_empty = ["brand", "model", "app_id", "os_version", "media_bundle"]
        self.statistics_txt(self.dir + self.file)

    def init_statistics_data(self):
        # 每个字段一个自己的set和dict,dict里面存有限个（100以内）个具体值和NULL和空串，以及对应的数量
        # set存每个字段的取值个数（超过1000则不再计算）,在多一个tsegcnt存每个set的size
        for seg in self.tag_seg:
            self.statistics_data[seg] = {}
            self.statistics_data_set[seg] = set()
            self.statistics_title[seg] = [u"取值", u"计数", u"转化数"]

    ###统计每个字段的取值个数，有没有NULL,空这样的特例，每种取值的样本数
    def statistics_row(self, data, row_cnt):
        # 每个字段一个自己的set和dict,dict里面存有限个（100以内）个具体值和NULL和空串，以及对应的数量
        # set存每个字段的取值个数（超过1000则不再计算）,在多一个tsegcnt存每个set的size
        convert_cnt = int(data["install_conversion"]) + int(data["event_conversion"])
        convert_cnt = int(bool(convert_cnt)) ##相加>1就是转化

        for seg in data.keys():
            value = data[seg]
            if value in self.statistics_data[seg]:  ##这个字段这个取值有记录，直接计数
                self.statistics_data[seg][value][0] += 1
                self.statistics_data[seg][value][1] += convert_cnt
            elif len(self.statistics_data[seg]) < self.check_seg_upb:
                self.statistics_data[seg][value] = [1, convert_cnt]
                self.statistics_data_set[seg].add(value)
            elif value == "" or value == 'NULL':
                self.statistics_data[seg][value] = [1, convert_cnt]
                self.statistics_data_set[seg].add(value)
            elif (value not in self.statistics_data_set) and (len(self.statistics_data_set[seg]) < self.count_seg_upb):
                self.statistics_data_set[seg].add(value)

    def save_statistics_data(self):
        self.statistics_title["tsegcnt"] = [u"字段名", u"不同取值数（超过1000不计）"]
        self.statistics_data["tsegcnt"] = {}
        for seg in self.statistics_data_set.keys():
            self.statistics_data["tsegcnt"][seg] = len(self.statistics_data_set[seg])


class tag_app(statisticsData):
    tag_seg = ["plat", "appid", "appbundlecategory", "category_id", "rating_value", "price", "file_size"]
    dir = "data/"
    file = "tag_app.txt"

    def __init__(self):
        super(tag_app, self).__init__()
        self.null_to_empty = ["appid", "appbundlecategory"]
        self.null_to_zero = ["plat", "rating_value", "price", "file_size"]
        self.statistics_txt(self.dir + self.file)

    def init_statistics_data(self):
        ##tag_app表转json， 舍去
        self.statistics_data["platform"] = {'0': 0, '1': 0, 'NULL': 0}
        self.statistics_title["platform"] = [u"平台", u"app总数"]
        self.statistics_data["category"] = {}
        self.statistics_title["category"] = [u"app分类id", u"app分类", u"app计数"]
        self.statistics_data["platcategory"] = {'0': {}, '1': {}, 'NULL': {}}
        self.statistics_title["platcategory"] = [u"平台", u"app分类id", u"app分类", u"app计数"]

        for seg in self.tag_seg:
            self.statistics_data[seg] = {}
            self.statistics_data_set[seg] = set()
            self.statistics_title[seg] = [u"取值", u"计数", u"转化数"]

    def statistics_row(self, data, row_cnt):
        plat, app_id, bundle_category, category_id, rating_value, price, file_size\
            = data["plat"], data["appid"], data["appbundlecategory"], data["category_id"], \
              data["rating_value"], data["price"], data["file_size"]

        self.statistics_data["platform"][plat] += 1
        bundle_category = bundle_category.encode("gbk").decode("gbk")

        try:
            self.statistics_data["category"][category_id][bundle_category] += 1
        except:
            if category_id not in self.statistics_data["category"].keys():
                self.statistics_data["category"][category_id] = {}
            self.statistics_data["category"][category_id][bundle_category] = 1  # [1, row_cnt]

        try:
            self.statistics_data["platcategory"][plat][category_id][bundle_category] += 1
        except:
            if category_id not in self.statistics_data["platcategory"][plat].keys():
                self.statistics_data["platcategory"][plat][category_id] = {}
            self.statistics_data["platcategory"][plat][category_id][bundle_category] = 1

        for seg in data.keys():
            value = data[seg]
            if value in self.statistics_data[seg]:  ##这个字段这个取值有记录，直接计数
                self.statistics_data[seg][value] += 1
            else:
                self.statistics_data[seg][value] = 1
                self.statistics_data_set[seg].add(value)

    def save_statistics_data(self):
        self.statistics_title["tsegcnt"] = [u"字段名", u"不同取值数"]
        self.statistics_data["tsegcnt"] = {}
        for seg in self.statistics_data_set.keys():
            self.statistics_data["tsegcnt"][seg] = len(self.statistics_data_set[seg])

class country_tag_click_convert(tag_click_convert):
    tag_seg = ["country", "brand", "model", "advertiser_network_id", "advertiser_id", "app_platform", "app_id",
                "device_id", "os_version", "media_bundle", "con_type", "install_conversion", "event_conversion", "ymd",
                "h"]
    dir = "apart/"
    file = "tag_click_convert_apart.csv"

    def __init__(self):
        self.check_seg_upb = 1000
        self.count_seg_upb = 100000
        self.delimiter = ","
        self.file_tail = ".csv"
        super(country_tag_click_convert, self).__init__()

class country_tag_click_convert_prev5(tag_click_convert):
    ##统计前5天的数据情况
    tag_seg = ["country", "brand", "model", "advertiser_network_id", "advertiser_id", "app_platform", "app_id",
                "device_id", "os_version", "media_bundle", "con_type", "install_conversion", "event_conversion", "ymd",
                "h"]
    dir = "apart/"
    file = "tag_click_convert_apart.csv"
    prev_seg = ["brand", "model", "advertiser_network_id", "advertiser_id", "app_id",
                "os_version", "media_bundle"]
    day_set = set()
    device_dict = {}    ##要打印
    device_info = {}    ##要打印
    app_info = {}       ##要打印
    contype_dict = {} ##要打印
    prev5_log = {}      ##要打印
    device_cnt = 0      ##要打印
    total_convert = 0   ##要打印
    total_request = 0   ##要打印
    country = "SAU"
    alpha = 0.4###指数平滑参数

    def __init__(self):
        self.delimiter = ","
        self.file_tail = ".csv"
        self.monitor_interval = 1000000
        self.saving_tail = "_prev5.csv"
        super(country_tag_click_convert_prev5, self).__init__()

    def save_appid(self, appid, ad_net_id, ad_id):
        try:
            self.app_info[appid]
        except:
            self.app_info[appid] = {}

        try:
            self.app_info[appid][ad_net_id]
        except:
            self.app_info[appid][ad_net_id] = {}

        try:
            self.app_info[appid][ad_net_id][ad_id] += 1
        except:
            self.app_info[appid][ad_net_id][ad_id] = 1

    def save_contype(self, contype):
        ##记录device_id在每天的点击/转换情况
        try:
            contype_loc = self.contype_dict[contype]
        except:
            contype_loc = len(self.contype_dict)
            self.contype_dict[contype] = contype_loc

    def count_device_id(self, device_id, day, data):
        ##记录device_id在每天的点击/转换情况
        try:
            device_loc = self.device_dict[device_id]
        except:
            self.device_dict[device_id] = self.device_cnt
            device_loc = self.device_cnt
            self.device_cnt += 1
            self.device_info[device_loc] = {
                "pf": data["app_platform"],
                "br": data["brand"],
                "md": data["model"],
                "os": data["os_version"]
            }

        self.statistics_data["device_id"][day]["lastday"].add(device_loc)

    def add_device_convert(self, device_id, day, convert, app_id, media_bundle):
        device_loc = self.device_dict[device_id]
        try:
            self.statistics_data["device_id"][day][device_loc][1][media_bundle] = \
                self.statistics_data["device_id"][day][device_loc][1].setdefault(media_bundle, 0) + 1
        except:
            self.statistics_data["device_id"][day][device_loc] = [set(), {}] ### 0位是appid的集合，1位是mb的dict
            self.statistics_data["device_id"][day][device_loc][1][media_bundle] = 1
        if convert == 0:
            return
        self.statistics_data["device_id"][day][device_loc][0].add(app_id)

    def init_statistics_data(self):
        # 每个字段一个自己的set和dict,dict里面存有限个（100以内）个具体值和NULL和空串，以及对应的数量
        # set存每个字段的取值个数（超过1000则不再计算）,在多一个tsegcnt存每个set的size
        for seg in self.prev_seg:
            self.statistics_data[seg] = {}
            self.statistics_title[seg] = [u"取值", u"日期", u"请求数", u"转化数", u"转化率"]
        self.statistics_data["device_id"] = {}

    def statistics_row(self, data, row_cnt):
        # 每个字段一个自己的set和dict,dict里面存有限个（100以内）个具体值和NULL和空串，以及对应的数量
        # set存每个字段的取值个数（超过1000则不再计算）,在多一个tsegcnt存每个set的size
        convert_cnt = int(data["install_conversion"]) + int(data["event_conversion"])
        convert_cnt = int(bool(convert_cnt))  ##相加>1就是转化
        self.total_convert += convert_cnt
        self.total_request += 1

        daytime = dt.datetime.strptime(data["ymd"], "%Y%m%d")
        if daytime not in self.day_set:
            self.day_set.add(daytime)
            self.statistics_data["device_id"][daytime] = {"lastday": set()}
            for seg in self.prev_seg:
                self.statistics_data[seg][daytime] = {}

        #对统计特征按天分别进行计数
        for seg in self.prev_seg:
            value = data[seg]
            try:  ##这个字段这个取值有记录，直接计数
                self.statistics_data[seg][daytime][value][0] += 1
                self.statistics_data[seg][daytime][value][1] += convert_cnt
            except:
                self.statistics_data[seg][daytime][value] = [1, convert_cnt]

        self.save_contype(data["con_type"])
        self.save_appid(data["app_id"], data["advertiser_network_id"], data["advertiser_id"])
        self.count_device_id(data["device_id"], daytime, data)
        self.add_device_convert(data["device_id"], daytime, convert_cnt, data["app_id"], data["media_bundle"])

        return True

    def save_statistics_data(self):
        ##主要整理device_id,将每天的结果移交到其后总计五天的内容里
        keep_days = list(self.day_set)
        keep_days.sort(reverse=True)
        keep_days = keep_days[:2]
        for day in keep_days:
            day_s = dt.datetime.strftime(day, "%Y%m%d")
            self.prev5_log[day_s] = {}

        exist_day = list(self.day_set)
        exist_day.sort()
        self.save_prev5_log(exist_day, keep_days)

        print "save statistics data"
        for day in exist_day:   ##处理其他统计字段
            for seg in self.prev_seg:
                #print day, seg
                for idx in self.statistics_data[seg][day]:
                    for delta in range(1, 6):
                        save_day = day + dt.timedelta(days=delta)
                        if save_day not in keep_days:
                            continue
                        save_day_s = dt.datetime.strftime(save_day, "%Y%m%d")
                        day_s = dt.datetime.strftime(day, "%Y%m%d")
                        try:
                            lastday = self.statistics_data[seg][idx][save_day_s][2]
                            thisday = (save_day - day).days
                            self.statistics_data[seg][idx][save_day_s][0] += self.statistics_data[seg][day][idx][0]  ##请求数累加
                            self.statistics_data[seg][idx][save_day_s][0] = round(self.statistics_data[seg][idx][save_day_s][0], 5)
                            self.statistics_data[seg][idx][save_day_s][1] += self.statistics_data[seg][day][idx][1]  ##转化数累加
                            self.statistics_data[seg][idx][save_day_s][1] = round(self.statistics_data[seg][idx][save_day_s][1], 5)
                            self.statistics_data[seg][idx][save_day_s][2] = thisday

                            self.statistics_data[seg][idx][save_day_s][3] += self.featurePow(self.statistics_data[seg][day][idx][0], thisday)##请求数累加
                            self.statistics_data[seg][idx][save_day_s][3] = round(self.statistics_data[seg][idx][save_day_s][3], 5)
                            self.statistics_data[seg][idx][save_day_s][4] += self.featurePow(self.statistics_data[seg][day][idx][1], thisday)##转化数累加
                            self.statistics_data[seg][idx][save_day_s][4] = round(self.statistics_data[seg][idx][save_day_s][4], 5)
                        except:
                            try:
                                self.statistics_data[seg][idx]
                            except:
                                self.statistics_data[seg][idx] = {}
                            req_cnt, conv_cnt = self.statistics_data[seg][day][idx][:]
                            segday = (save_day - day).days  ###天数差
                            self.statistics_data[seg][idx][save_day_s] = [req_cnt, conv_cnt, segday, self.featurePow(req_cnt, segday), self.featurePow(conv_cnt, segday)]
                        #    if idx == "apple":
                        #        import  traceback
                        #        print traceback.format_exc()
                        #if idx == "apple":
                        #   print save_day_s, day_s, self.statistics_data[seg][day][idx], self.statistics_data[seg][idx][save_day_s]
                #print self.statistics_data[seg]
                self.statistics_data[seg][day].clear()
                del self.statistics_data[seg][day]

        self.device_dict["device_cnt"] = self.device_cnt
        self.statistics_data["total_request"] = self.total_request
        self.statistics_data["total_convert"] = self.total_convert

    def save_prev5_log(self, exist_day, keep_days):
        cnt_convert = 0
        print "save prev5 log"
        for day in exist_day:  ##处理device
            for device_idx in self.statistics_data["device_id"][day]["lastday"]:
                try:
                    device_convert = self.statistics_data["device_id"][day][device_idx][0]
                except:
                    device_convert = set()
                device_mb = self.statistics_data["device_id"][day][device_idx][1]
                for delta in range(0, 6):
                    save_day = day + dt.timedelta(days=delta)
                    if save_day not in keep_days:
                        continue
                    save_day_s = dt.datetime.strftime(save_day, "%Y%m%d")

                    if delta == 0:
                        try:
                            self.prev5_log[save_day_s][device_idx]
                        except:
                            self.prev5_log[save_day_s][device_idx] = [7, 0, [], {}]  ##5天以内没有登陆过，活跃天为0，转化集为空
                        continue
                    try:
                        lastday = self.prev5_log[save_day_s][device_idx][0]
                        thisday = (save_day - day).days
                        self.prev5_log[save_day_s][device_idx][0] = thisday
                        self.prev5_log[save_day_s][device_idx][1] += 1
                        cnt_convert -= len(self.prev5_log[save_day_s][device_idx][2])
                        self.prev5_log[save_day_s][device_idx][2].union(device_convert)
                        cnt_convert += len(self.prev5_log[save_day_s][device_idx][2])
                        self.prev5_log[save_day_s][device_idx][3] = self.addTwoDict(self.prev5_log[save_day_s][device_idx][3], device_mb, lastday - thisday)
                        #print device_idx, self.prev5_log[save_day_s][device_idx][3]
                        #exit(0)
                    except:
                        self.prev5_log[save_day_s][device_idx] = [(save_day - day).days, 1, device_convert, device_mb]
                        cnt_convert += len(self.prev5_log[save_day_s][device_idx][2])
            self.statistics_data["device_id"][day].clear()
            del self.statistics_data["device_id"][day]
        # print "convertcnt = %d, total = %d"%(cnt_convert, self.total_convert)
        for save_day in keep_days:  ###set不可以序列化，转list
            save_day_s = dt.datetime.strftime(save_day, "%Y%m%d")
            for device_idx in self.prev5_log[save_day_s]:
                self.prev5_log[save_day_s][device_idx][2] = list(self.prev5_log[save_day_s][device_idx][2])
                for mbkey in self.prev5_log[save_day_s][device_idx][3].keys():
                    lastday = self.prev5_log[save_day_s][device_idx][0] - 1
                    self.prev5_log[save_day_s][device_idx][3][mbkey] *= (self.alpha ** lastday)
                    self.prev5_log[save_day_s][device_idx][3][mbkey] = round(self.prev5_log[save_day_s][device_idx][3][mbkey], 5)
        del self.statistics_data["device_id"]

    def write_res(self, writer):
        global statistics_data
        """
        output_order = self.prev_seg
        for key in output_order:
            writer.writerow(["table %s" % key])
            writer.writerow(self.statistics_title[key])
            self.output_dict(self.statistics_data[key], writer, 0, [])
            writer.writerow([])
        """

        ##统计信息保存为json字符串
        to_json_dict = {"statistics_data": self.statistics_data, "device_dict": self.device_dict, "prev5days_log": self.prev5_log}
        to_json_dict = {"device_info": self.device_info, "app_info": self.app_info,
                        "contype_dict": self.contype_dict}
        for diction_name in to_json_dict.keys():
            diction = to_json_dict[diction_name]
            print diction_name
            with open("statistics/" + diction_name + "_" + self.country + ".txt", "w") as addition_record:
                try:
                    output_json = json.dumps(diction)
                except:
                    print diction
                addition_record.write(output_json)

    def output_dict(self, data, writer, depth, prev):
        if type(data) != dict:
            if depth == 0:
                writer.writerow([data])
            elif type(data) == list:
                if len(data) == 5:
                    rate = round(float(data[1]) / data[0], 6)
                    data[2] = rate
                    rate_alpha = round(float(data[4]) / data[3], 6)
                    data.append(rate_alpha)
                prev.extend(data)
                writer.writerow(prev)
            else:
                prev.append(data)
                writer.writerow(prev)
            return
        for key in data.keys():
            nextdata = prev[:]
            nextdata.append(key)
            self.output_dict(data[key], writer, depth + 1, nextdata)

    def featurePow(self, data, days):
        return self.alpha * data + data * (1-self.alpha) ** (days - 1)

    def addTwoDict(self, dict1, dict2, daydelta):
        ## d1 should be the old value
        d1_key = set(dict1.keys())
        d2_key = set(dict2.keys())
        keyset = d1_key.union(d2_key)
        #print keyset
        res_dict = {}
        for key in keyset:
            d1_value = float(dict1.setdefault(key, -1))
            d2_value = float(dict2.setdefault(key, 0))
            if d1_value == -1:
                res_dict[key] = d2_value
            else:
                res_dict[key] = self.alpha * d2_value + (1 - self.alpha) ** daydelta * d1_value
        #print res_dict
        return res_dict

if __name__ == "__main__":
    #country_tag_click_convert()
    #tag_app()
    country_tag_click_convert_prev5()