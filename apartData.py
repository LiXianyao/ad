#-*-encoding:utf-8-*-#

import sys
import csv
#import pandas as pd
reload(sys)
sys.setdefaultencoding("utf-8")

class apartData(object):
    tag_seg = []
    addition_data = {}
    apart_data_set = {}
    apart_title = {}
    null_to_zero = []
    null_to_empty = []
    dir = ""
    file = ""
    file_tail = ".txt"
    delimiter = "\01"

    def __init__(self):
        self.partLen = len(self.tag_seg)

    def apart_txt(self, filename):
        ###不同平台的app、不同分类的app， 不同平台不同分类的app分别由多少
        fp = open(filename, "rU")
        with open(filename.replace(self.file_tail, "_apart.csv").replace(self.dir, "apart/"), "wb") as temp:
            writer = csv.writer(temp)
            row_cnt = 0
            apart_cnt = 0
            self.init_addition_data()
            for buf in fp:
                row_cnt += 1
                buf = buf.strip().lower()
                partition = buf.split(self.delimiter)
                if len(partition) != self.partLen:
                    # print u"第%d行数据字段异常, content = %s"%(row_cnt, buf)
                    continue
                data = dict(zip(self.tag_seg, partition))
                self.trans_null(data)
                is_need = self.apart_row(data, row_cnt, writer)
                if is_need == True:
                    apart_cnt += 1
                if row_cnt % 10000000 == 0:
                    print "row_cnt = %d" % (row_cnt / 10000000)
            print "proceed end, total row as %d, target row as %d" % (row_cnt, apart_cnt)
            self.save_addition_data()
            # write_res(filename, writer)
        fp.close()

    def init_addition_data(self):
        pass

    def trans_null(self, data_seg):
        for seg in self.null_to_zero:
            if data_seg[seg] == "null":
                data_seg[seg] = "0"
        for seg in self.null_to_empty:
            if data_seg[seg] == "null":
                data_seg[seg] = ""

    def apart_row(self, partition, row_cnt, writer):
        return False

    ####将额外信息的字段分文件保存下来
    def save_addition_data(self):
        ##额外信息保存下来
        for seg in self.addition_data:
            print "addition data %s has %d elements" % (seg, len(self.addition_data[seg]))
            with open("apart/" + str(seg) + ".txt", "w") as addition_record:
                if type(self.addition_data[seg]) == list or type(self.addition_data[seg]) == set:  ##列表信息按行存储
                    writer = csv.writer(addition_record)
                    for element in self.addition_data[seg]:
                        out_list = [element]
                        writer.writerow(out_list)

                elif type(self.addition_data[seg]) == dict:  ##结构信息存json字符串
                    import json
                    output_json = json.dumps(self.addition_data[seg])
                    addition_record.write(output_json)

class tag_click_convert(apartData):

    tag_seg = ["click_id", "source", "country", "sdk_version", "make", "brand", "model", "request_id", "bid_time",
               "advertiser_network_id", "advertiser_id", "campaign_id", "exchange_id", "app_platform", "app_id",
               "inner_subid", "position_id", "device_id", "os_version", "carrier", "ts", "media_bundle", "con_type",
               "revenue", "install_conversion", "event_conversion", "ymd", "h", "r"]

    keep_seg = ["country", "brand", "model", "advertiser_network_id", "advertiser_id", "app_platform", "app_id",
                "device_id", "os_version", "media_bundle", "con_type", "install_conversion", "event_conversion", "ymd",
                "h"]

    dir = "data/"
    file = "tag_click_convert.txt"

    def __init__(self):
        super(tag_click_convert, self).__init__()
        self.null_to_zero = ["advertiser_network_id", "advertiser_id", "app_platform", "con_type","install_conversion", "event_conversion"]
        self.null_to_empty = ["brand", "model", "app_id", "os_version", "media_bundle"]
        self.apart_txt(self.dir + self.file)

    def init_addition_data(self):
        ##"需要额外记录SAU国家的沙特的用户的device_id"
        self.addition_data["device_id"] = set()

    ###统计每个字段的取值个数，有没有NULL,空这样的特例，每种取值的样本数
    def apart_row(self, data, row_cnt, writer):
        # 分析行内容，并抽取出沙特国家(编号SAU）的行
        if data["country"] == "sau":
            self.addition_data["device_id"].add(data["device_id"])
            out_list = []
            for seg in self.keep_seg:
                out_list.append(data[seg].strip().replace(",", "."))
            writer.writerow(out_list)
            return True
        return False


class tag_app(apartData):
    tag_seg = ["plat", "appid", "appbundlecategory", "category_id", "rating_value", "price", "file_size"]
    dir = "data/"
    file = "tag_app.txt"

    def __init__(self):
        super(tag_app, self).__init__()
        self.null_to_empty = ["appid", "appbundlecategory"]
        self.null_to_zero = ["plat", "rating_value", "price", "file_size"]
        self.apart_txt(self.dir + self.file)

    def init_addition_data(self):
        self.addition_data["tag_app"] = {}
        self.addition_data["appbundle"] = {}
        self.bundle_size = 0

    def apart_row(self, data, row_cnt, writer):
        plat, app_id = data["plat"], data["appid"]
        if plat not in self.addition_data["tag_app"]:
            self.addition_data["tag_app"][plat] = {}
        try:
            bundle = self.addition_data["appbundle"][data["appbundlecategory"]]
        except:
            self.addition_data["appbundle"][data["appbundlecategory"]] = self.bundle_size
            bundle = self.bundle_size
            self.bundle_size += 1

        self.addition_data["tag_app"][plat][app_id] = {
            "bund": bundle,
            "rate": float(data["rating_value"]),
            "price": float(data["price"]),
            "size": round(float(data["file_size"])/(1024*1024), 2) ##转MB， 保留两位小数
        }
        return False

class tag_app_rank(apartData):
    tag_seg = ["app_id", "country", "rank"]
    dir = "data/"
    file = "tag_app_rank.txt"

    def __init__(self):
        super(tag_app_rank, self).__init__()
        self.null_to_empty = ["app_id", "country"]
        self.null_to_zero = ["rank"]
        self.apart_txt(self.dir + self.file)

    def init_addition_data(self):
        ##tag_app_rank表转json， 舍去
        self.addition_data["tag_app_rank"] = {}

    def apart_row(self, data, row_cnt, writer):
        country, app_id= data["country"], data["app_id"]
        if country not in self.addition_data["tag_app_rank"]:
            self.addition_data["tag_app_rank"][country] = {}
        self.addition_data["tag_app_rank"][country][app_id] = int(data["rank"])
        return False

if __name__ == "__main__":

    #tag_click_convert()
    tag_app()
    #tag_app_rank()