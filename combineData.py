#-*-encoding:utf-8-*-#

import sys
import csv
#import pandas as pd
reload(sys)
sys.setdefaultencoding( "utf-8" )
##

class combineData(object):
    tag_seg = []
    addition_data = {}
    combine_data_set = {}
    combine_title = {}
    null_to_zero = []
    null_to_empty = []
    tag_app_file = "apart/tag_app.txt"
    app_bundle_file = "apart/appbundle.txt"
    tag_app_rank_file = "apart/tag_app_rank.txt"
    device_dict_file = "statistics/device_dict_SAU.txt"
    prev5_log_file = "statistics/prev5days_log_SAU.txt"
    statistics_data_file = "statistics/statistics_data_SAU.txt"
    saving_tail = "_combine.csv"
    saving_dir = "combine/expAndnot/"
    dir = ""
    file = ""
    file_tail = ".txt"
    delimiter = "\01"
    monitor_interval = 10000000

    def __init__(self):
        self.partLen = len(self.tag_seg)
        self.tag_app = self.get_json(self.tag_app_file)
        self.app_bundle = self.get_json(self.app_bundle_file)
        self.app_bundle_size = len(self.app_bundle)
        self.tag_app_rank = self.get_json(self.tag_app_rank_file)
        self.statistics_data = self.get_json(self.statistics_data_file)
        self.device_dict = self.get_json(self.device_dict_file)
        self.prev5_log = self.get_json(self.prev5_log_file)
        self.save_days = self.prev5_log.keys()#最近五天使用情况中的可用日期就是训练集和测试集的日期
        print "save days = ",self.save_days

    def combine_txt(self, filename):
        ###不同平台的app、不同分类的app， 不同平台不同分类的app分别由多少
        fp = open(filename, "rU")
        fws = {}
        writers = {}
        for day in self.save_days:
            fws[day] = open(filename.replace(self.file_tail, day + self.saving_tail).replace(self.dir, self.saving_dir), "wb")
            writers[day] = csv.writer(fws[day])
            writers[day].writerow(self.res_seg)

        row_cnt = 0
        combine_cnt = 0
        self.init_addition_data()
        for buf in fp:
            row_cnt += 1
            buf = buf.strip().lower()
            partition = buf.split(self.delimiter)
            if len(partition) != self.partLen:
                print u"第%d行数据字段异常, content = %s"%(row_cnt, buf)
                continue
            data = dict(zip(self.tag_seg, partition))
            #self.trans_null(data)
            is_need = self.combine_row(data, row_cnt, writers)
            if is_need == True:
                combine_cnt += 1
            if row_cnt % self.monitor_interval == 0:
                print "row_cnt = %d" % (row_cnt / self.monitor_interval)
                #break

        print "proceed end, total row as %d, target row as %d" % (row_cnt, combine_cnt)
        self.save_addition_data()
        fp.close()
        for fw in fws.values():
            fw.close()

    def get_json(self, file_path):
        with open(file_path, "rb") as df:
            json_str = df.readline()
            import json
            data = json.loads(json_str)
        print len(data)
        return data

    def init_addition_data(self):
        pass

    def combine_row(self, partition, row_cnt, writer):
        return False

    ####将额外信息的字段分文件保存下来
    def save_addition_data(self):
        ##额外信息保存下来
        for seg in self.addition_data:
            print "addition data %s has %d elements" % (seg, len(self.addition_data[seg]))
            with open(self.saving_dir + str(seg) + ".txt", "w") as addition_record:
                if type(self.addition_data[seg]) == list or type(self.addition_data[seg]) == set:  ##列表信息按行存储
                    writer = csv.writer(addition_record)
                    for element in self.addition_data[seg]:
                        out_list = [element]
                        writer.writerow(out_list)

                elif type(self.addition_data[seg]) == dict:  ##结构信息存json字符串
                    import json
                    output_json = json.dumps(self.addition_data[seg])
                    addition_record.write(output_json)

class tag_click_convert(combineData):

    tag_seg = ["country", "brand", "model", "advertiser_network_id", "advertiser_id", "app_platform", "app_id",
                "device_id", "os_version", "media_bundle", "con_type", "install_conversion", "event_conversion", "ymd",
                "h"]
    prev_seg = ["brand", "model", "advertiser_network_id", "advertiser_id", "app_id",
                "os_version", "media_bundle"]
    fix_seg = ["app_platform", "con_type", "h"]#, "device_id"]
    dir = "apart/"
    file = "tag_click_convert_apart.csv"

    def __init__(self, isSuper=False):
        self.delimiter = ","
        self.file_tail = ".csv"
        super(tag_click_convert, self).__init__()
        self.null_to_zero = ["advertiser_network_id", "advertiser_id", "app_platform", "con_type","install_conversion", "event_conversion"]
        self.null_to_empty = ["brand", "model", "app_id", "os_version", "media_bundle"]
        self.monitor_interval = 1000000

        ###结果表头
        self.res_seg = ["convert", "app_plat", "con_type", "h", "app_rank", "app_rate", "app_price",
                   "app_bundle", "app_size",
                   "mb_rank", "mb_rate", "mb_price", "mb_bundle", "mb_size", "did_last", "did_cnt", "did_conv_app",
                   "did_conv_bundle"]
        for seg in self.prev_seg:
            ###每个统计指标有三个字段
            statis_seg = [seg + "_req", seg + "_conv", seg + "_conv_r", seg + "_req_f", seg + "_conv_f", seg + "_conv_r_f"]
            self.res_seg.extend(statis_seg)
        if isSuper == False:
            print u"=========初始化统计信息导入完毕，开始筛选数据==========="
            self.combine_txt(self.dir + self.file)

    def init_addition_data(self):
        self.addition_data["missingAppRank"] = set()
        self.addition_data["missingApp"] = set()
        self.addition_data["maxcnt"] = {}
        self.addition_data["scaling"] = {"app_rate": {"max": 0.0, "min": 10000000}, "app_size": {"max": 0.0, "min": 10000000}}
        for seg in self.prev_seg:
            self.addition_data["maxcnt"][seg] = [0, 0, 0, 0, 0, 0]

    ###统计每个字段的取值个数，有没有NULL,空这样的特例，每种取值的样本数
    def combine_row(self, data, row_cnt, writers):
        if data["ymd"] not in self.save_days: ##数据不在可用的日期内，跳过
            return False
        data_line = []
        ##label
        convert_cnt = self.get_convert_res(data)
        data_line.append(convert_cnt)

        ##编码数据：原样保留
        fix_data = self.get_fix_data(data)
        data_line.extend(fix_data)

        ##查询数据
        app_info = self.get_app_info(data["app_platform"], data["country"], data["app_id"])
        media_info = self.get_app_info(data["app_platform"], data["country"], data["media_bundle"])
        device_info = self.get_device_info(data["device_id"], data["app_platform"], data["app_id"], app_info[3], data["ymd"])
        data_line.extend(app_info)
        data_line.extend(media_info)
        data_line.extend(device_info)

        ###统计数据:将取值替换成该取值在某天的统计值
        statistic_info = self.get_statistics_info(data)
        data_line.extend(statistic_info)

        writers[data["ymd"]].writerow(data_line)
        return True

    ##根据字段信息计算确定此行数据是否为转化数据
    def get_convert_res(self, data):
        convert_cnt = int(data["install_conversion"]) + int(data["event_conversion"])
        convert_cnt = int(bool(convert_cnt))  ##相加>1就是转化\
        return convert_cnt

    ##根据国家和appid获取app的有关信息
    def get_app_info(self, plat, country, app_id):
        ##app排名数据
        if (country not in self.tag_app_rank) or \
                (app_id not in self.tag_app_rank[country]):  ##没有这个国家/app的排名数据
            rank = 500
            self.addition_data["missingAppRank"].add(app_id)
        else:
            rank = self.tag_app_rank[country][app_id]

        ##app信息：分类，评分，价格，大小
        if (plat not in self.tag_app) or \
                (app_id not in self.tag_app[plat]):  ##没有这个app的数据
            rate = self.setDefault("app_rate")
            price = 0.0
            bundle = self.app_bundle_size
            fszie = self.setDefault("app_size")
            self.addition_data["missingApp"].add(app_id)
        else:
            rate = self.tag_app[plat][app_id]["rate"]
            price = self.tag_app[plat][app_id]["price"]
            bundle = self.tag_app[plat][app_id]["bund"]
            fszie = self.tag_app[plat][app_id]["size"]

        self.addition_data["scaling"]["app_rate"]["max"] = max(self.addition_data["scaling"]["app_rate"]["max"], rate)
        self.addition_data["scaling"]["app_rate"]["min"] = min(self.addition_data["scaling"]["app_rate"]["min"], rate)
        self.addition_data["scaling"]["app_size"]["max"] = max(self.addition_data["scaling"]["app_size"]["max"], fszie)
        self.addition_data["scaling"]["app_size"]["min"] = min(self.addition_data["scaling"]["app_size"]["min"], fszie)
        return rank, rate, price, bundle, fszie

    def setDefault(self, seg):
        return (self.addition_data["scaling"][seg]["min"] + self.addition_data["scaling"][seg]["max"]) / 2

    def get_device_idx(self, device_id):
        return str(self.device_dict[device_id])

    ###根据device_id和日期获得某一天的几项统计数据：最后一次活跃时间（到当天的距离）、统计区间内的总活跃天数，统计区间内是否下载过此类app
    ## in：设备id、设备平台,这条广告请求app的类型, 时间ymd
    def get_device_info(self, device_id, plat, app, bundle, ymd):
        device_idx = self.get_device_idx(device_id)
        try:
            last = self.prev5_log[ymd][device_idx][0]#最后一次活跃时间（到当天的距离）
        except:
            print device_id, app, ymd
        cnt = self.prev5_log[ymd][device_idx][1]
        convert = self.prev5_log[ymd][device_idx][2]
        had_convert_app = 0
        had_convert_bundle = 0
        for app_id in convert:
            if app == app_id:
                had_convert_app = 1
            try:
                convert_bundle = self.tag_app[plat][app_id]
            except:
                print "plat = %s, appid = %s not in the dict!"%(plat, app_id)
                continue
            if convert_bundle == bundle:
                had_convert_bundle = 1
        return last, cnt, had_convert_app, had_convert_bundle

    ##编码数据：这些字段原样保留
    def get_fix_data(self, data):
        fix_data = []
        for seg in self.fix_seg:
            fix_data.append(data[seg])
        return fix_data

    ###统计数据:将取值替换成该取值在某天的统计值:最近5天的请求总数、转化总数、转化率
    def get_statistics_info(self, data):
        statistics_info = []
        ymd = data["ymd"]
        for seg in self.prev_seg:
            idx = data[seg]
            try:
                seg_statistics = self.statistics_data[seg][idx][ymd]
            except:
                seg_statistics = [0, 0, 0.0, 0, 0, 0.0]
            statistics_info.extend(seg_statistics)
            for i in range(len(self.addition_data["maxcnt"][seg])):
                self.addition_data["maxcnt"][seg][i] = max(self.addition_data["maxcnt"][seg][i], seg_statistics[i])
        return statistics_info

class generate_data(tag_click_convert):
    tag_seg = ["country", "brand", "model", "advertiser_network_id", "advertiser_id", "app_platform", "app_id",
               "device_id", "os_version", "media_bundle", "con_type", "ymd", "h"]
    prev_seg = ["brand", "model", "advertiser_network_id", "advertiser_id", "app_id",
                "os_version", "media_bundle"]
    fix_seg = ["app_platform", "con_type", "h"]#, "device_id"]
    dir = "apart/"

    def __init__(self, file, isSuper=False):
        super(generate_data, self).__init__(isSuper = True)
        self.file = file
        self.null_to_zero = ["advertiser_network_id", "advertiser_id", "app_platform", "con_type"]

        if super == False:
            print u"=========初始化统计信息导入完毕，开始筛选数据==========="
            self.combine_txt(self.file)

    def get_convert_res(self, data):
        ###测试数据没有实际label
        return 0

    def get_device_idx(self, device_id):
        return device_id


if __name__ == "__main__":
    tag_click_convert()