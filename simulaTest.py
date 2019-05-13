#-*-encoding:utf-8-*-#
import ConfigParser
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

"""
使用训练好的模型、所有已有的device信息和已投放过的app信息，对每个用户计算他对每个app的转化率
"""
class simulaTest(object):

    def __init__(self, configFile):
        self.configFile = configFile
        self.getConfig()
        self.loadJsonFiles()
        self.cnt_out = 0
        self.testStart()

    def testStart(self):
        ###每条数据本身包括用户侧数据和广告侧数据，广告侧数据使用之前统计到的内容枚举生成
        self.enumerateList = self.enumerateData()
        self.gernerateData()

        ##根据参数设置的模型类型，调用接口将生成的数据转化为模型接受的类型
        if self.modelsParam["type"] == "xgb":
            pre_res = self.predictWithXgboost()
        self.match_res()

    ###枚举生成数据中的固定部分
    ###枚举的内容包括：投放时间（小时），app、广告主来源id，广告主id,连接类型
    ###是个见了鬼的五重循环
    def enumerateData(self):
        enumerateList = []
        enumerateWeightList = []
        for appid in self.jsonDict["app_info"]:
            app_total = 0
            """
            for ad_net_id in self.jsonDict["app_info"][appid].keys():
                for ad_id in self.jsonDict["app_info"][appid][ad_net_id].keys():
                    app_total += self.jsonDict["app_info"][appid][ad_net_id][ad_id]
            """

            #for ad_net_id in self.jsonDict["app_info"][appid].keys():
                #for ad_id in self.jsonDict["app_info"][appid][ad_net_id].keys():
            for contype in self.jsonDict["contype_dict"].keys():
                h = 0
                while h < 24:
                    data = {
                        "app_id": appid,
                        #"advertiser_network_id": ad_net_id,
                        #"advertiser_id": ad_id,
                        "h": str(h),
                        "con_type": contype
                    }
                    h += 2
                    enumerateList.append(data)
        print len(enumerateList)
        return enumerateList

    def match_res(self):
        """
        :假定测试文件格式： device_id, app_id, h, convert(bool)
        """
        pass

    def getConfig(self):##获取配置文件的内容
        self.cf = ConfigParser.ConfigParser()
        self.cf.read(self.configFile)
        self.jsonFiles = self.config_from_sec("jsonFiles")
        self.modelsParam = self.config_from_sec("models")

    def config_from_sec(self, sec_name):
        sec = self.cf.options(sec_name)
        config_dict = {}
        for opt_name in sec:
            opt_value = self.cf.get(sec_name, opt_name)
            config_dict[opt_name] = opt_value
        print config_dict
        return config_dict

    def loadJsonFiles(self):
        print self.jsonFiles
        import json
        self.jsonDict = {}
        for fileKey in self.jsonFiles:
            with open(self.jsonFiles[fileKey], "rb") as jsonFile:
                json_str = jsonFile.readline()
                self.jsonDict[fileKey] = json.loads(json_str)
                print "file %s successful loaded!"%(fileKey)

    """
    #根据已知的统计数据表生成测试日的虚拟数据
    """
    def gernerateData(self):
        ##确定生成的数据的日期
        test_date = str(self.modelsParam["test_date"])

        ##确定生成数据保存的csv文件（数据文件，对应的权重、用于比对结果的id记录）
        with open(self.modelsParam["generate_file_dir"] + test_date + "_generate.csv", "wb") as csv_file:
            with open(self.modelsParam["generate_file_dir"] + test_date + "_weight.txt", "wb") as weight_file:
                import csv
                csv_writer = csv.writer(csv_file)
                weight_writer = csv.writer(weight_file)

                skip_did_cnt = 0
                accept_did_cnt = 0
                for device_id in self.jsonDict["device_dict"]:
                    did = str(self.jsonDict["device_dict"][device_id])

                    ###只使用在测试日里有使用的did
                    try:
                        self.jsonDict["prev5days_log"][test_date][did]
                        accept_did_cnt += 1
                    except:
                        skip_did_cnt += 1
                        continue

                    did_info = {
                        "brand": self.jsonDict["device_info"][did]["br"],
                        "model": self.jsonDict["device_info"][did]["md"],
                        "os_version": self.jsonDict["device_info"][did]["os"],
                        "app_platform": self.jsonDict["device_info"][did]["pf"],
                        "device_id": did,
                        "country": self.modelsParam["country"],
                        "ymd": self.modelsParam["test_date"]
                    }
                    ##last login, login sum, convert app, media{mb:cnt}
                    prev5days_log = self.jsonDict["prev5days_log"][test_date][did]
                    [mb_list, mb_weight] = self.countMediaWeight(prev5days_log[3])
                    self.transDataAndSave(did_info, mb_list, mb_weight, csv_writer, weight_writer)
                    if accept_did_cnt %100000 == 0:
                        print "总计采用设备id %d 个，跳过设备id %d 个" % (accept_did_cnt, skip_did_cnt)
                    #if accept_did_cnt == 1000:
                    #    break
                print "总计采用设备id %d 个，跳过设备id %d 个"%(accept_did_cnt, skip_did_cnt)

    def predictWithXgboost(self):
        """
        using xgboost model
        generate input data through the jsonFiles
        :return dict{  appid:[(did, pro)] }:
        """
        #import xgboost as xgb
        #bst = xgb.Booster()
        #bst.load_model(self.modelsParam["model"])
        pass


    def countMediaWeight(self, mb_dict):
        sum_cnt = sum(mb_dict.values())
        media_weight_list = [[], []]##meida_bundle,  weight of correspond media bundle
        for media_bunlde in mb_dict.keys():
            if sum_cnt == 0.0:
                media_weight = 0.0
            else:
                media_weight = mb_dict[media_bunlde] / sum_cnt
            if len(media_weight_list[0]) == 5:
                min_weight = min(media_weight_list[1])
                if min_weight < media_weight:
                    min_loc = media_weight_list[1].index(min_weight)
                    del(media_weight_list[0][min_loc])
                    del (media_weight_list[1][min_loc])
                else:
                    continue
            media_weight_list[0].append(media_bunlde)
            media_weight_list[1].append(media_bunlde)

        return media_weight_list

    ###特征值转换
    #存储一行生成的数据
    #同时还要生成另一个文件，每行对应记录这条数据是哪个did， app， 权值是多少
    def transDataAndSave(self, data, mb_list, mb_weight,csv_writer,weight_writer):
        tag_seg = ["country", "brand", "model", #"advertiser_network_id", "advertiser_id",
                   "app_platform", "app_id",
                   "device_id", "os_version", "media_bundle", "con_type", "ymd", "h"]
        ###枚举media_bundle并保存生成的数据
        #print len(self.enumerateList), "*", len(mb_list) ," = ",len(self.enumerateList) * len(mb_list), "while out=", self.cnt_out
        #self.cnt_out += len(self.enumerateList) * len(mb_list)
        for enumerateUnit in self.enumerateList:
            data.update(enumerateUnit)
            for mb_indx in range(len(mb_list)):
                data["media_bundle"] = mb_list[mb_indx]
                data_line = []
                weight_line = [data["device_id"], data["app_id"], mb_weight[mb_indx], mb_indx] ###设备id，app_id和枚举的这个media_bunlde的权重
                for seg in tag_seg:
                    data_line.append(data[seg])
                csv_writer.writerow(data_line)
                weight_writer.writerow(weight_line)



"""脚本的命令行输入提示"""
def printUsage():
    print "usage: simulaTest.py -f <configFileName>"

if __name__=="__main__":
    import getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:", ["file="])
    except getopt.GetoptError:
        printUsage()
        sys.exit(-1)

    configFile = "test.config"
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            configFile = arg
    print configFile
    new_test = simulaTest(configFile)
