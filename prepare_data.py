#!/usr/local/python/bin/python
#coding=utf-8

import numpy as np
import time
import datetime
import sys
import os
import pyhs2
import json
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cPickle as pickle
import MySQLdb
import uuid
from hdfs.client import Client

def insert_hive(table,filename):
    url = "http://192.168.0.30:50070"
    client = Client(url, root="/", proxy=None, timeout=None, session=None) 
    client.upload( 
        hdfs_path= "/user/hive/warehouse/adt.db/" + table, 
        local_path=filename)
    print 'upload ok, ', filename


def exec_sql(conn, sql):
    print sql
    start_sel = time.time()
    cur= conn.cursor()
    cur.execute(sql)
    cur.close()
    end_sel = time.time()
    print 'execute time spent = ', (end_sel-start_sel) , ' seconds'    

def mysql_select(sql):
    ret = []
    try:
      conn= MySQLdb.connect(
              host  = '192.168.0.210',
              port  = 3306,
              user  = '1tu1shu',
              passwd= '1tu1shu',
              db    = 'adtiming')
      cur= conn.cursor()

      result= cur.execute( sql )

      list = cur.fetchall();
      for s in list:
          ret.append(s)

      conn.commit()
      conn.close()
      return ret
    except Exception,e:
      print "Error :", Exception,e,sql
      if not conn is None:
          conn.close()
      return ret

    return ret        


SKIP_STEP_1=False
SKIP_STEP_2=False
SKIP_STEP_3=False
SKIP_STEP_4=False
SKIP_STEP_5=False
SKIP_STEP_6=False
SKIP_STEP_7=False


print datetime.datetime.now(),"准备训练数据"

conn = pyhs2.connect(host='192.168.0.30',
                                  port=10500,
                                  authMechanism='PLAIN',
                                  user='hive',
                                  password='',
                                  database='adt',
                                  )


yesterday=time.time()-3600*24
yesterday_str=time.strftime('%Y%m%d', time.localtime(yesterday))
fromday = yesterday-15*3600*24
process_day = fromday
process_day_str = time.strftime('%Y%m%d', time.localtime(process_day))
# 1.更新15天tag_click_convert数据,因为某些callback会在14,15日才到达

step = '1.更新15天tag_click_convert数据,因为某些callback会在14,15日才到达,大约4小时'
if not SKIP_STEP_1:
    print datetime.datetime.now(),step
    while process_day_str<=yesterday_str:
        ymd=process_day_str
        ymd_next_15_str=time.strftime("%Y%m%d", time.localtime(process_day+15*3600*24))

        sql = "insert overwrite table tag_click_convert partition(ymd=" + ymd +",h) " +\
          "select a.click_id,a.source,a.country,a.sdk_version,a.make,a.brand,a.model,a.request_id,a.bid_time,a.advertiser_network_id,a.advertiser_id,a.campaign_id,a.exchange_id,a.app_platform,a.app_id,a.inner_subid,a.position_id,a.device_id,a.os_version,a.carrier,a.ts,a.media_bundle,a.con_type,b.revenue,b.install_conversion,b.event_conversion,a.h from " +\
          "(SELECT click_id, source, day, hour, time, ip, worker_ip, country, latitude, longitude, referer, ua, sdk_version, make, brand, model, request_id, bid_time, advertiser_network_id, advertiser_id, campaign_id, creative_id, exchange_id, app_platform, app_id, inner_subid, position_id, ad_size, device_id, jump_type, sock_ip, worker_id, ro_type, source_id, carrier, province_code, adt_sub_id, os_version, publisher_id, publisher_app_id, publisher_placement_id, publisher_click_id, publisher_sub_id, dinfo_clickid_flag, company_id, business_type, publisher_type, ts, media_bundle, publisher_ext1, publisher_ext2, publisher_ext3, billing_type, bundle_name, con_type, deal_id, attribution_type, network_bm_id, network_am_id, advertiser_bm_id, advertiser_am_id, campaign_manager_id, publisher_manager_id, pub_app_type, placement_type, ymd, h FROM adt_click where ymd=" + ymd +" and source in(2,5)) a " + \
          " left join (SELECT click_id, click_source, click_time, country, media_bundle, tmp1, request_id, bid_time, network_id, advertiser_id, campaign_id, creative_id, exchange_id, position_id, app_platform, app_id, app_package, device_id, active_time, source_id, make, model, os_version, publisher_id, publisher_app_id, publisher_placement_id, company_id, province_code, business_type, type, ip, user_agent, worker_ip, revenue, publisher_cost, publisher_gt_cost, deduct, publisher_click_id, publisher_sub_id, worker_id, billing_type, install_conversion, event_conversion, con_type, is_reattribution, is_reengagement, attribution_type, deal_id, publisher_cast_cost, publisher_subscription_cost, adt_sub_id, inner_subid, network_bm_id, network_am_id, advertiser_bm_id, advertiser_am_id, campaign_manager_id, publisher_manager_id, pub_app_type, placement_type, publisher_type, ymd, h FROM adt_callback " + \
          " where ymd>=" + ymd + " and ymd<=" + ymd_next_15_str+ " and click_source in (2,5)) b " + \
          " on a.click_id=b.click_id "
        exec_sql(conn, sql)
        #准备下一天
        process_day = process_day + 3600*24
        process_day_str = time.strftime('%Y%m%d', time.localtime(process_day))

step = '2.统计bundle的转化数和点击数'

if not SKIP_STEP_2:
    print datetime.datetime.now(),step
    sql = "drop table if exists tag_bundle"
    exec_sql(conn, sql)
    sql = "create table tag_bundle as " +\
          "select media_bundle,count(*) as click_cnt, sum(case when install_conversion+event_conversion>0 then 1 else 0 end) convert_cnt " + \
          "from tag_click_convert where ymd>=" + time.strftime('%Y%m%d', time.localtime(fromday)) + " and ymd<=" + yesterday_str +" group by media_bundle"
    exec_sql(conn, sql)

step = '3.从mysql 更新app信息到hive tag_app表'
if not SKIP_STEP_3:
    print datetime.datetime.now(),step
    sql = "select plat,app_id,category,category_id,rating_value,price,file_size from app"
    result = mysql_select(sql)
    tmp_filename = str(uuid.uuid1())
    f = open(tmp_filename,"w")
    for r in result:
        s = ""
        for i in range(0,len(r)):
           s = s + str(r[i])
           if i == len(r)-1:
                 s = s + "\n"
           else:
                 s = s + "\01"
        f.write(s)
    f.close()
    # 入hive
    sql = "drop table if exists tag_app"
    exec_sql(conn, sql)    
    sql = "create table tag_app( " + \
          "plat int, " +\
          "app_id string, " +\
          "category string, " +\
          "category_id int, " +\
          "rating_value double, " +\
          "price double, " +\
          "file_size int)"
    exec_sql(conn, sql)    
    insert_hive('tag_app',tmp_filename)
    os.remove(tmp_filename)

step = '4.统计15 日model 转换数与点击数'

if not SKIP_STEP_4:
    print datetime.datetime.now(),step
    sql = "drop table if exists tag_model"
    exec_sql(conn, sql)    
    sql="create table tag_model as " + \
        "select model, " + \
        "count(*) as click_cnt, " + \
        "sum(case when install_conversion+event_conversion>0 then 1 else 0 end) convert_cnt " + \
        "from tag_click_convert where ymd>=" + time.strftime('%Y%m%d', time.localtime(fromday)) + " and ymd<=" + yesterday_str +" group by model"
    exec_sql(conn, sql)    

step = '5.统计15 日ad_app_id 转换数与点击数'

if not SKIP_STEP_5:
    print datetime.datetime.now(),step
    sql = "drop table if exists tag_ad_app_id"
    exec_sql(conn, sql)    
    sql = "create table tag_ad_app_id as " + \
          "select app_id,count(*) as click_cnt, sum(case when install_conversion+event_conversion>0 then 1 else 0 end) convert_cnt " +\
          "from tag_click_convert where ymd>=" + time.strftime('%Y%m%d', time.localtime(fromday)) + " and ymd<=" + yesterday_str +" group by app_id"
    exec_sql(conn, sql)  


step = '6.统计15日移动平均 tag_did_media_bundle'

if not SKIP_STEP_6:
    print datetime.datetime.now(),step
    sql = "insert overwrite table tag_did_media_bundle partition(ymd=" + yesterday_str + ") " + \
          "select if(a.device_id is null, b.device_id,a.device_id) device_id, if(a.media_bundle is null, b.media_bundle, a.media_bundle) media_bundle, (2*if(a.cnt is null,0,a.cnt)+14*if(b.cnt is null,0,b.cnt))/16 cnt " + \
          "from " + \
          "(select device_id,media_bundle,sum(cnt) cnt from did_media_bundle where ymd=" + yesterday_str + " group by device_id,media_bundle) a full join " + \
          "(select device_id,media_bundle,cnt from tag_did_media_bundle where ymd=" +  time.strftime('%Y%m%d', time.localtime(yesterday-3600*24)) + ") b " + \
          "on a.device_id=b.device_id and a.media_bundle=b.media_bundle " + \
          "where a.cnt>0 or b.cnt>0.01"
    exec_sql(conn, sql)  

step = '7.创建与tag_did_media_bundle关联的昨日数据采样训练集'

if not SKIP_STEP_7:
    print datetime.datetime.now(),step
    process_day = fromday
    process_day_str = time.strftime('%Y%m%d', time.localtime(process_day))
    while process_day_str<=yesterday_str:
        ymd=process_day_str


        sql = "insert overwrite table tag_sample_v2 partition(ymd=" + ymd +") " + \
              "select a.click_id,a.device_id,a.model,a.app_id,a.country,a.conversion,if(b.media_bundle is null,a.media_bundle,b.media_bundle),if(b.cnt is null,1.0,b.cnt),rand(221) r " + \
              "from " + \
              "(select  click_id,device_id,media_bundle,model,app_id,country,install_conversion+event_conversion as conversion, ymd from " +\
                  "(select a.*,rand(1000) as random from tag_click_convert a where ymd=" + ymd +") f " + \
                  "where (int(random*10000)=1 or install_conversion+event_conversion>0) " + \
              ") a full join " + \
              "( select device_id,media_bundle, cnt from tag_did_media_bundle where ymd=" + ymd + \
              ") b on a.device_id=b.device_id " + \
              "where a.device_id is not null " + \
              "order by r"

        exec_sql(conn, sql)
        #准备下一天
        process_day = process_day + 3600*24
        process_day_str = time.strftime('%Y%m%d', time.localtime(process_day))    

print datetime.datetime.now(),'准备训练数据完成'