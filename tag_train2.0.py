#!/usr/local/python/bin/python
#coding=utf-8

import numpy as np
import time
import datetime
import sys
import pyhs2
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.grid_search import GridSearchCV
import cPickle as pickle
#added zwt
import pandas as pd
import xgboost as xgb
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
#added zwt

#added zwt
def modelfit(alg, X_train, y_train, userTrainCV=True, cv_folds=5, early_stopping_rounds=50):
	X_train = pd.DataFrame(X_train)
	y_train = pd.DataFrame(y_train)
	if userTrainCV:
		xgb_param = alg.get_xgb_params()
		#xgtrain = xgb.DMatrix(dtrain[predictor].values, label=dtrain[target].values)
		xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)
		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
		alg.set_params(n_estimators=cvresult.shape[0])
		
	#Fit the algorithm on the data
	#alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
	alg.fit(X_train, y_train, eval_metric='auc')
	
	#Predict training set:
	dtrain_predictions = alg.predict(X_train)
	dtrain_predprob = alg.predict_proba(X_train)[:,1]

	#Print model report:
	print "\nModel Report"
	print "Accuracy : %.4g" % metrics.accuracy_score(y_train.values, dtrain_predictions)
	print "AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob)

	feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
	feat_imp.plot(kind='bar', title='Feature Importances')
	plt.ylabel('Feature Importance Score')

def hive_execute(conn,sql):
    print sql
    start_sel = time.time()
    cur= conn.cursor()
    cur.execute(sql)
    end_sel = time.time()
    print "execute sql time spent =", (end_sel-start_sel) , " seconds"

def hive_select(conn,sql):
    ret =[]
    print sql
    start_sel = time.time()
    cur= conn.cursor()
    cur.execute(sql)
    end_sel = time.time()


    #打印数据
    result= cur.fetchall();
    for item in result:
    	ret.append(item[0])

    cur.close()

    print "select count= ", len(ret), ', time spent = ', (end_sel-start_sel) , ' seconds'
    return ret

def build_in_add_1(item, line, dicts, idx):
  in_dicts = False
  for d in dicts:
    if item[idx] == d:
      line.append(1.0)
      in_dicts = True
    else:
      line.append(0.0)

  if in_dicts:
    line.append(0.0)
  else:
    line.append(1.0) 
  return 
def build_1_plus(item, line, map, idx):
  if item[idx] in map.keys():
    line.append(map[item[idx]])
  else:
    line.append(float(len(map)))
  return 

def build_map(list):
  m = {}
  for i in range(0,len(list)):
    m[list[i]] = float(i)
  return m

def build_collection(cur):

  START_COL_MB_TYPE = len(media_bundle_names)+1
  lines = []
  click_id = ''
  line = []
  result= cur.fetchall();
  for item in result:
    new_click_id = item[IDX_CLICK_ID]
    
    if click_id != new_click_id: #新 click_id  
      click_id = new_click_id   
      #原来的加入到数据列表
      if len(line) > 0:
        lines.append(line)     
      line = []
      # Mediabundle+1
      for i in range(0,len(media_bundle_names)+1):
        line.append(0.0)
      # mediabundle_type+1  
      for i in range(0,len(mediabundle_type_ids)+1):
        line.append(0.0)
      # model+1
      build_1_plus(item, line, model_map, IDX_MODEL)

      # app_id+1
      build_1_plus(item, line, app_id_map, IDX_APP_ID)

      # app_type+1
      build_in_add_1(item, line, mediabundle_type_ids, IDX_APP_TYPE)
      
      # app_rank, 做1/RANK排名，没有排名的为1/200
      rank = item[IDX_APP_RANK]
      if rank is None:
        rank = 200
      line.append(1.0/rank)

      conversion = item[IDX_CONVERSION]
      if conversion is None:
        conversion = 0.0
      line.append(conversion)    
    
    #  将mediabundle  统计 放入对应的位置
    mediabundle = item[IDX_MEDIA_BUNDLE]
    cnt = item[IDX_CNT]
    if cnt is None:
      cnt = 0.0
    if mediabundle in media_bundle_map.keys():
      line[ int(media_bundle_map[mediabundle]) ] = line[ int(media_bundle_map[mediabundle]) ] + cnt
    else:
      line[ len(media_bundle_names) ] = line[ len(media_bundle_names) ] + cnt


    # 将mediabundle_type  统计放入对应的位置
    mediabundle_type = item[IDX_MEDIA_BUNDLE_TYPE]
    if mediabundle_type in mediabundle_type_map.keys():
      line[ START_COL_MB_TYPE + int(mediabundle_type_map[mediabundle_type]) ] = line[ START_COL_MB_TYPE + int(mediabundle_type_map[mediabundle_type]) ] + cnt
    else:
      line[ START_COL_MB_TYPE + len(mediabundle_type_ids) ] = line[ START_COL_MB_TYPE + len(mediabundle_type_ids) ] + cnt


  #最后一行
  if len(line) > 0:
    lines.append(line) 

  cur.close()
  return lines

################################################################

USE_CV         = True      #使用CV遍历参数，很慢
XGB_CV		   = True      #使用CV获取迭代次数

################################################################


ymd_str= time.strftime("%Y%m%d", time.localtime(time.time() - 3600*24))

timeArray= time.localtime(time.time())
ymdhm_str= time.strftime("%Y%m%d_%H%M", timeArray)


conn = pyhs2.connect(host='192.168.0.30',
                                  port=10500,
                                  authMechanism='PLAIN',
                                  user='hive',
                                  password='',
                                  database='adt',
                                  )

step_name = '1.获取模型输入格式化参数'
print datetime.datetime.now(),step_name

# a. mediabundle, 转化>45，转化率从高到低排序  ＋  １，大约1500
sql = "select media_bundle,convert_cnt,click_cnt from tag_bundle where convert_cnt>45 order by convert_cnt/click_cnt desc"
media_bundle_names = hive_select(conn,sql)
media_bundle_map = build_map(media_bundle_names)


# b. mediabundle_type, 准备数据阶段已经准备好,大约44个
sql = "select distinct category_id from tag_app where category_id is not null order  by category_id"
mediabundle_type_ids = hive_select(conn,sql)
mediabundle_type_map = build_map(mediabundle_type_ids)

# c. model, 转化>95+其它,大约1000个
sql = "select model,convert_cnt from tag_model where convert_cnt>95 order by convert_cnt desc"
model_names =  hive_select(conn,sql)
model_map = build_map(model_names)

# d.app_id	转化>20+1，  大约410个
sql = "select app_id,convert_cnt from tag_ad_app_id where convert_cnt>20 order by convert_cnt desc"
app_id_names = hive_select(conn,sql)
app_id_map = build_map(app_id_names)

format = {"media_bundle_names":media_bundle_names,
          "mediabundle_type_ids":mediabundle_type_ids,
          "model_names":model_names,
          "app_id_names":app_id_names}

format_json=json.dumps(format)
f = open("format.json." + ymdhm_str,"wt")
f.write(format_json)
f.close()

# 训练数据格式:Mediabundle+1,mediabundle_type+1,model+1,app_id+1,app_type+1,app_rank, 结果
col_cnt = len(media_bundle_names) + 1 + \
          len(mediabundle_type_ids) + 1 + \
          1 + \
          1 + \
          len(mediabundle_type_ids) + 1 + \
          1 + 1

print datetime.datetime.now(),'数据列的数量为',col_cnt


step_name = '2.查询，组合训练数据'
print datetime.datetime.now(),step_name


sql = "select a.click_id,a.media_bundle, b.category_id as mediabundle_type,model,a.app_id, c.category_id as app_type, d.rank as app_rank,a.conversion,a.cnt from " + \
      "(select * from tag_sample_v2 where ymd>=20180827 and app_id<>'com.forshared' and cnt<100 ) a " + \
      "left join tag_app b on a.media_bundle=b.app_id " +\
      "left join tag_app c on a.app_id=c.app_id " + \
      "left join tag_app_rank d on a.app_id=d.app_id and a.country=d.country " + \
      "order by click_id"

print sql
start_sel = time.time()
cur= conn.cursor()
cur.execute(sql)

IDX_CLICK_ID      = 0
IDX_MEDIA_BUNDLE  = 1
IDX_MEDIA_BUNDLE_TYPE = 2
IDX_MODEL         = 3
IDX_APP_ID        = 4
IDX_APP_TYPE      = 5
IDX_APP_RANK      = 6
IDX_CONVERSION    = 7
IDX_CNT           = 8



lines_train = build_collection(cur)
end_sel = time.time()

print datetime.datetime.now(),"select count= ", len(lines_train), ', time spent = ', (end_sel-start_sel) , ' seconds'       


step_name = '3.训练'
print datetime.datetime.now(),step_name

start_train = time.time()
train_data_set = np.array(lines_train)



print datetime.datetime.now(),'train_data_set.shape',train_data_set.shape

X=train_data_set[:,0:col_cnt-1]
Y=train_data_set[:,col_cnt-1]


seed = 7
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# 缺省参数 XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
#       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
#       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#       silent=True, subsample=1)

model = XGBClassifier(
  learning_rate =0.1,
  n_estimators = 100,
  max_depth = 9,
  min_child_weight = 1,
  max_delta_step = 0,
  lambda = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = 1,
  nthread=8,
  seed=27
  )
print datetime.datetime.now(),'model = ', model

if USE_CV:
  param_test2={
   'max_depth':range(7,10,2),
   'min_child_weight':range(1,5,2)
  }
  gs = GridSearchCV(estimator=model, 
    param_grid = param_test2, scoring='roc_auc',n_jobs=3,iid=False, cv=5)

  gs.fit(X_train, y_train)
#added zwt
else if XGB_CV:
  modelfit(model, X_train, y_train)
#added zwt
else:
  model.fit(X_train, y_train)
  with open('xgboost.model.'+ymdhm_str, "wb") as f:  
    pickle.dump(model, f)   


end_train = time.time()

print datetime.datetime.now(),"train time spent = ", (end_train-start_train) , ' seconds'       

if USE_CV:
  print gs.grid_scores_,gs.best_params_,gs.best_score_


step_name = '4.验证'
print datetime.datetime.now(),step_name

start_verify = time.time()
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
end_verify = time.time()

print datetime.datetime.now(),"verify 1 time spent = ", (end_verify-start_verify) , ' seconds'       


accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

recall = recall_score(y_test,predictions)
print("Recall: %.2f%%" % (recall * 100.0))

auc = roc_auc_score(y_test,y_pred)
print("Auc: %.2f%%" % (auc * 100.0))

precision = precision_score(y_test,y_pred)
print("Precision: %.2f%%" % (precision * 100.0))


start_verify = time.time()
y_pred = model.predict(X_train)
predictions = [round(value) for value in y_pred]
end_verify = time.time()

print datetime.datetime.now(),"verify 2 time spent = ", (end_verify-start_verify) , ' seconds'       


accuracy = accuracy_score(y_train, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

recall = recall_score(y_train,predictions)
print("Recall: %.2f%%" % (recall * 100.0))

auc = roc_auc_score(y_train,y_pred)
print("Auc: %.2f%%" % (auc * 100.0))

precision = precision_score(y_train,y_pred)
print("Precision: %.2f%%" % (precision * 100.0))
