import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
#### Load Data ####
pv=pd.read_csv('Final_Detection.csv')
x=pv.drop(columns=['Case'])
y=pv.Case
##### prepare data####
for i in range(1,70):

    train_x, test_x, train_y, test_y= train_test_split(x, y, test_size=0.1, random_state=50, stratify=y)
    scaler=MinMaxScaler()
    train_x=scaler.fit_transform(train_x)
    test_x=scaler.fit_transform(test_x)
####### building classifier#####
#for i in range(1,70):
    clf_xgb = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
              gamma=1, gpu_id=-1, importance_type='gain',
              interaction_constraints='', learning_rate=1,
              max_delta_step=0, max_depth=2, min_child_weight=2,
              monotone_constraints='()', n_estimators=100, n_jobs=16,
              num_parallel_tree=1, objective='binary:logic',
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=0.5,
              tree_method='hist',process_type='default',
              validate_parameters=1, verbosity=None,random_state=67)
    clf_xgb.fit(train_x, train_y)

    y_predict=clf_xgb.predict(test_x)
    print(i,clf_xgb.score(test_x, test_y))