import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer

#### Load Data ####
pv=pd.read_csv('../xgboost detect/Final_Detection.csv')
x=pv.drop(columns=['Case'])
y=pv.Case
##### prepare data####
for i in range(1,70):
    train_x, test_x, train_y, test_y= train_test_split(x, y, test_size=0.1, random_state=7, stratify=y)

    scaler=MaxAbsScaler()
    train_x=scaler.fit_transform(train_x)
    test_x=scaler.fit_transform(test_x)

####### dt #######



    clf_rf=RandomForestClassifier(max_depth=8, n_estimators=46, max_features=1,random_state=48,criterion='gini')
#

    clf_rf.fit(train_x,train_y)
    y_predict=clf_rf.predict(test_x)
    print(i,clf_rf.score(test_x, test_y))