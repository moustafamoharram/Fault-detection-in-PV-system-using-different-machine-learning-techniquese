import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler


#### Load Data ####
pv=pd.read_csv('Final_Detection.csv')
x=pv.drop(columns=['Case'])
y=pv.Case
##### prepare data####
for i in range(0,50):
   train_x, test_x, train_y, test_y= train_test_split(x, y, test_size=0.1, random_state=2, stratify=y)












####### dt #######
#for i in range(1,50):

   clf_dt= DecisionTreeClassifier(max_depth=10,criterion='gini',random_state=0)
#

   clf_dt.fit(train_x,train_y)
   y_predict=clf_dt.predict(test_x)
   print(i,clf_dt.score(test_x, test_y)*100)#f1_score(test_y, y_predict, average=None))

########### test new data ######
   # test=pd.read_csv('new_cases.csv')
   # yy=test.Case
   #
   # test=test.drop(columns=['Case'])
   # test=scaler.fit_transform(test)
   # counter=0
   # xx=test.size-1
   # for ii in range(0,273):
   #     if clf_dt.predict(test)[ii]== yy[ii]:
   #         counter=counter+1
   #     print(clf_dt.predict(test)[ii],yy[ii])
   #     counter_per=(counter/273)*100
   # print(i,(clf_dt.score(test_x, test_y) )*100,counter_per)