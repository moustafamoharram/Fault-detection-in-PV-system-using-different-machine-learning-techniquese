import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
for i in range(1,2):
   train_x, test_x, train_y, test_y= train_test_split(x, y, test_size=0.1, random_state=0, stratify=y)

   scaler= MaxAbsScaler()
   train_x=scaler.fit_transform(train_x)
   test_x=scaler.fit_transform(test_x)

###### dt #######
#for i in range(1,50):

   cl_knn= KNeighborsClassifier(4)
#

   cl_knn.fit(train_x,train_y)
   y_predict= cl_knn.predict(test_x)
   print(i,cl_knn.score(test_x, test_y)