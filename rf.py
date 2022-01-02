# -*- coding: utf-8 -*-
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : http://tekrajchhetri.com/
# @File    : rf.py
# @Software: PyCharm

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import joblib
import pickle
from sklearn import metrics
from sklearn.metrics import precision_recall_curve


df = pd.read_csv("data.csv")
inputdf = df[['cpu_utilization',
             'memory_utilization', 
             'network_overhead', 
             'io_utilization',
             'bits_outputted', 
             'bits_inputted',
             'smart_188', 
             'smart_197', 
             'smart_198', 
             'smart_9', 
             'smart_1',
             'smart_5',
             'smart_187', 
             'smart_7', 
             'smart_3', 
             'smart_4',
             'smart_194',
             'smart_199']]

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
inputdf = imp.fit_transform(inputdf)  
scaler = StandardScaler()
inputdf = scaler.fit_transform(inputdf)
X_train, X_test, y_train, y_test = train_test_split(
    inputdf, df["target"], test_size=0.4, random_state=42)

rfparams = {'n_estimators':[100,200,300,500],
            'max_features':['auto','log2','sqrt'],
            'criterion':['gini','entropy'],
            'min_samples_split': [3,5,8,9,10,30,50]}
rf_gridcv = GridSearchCV(RandomForestClassifier(random_state=42), rfparams, verbose=1, n_jobs=12,cv=3)
rf_gridcv.fit(X_train, y_train)
y_pred = rf_gridcv.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
precision, recall, thresholds = precision_recall_curve(
    y_test, y_pred)
print("Precision = {} and Recall = {}".format(precision, recall))
