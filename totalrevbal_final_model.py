# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:47:39 2020

@author: admin
"""

# Load all the required Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import statsmodels.api as sm
import scipy.stats as stats
import pickle
import xgboost as xgb

#Load the dataset :

#dataset = pd.read_csv("C:/Users/admin/Desktop/python program files/Project/Data.csv", encoding='latin1' )

dataset = pd.read_csv("Data.csv", encoding='latin1' )

# Extract only required features

dataset_main = dataset[['loan_amnt ','Rate_of_intrst','annual_inc','debt_income_ratio','numb_credit','total_credits','total_rec_int','tot_curr_bal','total revol_bal']]

dataset_main.head()


#Data Treatment :

#Clean the dataset:

#Fill the missing values :

    
dataset_main.fillna(dataset_main.mean(),inplace=True)


dataset_main.isnull().sum()

#Outlier Treatment (Capping and Flooring Approach) :

for col in dataset_main.columns:
    percentiles = dataset_main[col].quantile([0.10,0.90]).values
    dataset_main[col][dataset_main[col] <=percentiles[0]]= percentiles[0]
    dataset_main[col][dataset_main[col]>=percentiles[1]]= percentiles[1]

#Splitting the data into Input and Output

X = dataset_main.iloc[:,0:8]
Y = dataset_main.iloc[:,8]




def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

#Normalise the Input Variables :
norm_func(X)

#Convert into matrix form :
   # data_dmatrix = xgb.DMatrix(data=X,label=y)


# Split the data in test and train :
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=10)
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

regressor=xgb.XGBRegressor(base_score=1, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.15, max_delta_step=0, max_depth=5,
             min_child_weight=2, missing=None, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)




#regressor.fit(x_train,y_train)
regressor.fit(X,Y)

prd = regressor.predict(x_test)

#rmse = np.sqrt(mean_squared_error(y_test, prd))
#print("RMSE: %f" % (rmse))

pickle.dump(regressor,open('finalmodel_TRB.pkl','wb'))



#prd_V = pd.DataFrame(prd_V, columns=['total revol_bal']).to_csv('prediction_VV.csv')

