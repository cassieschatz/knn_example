#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:25:16 2021

@author: cassieschatz
"""
#Import Statements:
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



raw_data = pd.read_csv('data_mnist.csv')

#Used to count each number's frequency, will also be used later.
y = raw_data['label']
  
#Scalling the data, making panda array:
scaler = StandardScaler()
scaler.fit(raw_data.drop(['label'], axis=1))
scaled_features = scaler.transform(raw_data.drop('label', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns = raw_data.drop('label', axis=1).columns)

#Here is just the data, without labels:
x = scaled_data

#Splitting the data:
x_training_data, x_testing_data, y_training_data, y_testing_data = train_test_split(x, y, test_size = 0.30)

knn = KNeighborsClassifier(n_neighbors = 2)
    
knn.fit(x_training_data, y_training_data)



#The Predictions:
final_test = pd.read_csv('test_mnist.csv')    

scaler = StandardScaler()

scaler.fit(final_test)
scaled_final_features = scaler.transform(final_test)
scaled_final_data = pd.DataFrame(scaled_final_features, columns = final_test.columns)

pred = knn.predict(scaled_final_data)
count = 1
print("ImageId, Actual")
for p in pred:
    print("{count}, {p}".format(count=count,p=p))
    count = count + 1
    
