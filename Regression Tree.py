#!/usr/bin/env python
# coding: utf-8

# Mengimpor package yang diperlukan

# In[7]:


import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import matplotlib.pyplot as plt 


# Fungsi untuk mengimpor dataset

# In[8]:


def importdata(): 
    dataset = pd.read_csv(r"C:\Users\Shella Lolitha\Documents\Python Jupyter\datamhs_clean2.csv")
    dataset2 = dataset[['kdjur', 'kdsla', 'nem', 'sks', 'ipk']]
    dataset2 = dataset2._get_numeric_data()
      
    # Printing the dataswet shape 
    print ("Dataset Lenght: ", len(dataset2)) 
    print ("Dataset Shape: ", dataset2.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: ",dataset2.head()) 
    return dataset2 


# Fungsi untuk melakukan split pada dataset

# In[9]:


def splitdataset(dataset2): 
  
    # Seperating the target variable 
    X = dataset2[['nem']].astype(int) 
    Y = dataset2[['kdjur']].astype(int) 
  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 


# Fungsi untuk melakukan training menggunakan giniIndex

# In[10]:


def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 


# Fungsi untuk melakukan data training dengan entropy.

# In[11]:


def tarin_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 


# Fungsi untuk membuat prediksi

# In[12]:


def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 


# Fungsi untuk mengkalkulasi akurasi

# In[13]:


def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 


# Fungsi main

# In[14]:


def main(): 
      
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
      
    # Operational Phase 
    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 


# Memanggil fungsi main

# In[15]:


if __name__=="__main__": 
    main() 

