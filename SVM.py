#!/usr/bin/env python
# coding: utf-8

# # University Study Prediction Menggunakan SVM
# 
#  - Data yang diambil adalah data mahasiswa UPH yang telah lulus 
#  - Data yang diperoleh berupa nim, jurusan, nem SMA, SMA asal, sks, dan ipk
#  - Prediksi ini menggunakan SVM
#  
# ## Import Package
# 
# import package yang diperlukan

# In[73]:


# Splitting the dataset into the Training set and Test set 
import random
random.seed(123)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ## Prepare data
# 
# Membaca data dari file datamshclean.csv.
# 
# Mengambil kolom kdjur sebagai kdjur dan kdsla,nem,sks,ipk sebagai data2
# 
# Menampilkan data yang akan digunakan
# 
# Target yang digunakan yaitu kdjur

# In[89]:


#read data dr csv
data=pd.read_csv(r"C:\Users\Angel\Downloads\datamhsclean.csv")

#select kolom kdjur pada data jadiin variable kdjur
kdjur= data[['kdjur']]

#data2 ambil beberapa kolom dr data
data2 =data [['kdsla','nem','sks','ipk']]

#ambil dr data2 yang numeric
data2 = data2._get_numeric_data()

#print data yang ada sebanyak 5 baris
data.head(5)


# ## Splitting
# 
# data akan di pisah menjadi training dan test dimana jumlah test data sebanyak 25% dan training data sebanyak 75%
# 
# Menampilkan hasil data yang sudah di train dan test

# In[91]:


#split dataset jd training & test
data_train,data_test,kdjur_train,kdjur_test=train_test_split(data2,kdjur,test_size=0.25, random_state=123)

#tampilin yg udah di training & test
print("\ndata_train:\n")
print(data_train.head())
print(data_train.shape)

print("\nkdjur_train:\n")
print(kdjur_train.head())
print(kdjur_train.shape)

print("\ndata_test:\n")
print(data_test.head())
print(data_test.shape)

print("\nkdjur_test:\n")
print(kdjur_test.head())
print(kdjur_test.shape)


# Dapat dilihat 12039 dan 4014 merupakan jumlah data pada train dan test. Angka 1 dan 3 menunjukan banyak kolom pada data

# ## Feature Scaling
# 
# Scaling diperlukan untuk menghilangkan pengaruh feature yg memiliki range besar mendominasi feature yg memiliki range kecil yang biasa disebut dengan normalisasi

# In[93]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler

scale_data= StandardScaler()

data_train = scale_data.fit_transform(data_train)
data_test = scale_data.transform(data_test)

#print data train&test yang udah di scaling
print(data_train)
print(data_test)


# ## Fitting Classifier To The Training Set
# 
# Melakukan import package yang diperlukan
# 
# Menggunakan kernel linear
# 
# Fitting data yang telah di train pada tahap sebelumnya

# In[98]:


#import utk fitting classifier to the training set
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split

# Create a linear SVM classifier
clf = SVC(kernel ='linear', random_state=0)

# Train classifier (fitting data yg udh di train)
clf.fit(data_train, kdjur_train.values.ravel())


# ## Prediction

# In[115]:


#prediction test result
kdjur_pred=clf.predict(data_test)
print (kdjur_pred)


# ## Confusion Matrix

# In[116]:


#import confusion matrix
from sklearn.metrics import confusion_matrix

#confusion matrix
cm = confusion_matrix(kdjur_test, kdjur_pred)
print(cm)


# ## K Fold
# 
# accuracy digunakan untuk mengevaluasi model yang merupakan rasio dari jumlah kejadian yang diprediksi dengan benar dibagi dengan jumlah total kasus dalam dataset untuk memberikan persentase.  Kita akan menggunakan variabel penilaian saat menjalankan build dan mengevaluasi setiap model di langkah selanjutnya

# In[117]:


#K Fold
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

kfold = model_selection.KFold(n_splits=10, random_state=7)

modelCV = SVC(kernel ='linear', random_state=0)
scoring ='accuracy'
results = model_selection.cross_val_score(modelCV,data_train,kdjur_train.values.ravel(), cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f"%(results.mean()))


# ## Evaluating Classification Report

# In[118]:


#evaluating classification report
from sklearn.metrics import classification_report
print(classification_report(kdjur_test, kdjur_pred))


# In[ ]:




