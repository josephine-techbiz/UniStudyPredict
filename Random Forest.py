#!/usr/bin/env python
# coding: utf-8

# # University Study Prediction Menggunakan Random Forest
# 
#  - Data yang diambil adalah data mahasiswa UPH yang telah lulus 
#  - Data yang diperoleh berupa nim, jurusan, nem SMA, SMA asal, sks, dan ipk
#  - Prediksi ini menggunakan Random Forest
# 
# ## Akusisi Data
# 
# Kode Python berikut memuat dalam data csv dan menampilkan struktur data:

# In[6]:


# Pandas is used for data manipulation
import pandas as pd
from sklearn.model_selection import train_test_split
# Read in data and display first 5 rows
features = pd.read_csv('data4.csv')
features.head(5)


# ## Penjelasan tiap kolom:
# 
# - no: nomor urut data
# - kdjur: kode untuk masing- masing jurusan
# - nmjur: nama jurusan
# - nem: nilai ebtanas murni SMA masing-masing mahasiswa
# - sks: satuan kredit semester mahasiswa
# - status: keadaan mahasiswa lulus (angka 1) atau tidak (angka 0)
# - nemgrup: golongan nilai nem 
# - ipk: indeks prestasi kumulatif mahasiswa
# - kdsla: kode SMA asal
# - nmsla: nama SMA asal
# 

# ## Mengidentifikasi Anomali / Missing Data 
# 
# Data yang hilang dapat memengaruhi analisis sebagaimana data yang salah atau outlier. Dalam hal ini, data yang hilang tidak akan memiliki efek yang besar, dan kualitas datanya bagus karena sumbernya. Kita juga dapat melihat ada sebelas kolom yang mewakili sepuluh fitur dan satu target ('ipk').

# In[7]:


print('The shape of our features is:', features.shape)


# Untuk mengidentifikasi anomali, kami dapat dengan cepat menghitung statistik.

# In[8]:


# Descriptive statistics for each column
features.describe()


# ## One-Hot Encoding
# 
# Langkah pertama bagi kami dikenal sebagai One-Hot Encoding. Proses ini mengambil variabel kategorikal, seperti dan mengubahnya menjadi representasi numerik tanpa arbotary ordering.

# In[9]:


# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)
# Display the first 5 rows of the last 12 columns
features.iloc[:,50:].head(50)


# ## Fitur dan Target dan Konversi Data ke Array
# 
# Sekarang, kita perlu memisahkan data menjadi fitur dan target. Target, juga dikenal sebagai label, adalah nilai yang ingin kita prediksi, dalam hal ini ipk dan fitur adalah semua kolom yang digunakan model untuk membuat prediksi. Kami juga akan mengonversi dataframe Pandas ke array Numpy karena itulah cara algoritma bekerja. 

# In[10]:


# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(features['ipk'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('ipk', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)


# ## Training dan Testing Sets
# 
# Ada satu langkah terakhir dari persiapan data: memecah data menjadi training set dan testing. Selama training, biarkan model 'melihat' jawabannya, dalam hal ini ipk, sehingga dapat mempelajari bagaimana memperediksi ipk dari fitur. Kami berharap akan ada hubungan antara semua fitur dan nilai target, dan tugas model adalah mempelajari hubungan ini selama pelatihan. Kemudian, ketika tiba saatnya untuk mengevaluasi model, kami memintanya untuk membuat prediksi pada testing set di mana ia hanya memiliki akses ke fitur. Karena kami memiliki jawaban aktual untuk test set, kami dapat membandingkan prediksi ini dengan nilai sebenarnya untuk menilai seberapa akurat model tersebut. Secara umum, saat melatih suatu model, kami secara acak membagi data menjadi training set dan testing set untuk mendapatkan representasi semua poin data. Kami menetapkan random state ke 42 yang berarti hasilnya akan sama setiap kali saya menjalankan pemisahan untuk hasil yang dapat direproduksi.

# In[11]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# Kita dapat melihat bentuk semua data untuk memastikan kami melakukan semuanya dengan benar. Kami berharap jumlah fitur latihan kolom yang cocok dengan jumlah fitur fitur kolom dan jumlah baris yang cocok untuk masing-masing fitur training dan testing dan label:

# In[12]:


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# ## Establish Baseline
# 
# Sebelum kita dapat membuat dan mengevaluasi prediksi, kita perlu menetapkan baseline, ukuran yang masuk akal yang ingin kita kalahkan dengan model kita. Jika model kami tidak dapat meningkatkan pada baseline, maka itu akan menjadi kegagalan dan kami harus mencoba model yang berbeda atau mengakui bahwa machine learning tidak tepat untuk masalah kami. Dengan kata lain, baseline adalah error yang akan kami dapatkan jika kami memperkirakan NEM.

# In[13]:


# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('nem')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('nem baseline error: ', round(np.mean(baseline_errors), 2))


# ## Train Model
# Setelah semua pekerjaan persiapan data, membuat dan train model ini cukup sederhana menggunakan Scikit-learn. Kami mengimpor model random forest regression dari skicit-learning, instantiate model, dan fit (scikit-learn's name untuk training) model pada data training. (Sekali lagi mengatur random state untuk hasil yang dapat direproduksi). 

# In[15]:


#### TRAIN #####
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor


# In[16]:


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# In[17]:


# Train the model on training data
rf.fit(train_features, train_labels);


# ## Membuat Prediksi pada Test Set
# Model kami sekarang telah dilatih untuk mempelajari hubungan antara fitur dan target. Langkah selanjutnya adalah mencari tahu seberapa bagus modelnya! Untuk melakukan ini kami membuat prediksi pada fitur test (model tidak pernah diizinkan untuk melihat jawaban test). Kami kemudian membandingkan prediksi dengan jawaban yang diketahui. Saat melakukan regresi, kita perlu memastikan untuk menggunakan kesalahan absolut karena kita mengharapkan beberapa jawaban kita rendah dan sebagian lagi tinggi. Kami tertarik pada seberapa jauh prediksi rata-rata kami dari nilai aktual sehingga kami mengambil nilai absolut (seperti yang juga kami lakukan ketika menetapkan baseline).
# 

# In[18]:


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# ## Menentukan Performance Metrics
# Untuk menempatkan prediksi kami dalam perspektif, kami dapat menghitung akurasi menggunakan persentase kesalahan rata-rata dikurangi dari 100%.

# In[19]:


# Determine Performance Metrics
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels.shape)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 3), '%.')


# ## Variable Importance
# Untuk mengukur kegunaan semua variabel di seluruh random forest, kita dapat melihat relative importances dari variabel.

# In[20]:


# VARIABLE IMPORTANCE
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[21]:


# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
important_indices = [feature_list.index('nem'), feature_list.index('sks')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# ## Visualisasi
# Bagan plot sederhana dari pentingnya fitur untuk menggambarkan perbedaan dalam signifikansi relatif dari variabel.

# In[22]:


# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[ ]:




