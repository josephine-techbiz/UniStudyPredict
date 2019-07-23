# University Study (UPH) Prediction using SVM, RT, and RF through Python and Dash
Predicting University Study in UPH using  Support Vector Machine, Regression Tree, and Random Forest through Python and Dash


# Latar Belakang
   Pada umumnya siswa Sekolah Menengah Atas (SMA) atau calon mahasiswa akan bertanya-tanya dalam pemilihan program studi dan perguruan tinggi yang akan dipilih ketika mereka lulus. Hal ini disebabkan karena kurangnya pengetahuan akan fakultas dan program studi lalu mata kuliah apa yang akan digeluti dan prospek kerja setelah lulus. Penyebab-penyebab inilah yang menjadikan begitu banyak siswa SMA bingung dan salah mengambil keputusan sehingga mereka menyesal saat sudah memasuki dunia perkuliahan. Selain itu faktor-faktor lain yang dapat membuat salah keputusan adalah pengaruh teman, desakan orang tua atau keluarga, dan pertimbangan gengsi.
  
  Ada beberapa pertimbangan yang dapat mendukung calon mahasiswa dalam menentukan program studi, yang pertama adalah minat dan bakat. Minat dan bakat yang dimiliki tentunya juga harus dicocokan dengan silabus kuliah tiap program studi. Kemudian pertimbangan kedua adalah kemampuan intelektual, hal ini berkaitan dengan kemampuan dasar tertentu dari program studi yang diambil. Pertimbangan ketiga adalah kemampuan keuangan keluarga. Kuliah di suatu perguruan tinggi tidak lepas dari komponen biaya yang harus dikeluarkan dan jumlahnya cukup besar. Dengan mempertimbangkan faktor-faktor tersebut, maka keputusan pemilihan program studi ini harus dipikirkan dengan matang.
   
   Kesalahan dalam pemilihan program studi dapat berakibat fatal. Terutama apabila mahasiswa sampai pindah program studi bahkan berhenti kuliah karena tidak sesuai dengan minat dan bakat atau kemampuan intelektual yang mereka miliki. Hal tersebut tentunya akan membuang-buang waktu, tenaga, dan juga biaya yang tidak sedikit. Untuk dapat membantu siswa SMA atau calon mahasiswa dalam memilih program studi, maka diperlukan profiling calon mahasiswa agar mereka dapat memiliki bahan pertimbangan yang cukup dari data yang mereka miliki. 
   
   Pada penelitian mengenai profiling calon mahasiswa yang dirancang untuk penentuan program studi diharapkan dapat diterapkan pada calon mahasiswa. Profiling ini akan dilakukan pada calon mahasiswa Universitas Pelita Harapan (UPH). 

# Batasan Masalah
Dalam proyek ini, diperlukan beberapa batasan yang digunakan sebagai acuan dalam proyek untuk memberikan arah yang  jelas dalam pengembangannya.   Batasan-batasan   yang   terdapat   dalam   proyek ini   adalah sebagai berikut: 
1)	Aplikasi dikembangkan dengan menggunakan bahasa pemrograman Python dan menggunakan Jupyter Notebook sebagai Integrated Development Environment (IDE).
2)	Visualisasi aplikasi menggunakan Python Dash.
3)	Metode yang digunakan adalah Ensemble Learning.
4)	Algoritma-algoritma yang digunakan dalam penelitian ini adalah Support Vector Machine (SVM), Regression Tree, dan Random Forest.
5)	Data yang diambil berasal dari Kementerian Riset, Teknologi, Dan Pendidikan Tinggi Republik Indonesia (Ristekdikti).
6)	Data siswa yang diambil berasal dari data mahasiswa yang telah berkuliah di Universitas Pelita Harapan (UPH). 
7)	Data yang diperoleh terdiri dari Nomor Induk Mahasiswa (NIM), kode sekolah, nama sekolah, kode jurusan/program studi, nama jurusan, Nilai Ebtanas Murni (NEM) SMA, Satuan Kredit Semester (SKS), dan Indeks Prestasi Kumulatif (IPK) dari mahasiswa. 
8)	Data yang digunakan untuk perancangan aplikasi ini diambil dari mahasiswa yang memiliki rentang NEM 30.00 hingga 60.00 dan data mahasiswa yang tidak memiliki nilai Not Available (NA), nol (0), maupun nilai-nilai lain yang tidak sesuai dengan tipe datanya. 
9)	Aplikasi yang dikembangkan akan menunjukkan perbandingan insight dari algoritma-algoritma yang digunakan dan akan menampilkan hasil visualisasi.
10)	Hasil yang ditampilkan berupa Website dari Python Dash 

# Data Cleansing
Data cleansing atau data cleaning sering digunakan untuk berbagai kasus tentang peningkatan kualitas data. Cleansing data di proyek ini menggunakan bahasa pemrogramman R.
- menghapus nilai "NA" atau NULL 

       na.omit(data)
       data<-datamhs[!(data$kdsla=="\\N"),] # berdasarkan kode sekolah
       data<-datamhs[!(data$kdsla=="\\N"),]
       
- menggunakan fungsi aggregat untuk mendapatkan data dengan kategori tertentu
   
      data<-data!(data$nem<=30.0),] # batas nem minimum
      data<-data[!(data$nem>60.0),] # batas nem maksium

# 1. Support Vector Machine (SVM)
Konsep Klasifikasi dengan Support Vector Machine (SVM) adalah mencari hyperplane terbaik yang berfungsi sebagai pemisah dua kelas data. Ide sederhana dari SVM adalah memaksimalkan margin, yang merupakan jarak pemisah antara kelas data

# 2. Regression Tree (RT)
Regression Tree dibangun melalui proses yang dikenal sebagai partisi rekursif biner, yang merupakan proses berulang yang membagi data menjadi partisi atau cabang, dan kemudian melanjutkan pemisahan setiap partisi menjadi kelompok-kelompok yang lebih kecil ketika metode bergerak naik setiap cabang. Awalnya, semua catatan dalam Set Training dikelompokkan ke dalam partisi yang sama. Algoritma kemudian mulai mengalokasikan data ke dalam dua partisi atau cabang pertama, menggunakan setiap kemungkinan pemisahan biner pada setiap bidang. Algoritma memilih pemisahan yang meminimalkan jumlah penyimpangan kuadrat dari rata-rata di dua partisi terpisah. Aturan pemisahan ini kemudian diterapkan ke masing-masing cabang baru. Proses ini berlanjut hingga setiap node mencapai ukuran simpul minimum yang ditentukan pengguna dan menjadi simpul terminal. (Jika jumlah deviasi kuadrat dari rata-rata dalam simpul adalah nol, maka simpul itu dianggap sebagai simpul terminal bahkan jika belum mencapai ukuran minimum.)

**Used Python Packages:**
1.	sklearn: 
   - Dalam python, sklearn adalah machine learning package yang mencakup banyak algoritma machine learning.
   - Di sini, kami menggunakan beberapa modulnya seperti train_test_split, DecisionTreeClassifier dan accuracy_score.
   
2.	NumPy: 
   - Ini adalah modul python numerik yang menyediakan fungsi matematika cepat untuk perhitungan.
   - Digunakan untuk membaca data dalam array numpy dan untuk tujuan manipulasi.
   
3.	Pandas: 
   - Digunakan untuk membaca dan menulis file yang berbeda.
   - Data manipulation dapat dilakukan dengan mudah dengan dataframes.
   
**Asumsi yang kami buat saat menggunakan Decision Tree:**
-	Pada awalnya, kami menganggap seluruh training ditetapkan sebagai root.
-	Atribut diasumsikan kategorikal untuk perolehan informasi dan untuk gini indeks, atribut diasumsikan continous.
-	Atas dasar nilai-nilai atribut, catatan didistribusikan secara rekursif.
-	Kami menggunakan metode statistik untuk memesan atribut sebagai root atau node internal.

**Pseudocode:**
-	Cari atribut terbaik dan letakkan pada root node pada tree
-	Split training set pada dataset menjadi subsets. Ketika membuat subset, pastikan setiap subset dari training dataset mempunyai nilai yang sama untuk atribut
-	Mencari nodes leaf pada keseluruhan tree dengan mengulang 1 dan 2 pada setiap subset

**Ketika mengimplementasi decision tree, terdapat 2 fase:**
**1.	Building Phase**
   - Preprocess the dataset.
   - Split the dataset dari train dan test menggunakan Python sklearn package.
   - Train the classifier.
**2.	Operational Phase** 
   - Membuat predictions.
   - Calculate the accuracy.
   

**Data Slicing:**
-	Sebelum melakukan training model, kami melakukan split pada dataset menjadi training dan testing dataset
-	Untuk melakukan split dataset untuk training dan testing, kami menggunakan sklearn module yaitu train_test_split
-	Pertama-tama, kami memisahkan target variable dari atribut pada dataset
-	Dari code di atas, terdapat atribut X dan Y, X adalah dataset, Y adalah kode jurusan
-	Step selanjutnya adalah untuk split dataset untuk kebutuhan training dan testing
-	Line di atas berfungsi untuk split dataset untuk training dan testing. (ratio of 70:30 diantara training dan testing maka dari itu test_size parameter value = 0.3.)
-	Random_state variable adalah pseudo-random number generator state yang digunakan untuk random sampling

**Terms used in code:**
Gini index dan information gain, keduanya adalah method yang digunakan ntuk memilih dari n attributes dari dataset, atribut mana yang akan diletakan pada rood node atau internal node.

**Gini index**
![Screenshot](https://cdncontribute.geeksforgeeks.org/wp-content/uploads/decisionTree3.png)
Gini Index adalah metric untuk mengukur seberapa sering elemen yang terpilih secara random akan salah diidentifikasi
-	Artinya atribut dengan gini index yang kecil akan lebih baik
-	Sklearn supports “gini” kriteria untuk Gini Index dan secara default, mengambil “gini” value

**Entropy**
Entropy digunakan untuk mengukur ketidakpastian dari random variabel, itu mencirikan ketidakmurnian dari kumpulan contoh tidak beraturan. Semakin tinggi entropi, semakin banyak information gainnya.

**Information Gain**
-	Entropy biasanya berubah ketika menggunakan node pada decision tree menjadi partisi training menjadi bagian yang lebih kecil. Information gain digunakan untuk mengukur perubahan di dalam entropy
-	Sklearn supports entropy criteria untuk information gain dan ketika ingin menggunakan information gain method di sklearn, harus digunakan secara eksplisit.

**Accuracy score**
Akurasi score digunakan untuk mengkalkulasi akurasi dari trained classifier

**Confusion Matrix**
-	Digunakan untuk memahami perilaku trained classifier di atas dataset tes atau memvalidasi dataset
-	Ringkasan hasil prediksi pada masalah klasifikasi.
-	Jumlah prediksi yang benar dan salah dirangkum dengan nilai-nilai hitung dan dipecah oleh setiap kelas. Ini adalah kunci dari confusion matrix.
-	Confusion matrix menunjukkan cara-cara yang membuat confused model klasifikasi kita ketika membuat prediksi.
-	Ini memberi kita wawasan tidak hanya tentang kesalahan yang dibuat oleh classifier tetapi lebih penting lagi jenis kesalahan yang sedang dibuat.


# 3. Random Forest (RF)
Random forest (RF) adalah suatu algoritma yang digunakan pada klasifikasi data dalam jumlah yang besar. Klasifikasi random forest dilakukan melalui penggabungan pohon (tree) dengan melakukan training pada sampel data yang dimiliki. Random forest menggunakan Decision Tree untuk melakukan proses seleksi.

https://github.com/josessca/UniStudyPredict/blob/master/Random%20Forest%202.ipynb

Implementation in Scikit-learn
For each decision tree, Scikit-learn calculates a nodes importance using Gini Importance, assuming only two child nodes (binary tree):

![Screenshot](https://miro.medium.com/max/770/1*C-bkgMBs4drNVyBb1VJcEQ.png)

ni sub(j)= the importance of node j
w sub(j) = weighted number of samples reaching node j
C sub(j)= the impurity value of node j
left(j) = child node from left split on node j
right(j) = child node from right split on node j
sub() is being used as subscript isn’t available in Medium
See method compute_feature_importances in _tree.pyx
The importance for each feature on a decision tree is then calculated as:
![Screenshot](https://miro.medium.com/max/770/1*oar13be_cUsLR35MA_t6WQ.png) 

fi sub(i)= the importance of feature i
ni sub(j)= the importance of node j
These can then be normalized to a value between 0 and 1 by dividing by the sum of all feature importance values:
![Screenshot](https://miro.medium.com/max/770/1*uZPnQKYNmy7Tf3DvZ0e5tQ.png) 

The final feature importance, at the Random Forest level, is it’s average over all the trees. The sum of the feature’s importance value on each trees is calculated and divided by the total number of trees:
![Screenshot]( https://miro.medium.com/max/770/1*gK2tXtlbz12oMCdniAPPlg.png) 

RFfi sub(i)= the importance of feature i calculated from all trees in the Random Forest model
normfi sub(ij)= the normalized feature importance for i in tree j
T = total number of trees
See method feature_importances_ in forest.py
Notation was inspired by this StackExchange thread which I found incredible useful for this post.
