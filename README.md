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
Data cleansing atau data cleaning sering digunakan untuk berbagai kasus tentang peningkatan kualitas data. 

# Ensemble Learning
Salah satu tugas dari machine learning,  pattern recognition dan data mining adalah untuk membangun model yang baik dari dataset. Proses menghasilkan model dari data dinamakan learning atau training, yang diselesaikan oleh learning algorithm. Model tersebut dapat disebut dengan sebuah hipotesis atau learner atau classifier. Ensemble learning adalah metode untuk memecahkan masalah yang sama dengan membangun dan mengkombinasikan suatu kumpulan classifier. Sebuah ensemble mengandung sejumlah learner yang dinamakan base learners, dihasilkan dari base learning algorithm seperti decision tree, neural network, naïve bayes classifier, dan lainnya. 

# 1. Support Vector Machine (SVM)
Konsep Klasifikasi dengan Support Vector Machine (SVM) adalah mencari hyperplane terbaik yang berfungsi sebagai pemisah dua kelas data. Ide sederhana dari SVM adalah memaksimalkan margin, yang merupakan jarak pemisah antara kelas data

# 2. Regression Tree (RT)
Regression Tree dibangun melalui proses yang dikenal sebagai partisi rekursif biner, yang merupakan proses berulang yang membagi data menjadi partisi atau cabang, dan kemudian melanjutkan pemisahan setiap partisi menjadi kelompok-kelompok yang lebih kecil ketika metode bergerak naik setiap cabang. Awalnya, semua catatan dalam Set Training dikelompokkan ke dalam partisi yang sama. Algoritma kemudian mulai mengalokasikan data ke dalam dua partisi atau cabang pertama, menggunakan setiap kemungkinan pemisahan biner pada setiap bidang. Algoritma memilih pemisahan yang meminimalkan jumlah penyimpangan kuadrat dari rata-rata di dua partisi terpisah. Aturan pemisahan ini kemudian diterapkan ke masing-masing cabang baru. Proses ini berlanjut hingga setiap node mencapai ukuran simpul minimum yang ditentukan pengguna dan menjadi simpul terminal. (Jika jumlah deviasi kuadrat dari rata-rata dalam simpul adalah nol, maka simpul itu dianggap sebagai simpul terminal bahkan jika belum mencapai ukuran minimum.)

# 3. Random Forest (RF)
Random forest (RF) adalah suatu algoritma yang digunakan pada klasifikasi data dalam jumlah yang besar. Klasifikasi random forest dilakukan melalui penggabungan pohon (tree) dengan melakukan training pada sampel data yang dimiliki. Random forest menggunakan Decision Tree untuk melakukan proses seleksi.
