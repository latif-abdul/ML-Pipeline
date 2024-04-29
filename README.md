# Submission 2: Sistem Prediksi Penyakit Jantung
Nama: Abdul Latif

Username dicoding: latif_abdul

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Heart Desease Prediction](https://www.kaggle.com/datasets/mexwell/heart-disease-dataset) |
| Masalah | Penyakit jantung merupakan salah satu penyakit kronis yang menjadi penyebab kematian terbanyak di dunia. Menurut data <i>World Health Organization (WHO)</i> pada tahun 2019, penyakit jantung bertanggung jawab atas 17,9 juta kematian, atau sekitar 32% dari total kematian global. Di Indonesia, penyakit jantung juga menjadi salah satu penyakit terbanyak yang menyebabkan kematian. Berdasarkan data Riskesdas tahun 2018, prevalensi penyakit jantung di Indonesia mencapai 7,3%. <b>Himpunan data ini terdiri dari 1190 instans dengan 11 fitur. Kumpulan data ini dikumpulkan dan digabungkan di satu tempat untuk membantu memajukan penelitian tentang pembelajaran mesin terkait CAD dan algoritma penambangan data, dan mudah-mudahan pada akhirnya dapat memajukan diagnosis klinis dan perawatan dini. Dengan menggunakan dataset ini diharapkan bisa menjadi pendukung dibuatnya sistem untuk memprediksi seseorang memiliki penyakit jantung atau tidak</b> |
| Solusi machine learning | Berdasarkan masalah tersebut dibutuhkan sistem machine learning yang bisa memprediksi seseorang memiliki penyakit jantung atau tidak |
| Metode pengolahan | 1. Pengumpulan Data :<br> Sumber Data: kaggle <br>2. Pra-pemrosesan Data:<br>Ekstraksi Fitur: Fitur-fitur pada data diekstraksi.<br>Representasi Numerik: Fitur yang berupa teks diubah menjadi representasi numerik yang dapat dipahami oleh model machine learning.<br>3. Pembuatan Model Machine Learning:<br>Pilihan Algoritma: Algoritma machine learning yang umum digunakan untuk klasifikasi adalah <i>Deep learning</i><br>Pelatihan Model: Model machine learning dilatih menggunakan data yang telah dipraproses.<br>Penyesuaian Hiperparameter: Parameter model machine learning dioptimalkan untuk mencapai kinerja terbaik dalam mengklasifikasikan data baru.<br>4. Evaluasi Model:<br>Metrik Evaluasi: Metrik evaluasi seperti akurasi, presisi, dan recall digunakan untuk mengukur kinerja model machine learning pada data uji.<br>Penyesuaian Model: Jika model tidak menunjukkan kinerja yang memuaskan, model perlu disesuaikan dengan mengubah algoritma, hiperparameter, atau data pelatihan. <br>Pemantauan dan Pembaruan: Kinerja model machine learning perlu dipantau secara berkala dan diperbarui dengan data baru untuk menjaga akurasinya. |
| Arsitektur model | arsitektur model yang digunakan adalah input layer dan 4 layer dense dan setelah proses tuning didapatkan learning rate terbaik yaitu 0.01 dan epoch terbaik 2 |
| Metrik evaluasi | metrik yang digunakan yaitu binary accuracy |
| Performa model | performa model yang dihasilkan cukup baik dengan binary_akurasi 99% dan val_binary_accuracy 79% |
| Opsi Deployment | Model dideploy ke server cloud cloudeka |
| Web App | http://103.190.215.239:8501/v1/models/cc-model/metadata |
| Hasil Monitoring | Hasil monitoring menunjukkan nilai performa model 2961 |