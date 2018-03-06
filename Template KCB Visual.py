"""
Template Kecerdasan Buatan (KCB)
Grup Riset Sensor Visual

Author: @lzharif
"""

# Impor Pustaka
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV

# Nama penyimpanan berkas-berkas
filename_noproc = 'SVM_noproc.sav'
filename_pca = 'SVM_pca.sav'
filename_lda = 'SVM_lda.sav'
filename_kpca = 'SVM_kpca.sav'
filename_scale = 'scale.sav'
filename_dr_pca = 'pca.sav'
filename_dr_lda = 'lda.sav'
filename_dr_kpca = 'kpca.sav'
filename_res_noproc = 'SVM_res_noproc.txt'
filename_res_pca = 'SVM_res_pca.txt'
filename_res_lda = 'SVM_res_lda.txt'
filename_res_kpca = 'SVM_res_kpca.txt'

# Implementasi Grid Search dengan Parallel Computing
# Grid Searching adalah salah satu metode tuning parameter2 kecerdasan buatan
# Sumber --> https://en.wikipedia.org/wiki/Hyperparameter_optimization
# Tutorial grid searching --> http://scikit-learn.org/0.16/auto_examples/model_selection/grid_search_digits.html
# NB: Fungsi ini baru dieksekusi pada baris terakhir program
def cariGrid(clsf, preproc, xtr, ytr, xte, yte, accu, std, test_accu):
    # Penentuan kombinasi-kombinasi parameter yang akan diujicobakan, bisa berbeda untuk tiap2 kecerdasan buatan
    # Contoh untuk SVM
    parameters = [{'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]}]

    # Inisialisasi fungsi GridSearchCV menggunakan classifier clsf
    grid_search = GridSearchCV(estimator=clsf,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=10,
                               n_jobs=-1,
                               verbose=0)

    # Fit grid search dengan data latih
    grid_search = grid_search.fit(xtr, ytr)

    # Catat skor akurasi terbaik
    best_accuracy = grid_search.best_score_

    # Catat index konfigurasi grid terbaik
    best_index = grid_search.best_index_
    best_std = grid_search.cv_results_['std_test_score'][best_index]

    # Catat parameter terbaik
    best_parameters = grid_search.best_params_

    # Latih model dengan best parameter
    clsf = SVC(**best_parameters).fit(xtr, ytr)

    # Hitung akurasi pada data uji menggunakan parameter terbaru
    test_optimized = grid_search.score(xte, yte)

    # Jika sudah selesai, simpan semua konfigurasi kecerdasan buatan hasil grid searching dalam berkas
    if preproc == 'noproc':
        with open(filename_noproc, 'wb') as f:
            pickle.dump(clsf, f)
        with open(filename_res_noproc, "w") as text_file:
            text_file.write("%f %f %f %f %f %f %s" % (accu, std, test_accu, best_accuracy, best_std, test_optimized, best_parameters))
    elif preproc == 'pca':
        with open(filename_pca, 'wb') as f:
            pickle.dump(clsf, f)
        with open(filename_res_pca, "w") as text_file:
            text_file.write("%f %f %f %f %f %f %s" % (accu, std, test_accu, best_accuracy, best_std, test_optimized, best_parameters))
    elif preproc == 'lda':
        with open(filename_lda, 'wb') as f:
            pickle.dump(clsf, f)
        with open(filename_res_lda, "w") as text_file:
            text_file.write("%f %f %f %f %f %f %s" % (accu, std, test_accu, best_accuracy, best_std, test_optimized, best_parameters))
    else:
        with open(filename_kpca, 'wb') as f:
            pickle.dump(clsf, f)
        with open(filename_res_kpca, "w") as text_file:
            text_file.write("%f %f %f %f %f %f %s" % (accu, std, test_accu, best_accuracy, best_std, test_optimized, best_parameters))
    
    # Cetak hasil grid searching
    print(best_accuracy)
    print(best_std)
    print(best_parameters)
    print(test_optimized)

# --- Proses dimulai dari sini---
# Buka dataset
# Dataset di sini asumsinya berasal dari berkas csv Excel, sehingga separatornya menggunakan tanda ";", bukan ","
# Untuk mengecek apakah dataset menggunakan ";" atau "," kita bisa membuka berkas CSV di notepad
dataset = pd.read_csv('data.csv', sep=';')

# X adalah Independent Variable, y adalah Dependent Variable
# Maksud 1:30 di sini adalah X membaca variabel pada kolom 2 hingga 30. Kolom 1 adalah nama berkas citra.
# Variabel y mengambil variabel di kolom 31
# Di Python indeks dimulai dari angka 0, bukan 1
X = dataset.iloc[:, 1:30].values
y = dataset.iloc[:, 30].values

# Pisahkan dataset menjadi data latih dan uji dengan perbandingan 80:20
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# Penyekalaan fitur dilakukan pada data latih dan uji.
# Detail kenapa ini penting dijelaskan di sini: http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Fungsi fit_transform hanya dilakukan di data latih
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Konfigurasi penyekalaan fitur yang dihasilkan (sc) disimpan dalam berkas, agar sewaktu-waktu bisa langsung digunakan
pickle.dump(sc, open(filename_scale, 'wb'))

# Pra-proses fitur-fitur. Ada empat jenis, yaitu Noproc, PCA, LDA, dan KPCA
# Pra-proses mempermudah agar fitur-fitur lebih ringkas dan mempercepat proses kecerdasan buatan dalam mencapai hasil yang baik
# Kalau noproc, tandanya fitur diolah
# PCA --> http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# LDA --> http://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html
# KPCA --> http://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html
preprocess = ['noproc', 'pca', 'lda', 'kpca']

# Iterasi dilakukan untuk keempat proses
# Loop (1)
for i in preprocess:
    if i == 'noproc':
        pass
    elif i == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components = 10)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        explained_variance = pca.explained_variance_ratio_
        pickle.dump(pca, open(filename_dr_pca, 'wb'))
    elif i == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components= 5, )
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)
        pickle.dump(lda, open(filename_dr_lda, 'wb'))
    else:
        from sklearn.decomposition import KernelPCA
        kpca = KernelPCA(n_components= 10, kernel='rbf')
        X_train = kpca.fit_transform(X_train)
        X_test = kpca.transform(X_test)
        pickle.dump(kpca, open(filename_dr_kpca, 'wb'))

    # Latih kecerdasan buatan dengan data latih, contoh menggunakan SVM C
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)

    # Lakukan proses cross-validation untuk mengetahui seberapa konsisten kemampuan kecerdasan buatan dengan data latih
    # yang berbeda-beda subset nya. Dalam contoh ini dilakukan 10 kali cross-validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

    # Catat rerata akurasi dan standar deviasi 10 percobaan pelatihan
    avg_accuracy = accuracies.mean()
    std_accuracy = accuracies.std()
    print('Akurasi: ', avg_accuracy)
    print('SD: ', std_accuracy)

    # Cek kemampuan kecerdasan buatan dengan data uji 
    test_accuracy = classifier.score(X_test, y_test)
    print('Akurasi Tes:', test_accuracy)

    # Buat Confusion Matrix
    # Detail di sini --> https://en.wikipedia.org/wiki/Confusion_matrix
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    # Masih dalam loop pra-proses (1), lakukan proses grid searching menggunakan fungsi pada baris 34
    if __name__ == '__main__':
        cariGrid(classifier, i, X_train, y_train, X_test, y_test, avg_accuracy, std_accuracy, test_accuracy)