import random
import warnings
import numpy as np
from time import time
from sklearn.svm import LinearSVC
warnings.filterwarnings("ignore")
from evaluation.helpers import *
from onlinelearning.ensemble import *
from onlinelearning.online_learning import *
from semi_supervised.semiSupervised import *
from em.trapezoidal_expectation_maximization2 import TrapezoidalExpectationMaximization2

if __name__ == "__main__":
    # dataset: wpbc; ionosphere; wdbc; australian; credit; wbc; diabetes; dna; german; splice; kr_vs_kp; magic04; a8a; stream
    dataset = "australian"

    batch_c = 8

    contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change, decay_choice, isshuffle = \
        get_tra_hyperparameter(dataset)

    if dataset == "magic04":
        all_cont_indices = np.array([False] * 3 + [True] * 3 + [False] * 4)
        all_ord_indices = np.array([True] * 3 + [False] * 3 + [True] * 4)

        MASK_NUM = 1
        file1 = open("../dataset/MaskData/" + dataset + "/trapezoid/X_trapezoid_new.txt", 'r')
        X_zero = pd.read_csv("../dataset/MaskData/" + dataset + "/trapezoid/X_trapezoid_zeros_new.txt",
                             sep=" ", header=None)
        X_zero = X_zero.fillna(0)
        Y_label = pd.read_csv("../dataset/DataLabel/" + dataset + "/Y_label_new.txt", sep=' ', header=None)

    elif dataset == "stream":
        all_cont_indices = np.array([False] * 300 + [True] * 400 + [False] * 300)
        all_ord_indices = np.array([True] * 300 + [False] * 400 + [True] * 300)

        MASK_NUM = 1
        file1 = open("../dataset/MaskData/" + dataset + "/trapezoid/steamData_X_trapezoid.txt", 'r')
        X_zero = pd.read_csv("../dataset/MaskData/" + dataset + "/trapezoid/steamData_X_trapezoid_zeros.txt",
                             sep=" ", header=None)
        X_zero = X_zero.fillna(0)
        Y_label = pd.read_csv("../dataset/DataLabel/" + dataset + "/Y_label.txt", sep=' ', header=None)

    else:
        all_cont_indices = np.array([False] * 11 + [True] * 12 + [False] * 11)
        all_ord_indices = np.array([True] * 11  + [False] * 12 + [True] * 11)

        MASK_NUM = 1
        file1 = open("../dataset/MaskData/" + dataset + "/trapezoid/X_trapezoid.txt", 'r')
        # file1 = file1.fillna(0)
        X_zero = pd.read_csv("../dataset/MaskData/" + dataset + "/trapezoid/X_trapezoid_zeros.txt", sep = " ",
                             header = None)
        X_zero = X_zero.fillna(0)
        Y_label = pd.read_csv("../dataset/DataLabel/" + dataset + "/Y_label.txt", sep = ' ', header = None)

    shuffle = isshuffle
    Y_label_masked = random.sample(range(1, Y_label.shape[0]), int(Y_label.shape[0] * 0.5))
    Y_label_masked.sort()
    Y_label_masked = np.array(Y_label_masked)
    X_masked = mask_types(X_zero, MASK_NUM, seed = 1)  # arbitrary setting Nan

    Y_label = Y_label.values
    Y_label = Y_label.flatten()

    X_masked = file1.readlines()
    X_zero = np.array((X_zero))
    n = len(X_masked)
    X_masked = chack_Nan(X_masked, n)

    #getting the hyperparameter
    BATCH_SIZE = math.ceil(n / batch_size_denominator)
    WINDOW_SIZE = math.ceil(n / window_size_denominator)
    WINDOW_WIDTH = len(X_masked[BATCH_SIZE])
    cont_indices = all_cont_indices[ :WINDOW_WIDTH]
    ord_indices = all_ord_indices[ :WINDOW_WIDTH]

    #starting trapezoidale imputation
    tra = TrapezoidalExpectationMaximization2(cont_indices, ord_indices, window_size = WINDOW_SIZE,
                                              window_width = WINDOW_WIDTH)
    j = 0
    X_imp = []
    Z_imp = []

    Y_label_fill_x = np.empty(Y_label.shape)
    Y_label_fill_z = np.empty(Y_label.shape)
    Y_label_fill_x_ensemble = np.empty(Y_label.shape)
    Y_label_fill_z_ensemble = np.empty(Y_label.shape)

    start = 0
    end = BATCH_SIZE
    WINDOW_WIDTH = len(X_masked[0])

    clf1  = LinearSVC(random_state=0, tol=1e-5)
    clf2  = LinearSVC(random_state=0, tol=1e-5)
    clf_x = LinearSVC(random_state=0, tol=1e-5)
    clf_z = LinearSVC(random_state=0, tol=1e-5)

    while end <= n:
        X_batch = X_masked[start:end]
        if decay_coef_change == 1:
            this_decay_coef = batch_c / (j + batch_c)
        else:
            this_decay_coef = 0.5
        if len(X_batch[-1]) > WINDOW_WIDTH:
            WINDOW_WIDTH = len(X_batch[-1])
            cont_indices = all_cont_indices[:WINDOW_WIDTH]
            ord_indices = all_ord_indices[:WINDOW_WIDTH]

        for i, row in enumerate(X_batch):
            now_width = len(row)
            if now_width < WINDOW_WIDTH:
                row = row + [np.nan for i in range(WINDOW_WIDTH - now_width)]
                X_batch[i] = row
        X_batch = np.array(X_batch)

        where_are_NaNs = np.isnan(X_batch)
        X_batch[where_are_NaNs] = 0

        Z_imp_batch, X_imp_batch = tra.partial_fit_and_predict(X_batch, cont_indices, ord_indices,
                                                               max_workers = 1, decay_coef = 0.5)
        Z_imp.append(Z_imp_batch[0].tolist())
        X_imp.append(X_imp_batch[0].tolist())

        if start == 0:
            train_x, label_train_x, initial_label_x = X_imp_batch, Y_label[start : end], Y_label_masked[
                (Y_label_masked > start) & (Y_label_masked < end)]
            train_z, label_train_z, initial_label_z = Z_imp_batch, Y_label[start : end], Y_label_masked[
                (Y_label_masked > start) & (Y_label_masked < end)]
        else:
            train_x, label_train_x, initial_label_x = X_imp_batch, Y_label[start : end], Y_label_masked[
                (Y_label_masked > start) & (Y_label_masked < end)] % start
            train_z, label_train_z, initial_label_z = Z_imp_batch, Y_label[start : end], Y_label_masked[
                (Y_label_masked > start) & (Y_label_masked < end)] % start

        percent = 5

        nneigh_x = DensityPeaks(train_x, percent)
        nneigh_z = DensityPeaks(train_z, percent)

        predict_label_train_x_ensemble = SSC_DensityPeaks_SVC_ensemble(train_x, label_train_x, train_z, label_train_z, initial_label_x, nneigh_x, nneigh_z, clf1, clf2)
        predict_label_train_z_ensemble = SSC_DensityPeaks_SVC_ensemble(train_z, label_train_z, train_x, label_train_x, initial_label_z, nneigh_z, nneigh_x, clf1, clf2)
        Y_label_fill_x_ensemble[start : end] = predict_label_train_x_ensemble
        Y_label_fill_z_ensemble[start : end] = predict_label_train_z_ensemble


        predict_label_train_x = SSC_DensityPeaks_SVC(train_x, label_train_x, initial_label_x, nneigh_x, clf_x)
        Y_label_fill_x[start: end] = predict_label_train_x

        predict_label_train_z = SSC_DensityPeaks_SVC(train_z, label_train_z, initial_label_z, nneigh_z, clf_z)
        Y_label_fill_z[start : end] = predict_label_train_z

        start = start + 1
        end = start + BATCH_SIZE
    for i in range(1, BATCH_SIZE):
        Z_imp.append(Z_imp_batch[i].tolist())
        X_imp.append(X_imp_batch[i].tolist())

    #getting the CER
    X_input1 = Z_imp
    X_input2 = X_zero
    temp = np.ones((n, 1))
    X_input2 = np.hstack((temp, X_input2))

    X_Zero_CER_fill = generate_X_Y_trap(n, X_input2, Y_label_fill_x, Y_label, decay_choice, contribute_error_rate)
    X_Zero_CER = generate_Xmask(n, X_input2, Y_label, Y_label_masked, decay_choice, contribute_error_rate)
    Z_impl_CER = generate_Xmask_trap(n, X_input1, Y_label, Y_label_masked, decay_choice, contribute_error_rate)

    Z_impl_CER_fill = generate_X_Y_trap(n, X_input1, Y_label_fill_z, Y_label, decay_choice, contribute_error_rate)

    ensemble_XZ_imp_CER_fill, lamda_array_XZ_imp_CER_fill   = ensemble_Y_trap(n, X_input1, X_input2, Y_label, Y_label_fill_x, decay_choice, contribute_error_rate)
    ensemble_XZ_imp_CER  = ensemble_Xmask(n, X_input1, X_input2, Y_label, Y_label_masked, decay_choice, contribute_error_rate)

    draw_cap_error_picture_tra(ensemble_XZ_imp_CER_fill,
                               X_Zero_CER_fill,
                               ensemble_XZ_imp_CER,
                               X_Zero_CER,
                               Z_impl_CER,
                               Z_impl_CER_fill,
                               dataset)

