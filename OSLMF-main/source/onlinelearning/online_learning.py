import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from source.onlinelearning.ftrl_adp import *
from source.onlinelearning.fobos import FOBOS
from sklearn import svm
import math

def svm_classifier(train_x, train_y, test_x, test_y):
    best_score = 0
    best_C = -1
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        clf = svm.LinearSVC(C = C, max_iter = 100000)
        clf.fit(train_x,train_y)
        score = clf.score(test_x, test_y)
        if score > best_score:
            best_score = score
            best_C = C
    return  best_score, best_C

def calculate_svm_error(X_input, Y_label,n):
    length = int(0.7*n)
    X_train = X_input[:length, :]
    Y_train = Y_label[:length]
    X_test = X_input[length:, :]
    Y_test = Y_label[length:]
    best_score, best_C = svm_classifier(X_train, Y_train, X_test, Y_test)
    error = 1.0 - best_score
    return error, best_C

def generate_Xmask(n, X_input, Y_label, Y_label_masked, decay_choice, contribute_error_rate):
    errors  = []
    decays  = []
    predict = []
    mse     = []

    classifier = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        if row in Y_label_masked:
            x = X_input[row]
            y = Y_label[row]
            y_not = 100
            p, w = classifier.fit(indices, x, y_not, decay_choice, contribute_error_rate)
            error = [int(np.abs(y - p) > 0.5)]

            errors.append(error)
            predict.append(p)
        else:
            x = X_input[row]
            y = Y_label[row]
            p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
            error = [int(np.abs(y - p) > 0.5)]

            errors.append(error)
            decays.append(decay)
            predict.append(p)

    X_Zero_CER = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    # svm_error, _ = calculate_svm_error(X_input[:, 1:], Y_label, n)

    return X_Zero_CER#, svm_error

def generate_Xmask_trap(n, X_input, Y_label, Y_label_masked, decay_choice, contribute_error_rate):
    errors  = []
    decays  = []
    predict = []
    mse     = []

    classifier = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = len(X_input[-1]))
    for row in range(n):
        indices = [i for i in range(len(X_input[row]))]
        if row in Y_label_masked:
            x = np.array(X_input[row]).data
            y = Y_label[row]
            y_not = 100
            p, w = classifier.fit(indices, x, y_not, decay_choice, contribute_error_rate)
            error = [int(np.abs(y - p) > 0.5)]

            errors.append(error)
            predict.append(p)
        else:
            x = X_input[row]
            y = Y_label[row]
            p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
            error = [int(np.abs(y - p) > 0.5)]

            errors.append(error)
            decays.append(decay)
            predict.append(p)

    X_Zero_CER = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    # svm_error, _ = calculate_svm_error(X_input[:, 1:], Y_label, n)

    return X_Zero_CER#, svm_error

def generate_X_Y(n, X_input, Y_label_fill_x, Y_label, decay_choice, contribute_error_rate):
    errors  = []
    decays  = []
    predict = []
    mse     = []

    classifier = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label_fill_x[row]
        p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row+1], Y_label[:row+1]))

    X_Zero_CER  = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    svm_error,_ = calculate_svm_error(X_input[:,1:], Y_label, n)

    return X_Zero_CER, svm_error

def generate_X_Y_trap(n, X_input, Y_label_fill_x, Y_label, decay_choice, contribute_error_rate):
    errors  = []
    decays  = []
    predict = []
    mse     = []

    classifier = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = len(X_input[-1]))
    for row in range(n):
        indices = [i for i in range(len(X_input[row]))]
        x = np.array(X_input[row]).data
        y = Y_label_fill_x[row]
        p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row+1], Y_label[:row+1]))

    X_Zero_CER  = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return X_Zero_CER

def generate_Z(n, X_input, Y_label_fill_z, Y_label, decay_choice, contribute_error_rate):
    errors  = []
    decays  = []
    predict = []
    mse     = []

    classifier = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label_fill_z[row]
        p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row+1], Y_label[:row+1]))

    X_Zero_CER  = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    svm_error,_ = calculate_svm_error(X_input[:,1:], Y_label, n)

    return X_Zero_CER, svm_error

def generate_cap(n, X_input, Y_label, decay_choice, contribute_error_rate):
    errors=[]
    decays=[]

    classifier = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])

    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row].data
        y = Y_label[row]
        p, decay,loss,w= classifier.fit(indices, x, y ,decay_choice,contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)

    Z_imp_CER = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return Z_imp_CER

def generate_tra(n, X_input, Y_label, decay_choice, contribute_error_rate):
    errors = []
    classifier = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=len(X_input[-1]))
    for row in range(n):
        indices = [i for i in range(len(X_input[row]))]
        x = np.array(X_input[row]).data
        y = Y_label[row]
        p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)
    imp_CER = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return imp_CER

def draw_cap_error_picture(ensemble_XZ_imp_CER_fill, X_Zero_CER_fill, ensemble_XZ_imp_CER, X_Zero_CER, Z_impl_CER,
                           Z_impl_CER_fill, svm_error, dataset):
    n = len(ensemble_XZ_imp_CER)
    plt.figure(figsize=(16, 10))
    plt.ylim((0, 1))
    plt.xlim((0, n))
    plt.ylabel("CER")  #
    x = range(n)

    Z_imp_CER_fill,  = plt.plot(x, ensemble_XZ_imp_CER_fill, color = 'green',    linestyle = "--")    # the error of z_imp
    X_Zero_CER_fill, = plt.plot(x, X_Zero_CER_fill,          color = 'blue',     linestyle = "-")      # the error of x_zero
    Z_imp_CER,       = plt.plot(x, ensemble_XZ_imp_CER,      color = 'magenta',  linestyle = "-.")    # the error of z_imp
    X_Zero_CER,      = plt.plot(x, X_Zero_CER,               color = 'black',    linestyle = ":")      # the error of x_zero
    Z_impl_CER,      = plt.plot(x, Z_impl_CER,               color = 'pink',     linestyle = "solid")      # the error of x_zero
    Z_impl_CER_fill, = plt.plot(x, Z_impl_CER_fill,          color = 'cyan')      # the error of x_zero

    svm_error, = plt.plot(x, [svm_error] * n ,color='red')   # the error of svm

    plt.legend(handles=[Z_imp_CER_fill,             X_Zero_CER_fill,   Z_imp_CER,             X_Zero_CER,   Z_impl_CER,  Z_impl_CER_fill,   svm_error],
               labels=["ensemble_XZ_imp_CER_fill", "X_Zero_CER_fill", "ensemble_XZ_imp_CER", "X_Zero_CER", "Z_impl_CER", "Z_impl_CER_fill", "svm_error"])

    plt.title(dataset + "_The Cumulative error rate(CER) of ensemble_XZ_imp_CER_fill, X_Zero_CER_fill, ensemble_XZ_imp_CER, "
                        "X_Zero_CER, Z_impl_CER, Z_impl_CER_fill, SVM_CER")
    plt.show()
    # plt.clf()

def draw_cap_error_picture_tra(ensemble_XZ_imp_CER_fill, X_Zero_CER_fill, ensemble_XZ_imp_CER, X_Zero_CER, Z_impl_CER, Z_impl_CER_fill, dataset):
    n = len(ensemble_XZ_imp_CER)
    plt.figure(figsize=(16, 10))
    plt.ylim((0, 1))
    plt.xlim((0, n))
    plt.ylabel("CER")
    x = range(n)

    Z_imp_CER_fill,      = plt.plot(x, ensemble_XZ_imp_CER_fill, color = 'green',    linestyle = "--")    # the error of z_imp
    X_Zero_CER_fill,     = plt.plot(x, X_Zero_CER_fill,          color = 'blue',     linestyle = "-")      # the error of x_zero
    ensemble_XZ_imp_CER, = plt.plot(x, ensemble_XZ_imp_CER,      color = 'magenta',  linestyle = "-.")    # the error of z_imp
    X_Zero_CER,          = plt.plot(x, X_Zero_CER,               color = 'black',    linestyle = ":")      # the error of x_zero
    Z_impl_CER,          = plt.plot(x, Z_impl_CER,               color = 'pink',     linestyle = "solid")      # the error of x_zero
    Z_impl_CER_fill,     = plt.plot(x, Z_impl_CER_fill,          color = 'cyan')      # the error of x_zero

    plt.legend(handles=[Z_imp_CER_fill, X_Zero_CER_fill, ensemble_XZ_imp_CER, X_Zero_CER, Z_impl_CER, Z_impl_CER_fill],
               labels=["ensemble_XZ_imp_CER_fill", "X_Zero_CER_fill", "ensemble_XZ_imp_CER", "X_Zero_CER", "Z_impl_CER", "Z_impl_CER_fill"])

    plt.title(dataset + "_The Cumulative error rate(CER) of ensemble_XZ_imp_CER_fill, X_Zero_CER_fill, ensemble_XZ_imp_CER, X_Zero_CER, Z_impl_CER, Z_impl_CER_fill")
    plt.show()

def draw_cap_error_picture_1(Z_imp_CER, X_Zero_CER,):
    n = len(Z_imp_CER)
    plt.figure(figsize=(16, 10))
    plt.ylim((0, 1))
    plt.xlim((0, n))
    plt.ylabel("CER")
    x = range(n)

    Z_imp_CER,  = plt.plot(x, Z_imp_CER  , color='green')
    X_Zero_CER, = plt.plot(x, X_Zero_CER, color='blue')

    plt.legend(handles=[Z_imp_CER, X_Zero_CER], labels=["Z_imp_CER", "X_Zero_CER"])

    plt.title("The Cumulative error rate(CER) of z_imp_CER, x_zero_CER")
    plt.show()
    plt.clf()

def draw_tra_error_picture(error_arr_Z, error_arr_X):
    n = len(error_arr_Z)
    plt.figure(figsize=(16, 10))
    plt.ylim((0, 1.0))
    plt.xlim((0, n))
    plt.ylabel("CER")  #

    x = range(n)
    error_arr_Z, = plt.plot(x, error_arr_Z, color='green')
    error_arr_X, = plt.plot(x, error_arr_X, color='blue')

    plt.legend(handles=[error_arr_Z, error_arr_X], labels=["error_arr_Z", "error_arr_X"])

    plt.title("The CER of trapezoid data stream")
    plt.show()
    plt.clf()


