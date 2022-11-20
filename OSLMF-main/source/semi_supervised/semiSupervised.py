# *_* coding : utf-8 *_*
import itertools
import numpy as np
from sklearn import neighbors
from source.onlinelearning.ftrl_adp import *

'''
    Function to calculate the distance between two data points of a matrix, applicable to density peak distance discovery data structures
    Input: data matrix Dots, number of rows NDim, number of columns numOfDots
    Output: distance matrix DistanceMat, first column is data point i, second column is another data point j, third column is the distance between ij
'''
def list_of_groups(list_info, per_list_len):
    list_of_group = zip(*(iter(list_info),) *per_list_len)
    end_list = [list(i) for i in list_of_group] # i is a tuple
    count = len(list_info) % per_list_len
    end_list.append(list_info[-count:]) if count !=0 else end_list
    return end_list

def PairDotsDistance(Dots, NDim, numOfDots):
    Len = numOfDots * (numOfDots - 1) / 2
    DistanceMat = []
    matIndex = 0
    for i in range(0, numOfDots - 1):
        for j in range(i + 1, numOfDots):
            DistanceMat.append(i)
            DistanceMat.append(j)
            DistanceMat.append(np.sqrt(np.sum((Dots[:, i] - Dots[:, j])**2)))
        matIndex = matIndex + 1
    Distance = np.array(list_of_groups(DistanceMat, 3))
    return Distance

'''
    Calculate peak data density and find data pointing to structural relationships
    Input: data xi, percentage of neighbours per cent.
    Output: data nneigh for surface data pointing to structure.
'''
def DensityPeaks(xi, precent):
    rowN, colN = xi.shape[0], xi.shape[1]
    xiT = xi.T
    xx = PairDotsDistance(xiT, colN, rowN)
    xxT = xx
    ND = np.max(xxT[:, 1]) + 1
    NL = np.max(xxT[:, 0]) + 1

    if NL > ND:
        ND = NL
    N = xxT.shape[0]

    dist = np.zeros((int(ND),int(ND)))

    for i in range(0, N):
        ii = xxT[i, 0]
        jj = xxT[i, 1]

        dist[int(ii), int(jj)] = xxT[i, 2]
        dist[int(jj), int(ii)] = xxT[i, 2]

    # print('average percentage of neighbours (hard coded): %5.6f\n', precent)

    position = round(N * precent / 100, 0)

    sda = np.sort(xxT[:, 2])

    dc = sda[int(position)]

    rho = np.zeros((1, int(ND))) # [i for i in range(1, ND)]

    for i in range(0, int(ND) - 1):
        for j in range(i + 1, int(ND)):
            rho[0, i] = rho[0, i] + np.math.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))
            rho[0, j] = rho[0, j] + np.math.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))

    maxd = max(np.max(dist, axis=0))

    rho_sorted = np.sort(rho, axis=1)

    rho_list = list(rho_sorted)
    rho_list = list(itertools.chain.from_iterable(rho_list))
    rho_list.reverse()
    rho_list = np.array(rho_list)
    rho_list = rho_list.reshape(-1, rho_list.shape[0])
    rho_sorted = rho_list
    ordrho = np.argsort(-rho,axis=1)

    delta = np.zeros((int(ND)))
    delta[ordrho[0,0] - 1] = -1.
    delta = delta.reshape(-1, delta.shape[0])

    nneigh = np.zeros((int(ND)))
    nneigh[ordrho[0,0] - 1] = 0
    nneigh = nneigh.reshape(-1, nneigh.shape[0])

    for ii in range(1, int(ND)):
        delta[0, ordrho[0, ii]] = maxd
        for jj in range(0, ii - 1):
            if (dist[ordrho[0, ii], ordrho[0, jj]] < delta[0, ordrho[0, ii]]):
                delta[0, ordrho[0,ii]] = dist[ordrho[0, ii], ordrho[0, jj]]
                nneigh[0, ordrho[0,ii]] = ordrho[0, jj]

    delta[0, ordrho[0, 0]] = maxd

    rho_sorted = rho_sorted.T
    rho = rho.T
    ordrho = ordrho.T
    nneigh = nneigh.T
    delta = delta.T

    b = np.argmin(nneigh)
    nneigh[b,0] = b

    return nneigh

'''
    Semi-supervised classification based on density peaks, KNN method;
    Inputs: training set xi labelable_xi; test set test labelable_test; number of classes class_n;
    Ratio of each class collected,0-1; K values of K nearest neighbors; Pointing graph relation nneigh after density peaking calculation;
    t is the initial label point index;
    Output: classification accuracyaccuracy, predict labelpredict_label;
'''
def SSC_DensityPeaks_KNN(train, label_train, t, K, nneigh): # , test, label_test #, clf, n, decay_choice, contribute_error_rate
    data = []
    label_data = []
    label_U = []

    clf = neighbors.KNeighborsClassifier(K, weights='distance', algorithm='brute')

    data.append(train[t-1,]) # t-1
    data = np.array(data)
    data = data.reshape(data.shape[1],data.shape[2])

    label_data.append(label_train[t-1,]) # t-1
    label_data = np.array(label_data)
    label_data = label_data.reshape(label_data.shape[1], label_data.shape[0])
    struct = t-1

    struct_record = struct
    length_data = len(data)
    length_struct = len(struct)
    diff_array = np.array([i for i in range(0,data.shape[0],1)])
    t_U = np.setdiff1d(diff_array, t-1)
    label_U.append(label_train[t_U])
    label_U = np.array(label_U)
    label_U = label_U.reshape(label_U.shape[1], label_U.shape[0])

    data_neigh = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        for i in range(0, len(struct),1 ):
            data_neigh.append(nneigh[int(struct[i])])
        data_neigh = np.array(data_neigh)

        struct = np.setdiff1d(data_neigh, struct_record)
        length_struct = len(struct)

        for i in range(0, length_struct):
            struct_record = np.append(struct_record, struct[i])
        struct_record = struct_record.reshape(struct_record.shape[0],1)

        data_TR = data
        lable_TR = label_data
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)

        clf.fit(data_TR, lable_TR)
        label_data = clf.predict(data)

    struct = struct_record
    length_struct = len(struct)
    del data_neigh

    def find(condition):
        res = np.nonzero(condition)
        return res
    data_neigh = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        k = 0
        for i in range(0, len(struct), 1):
            number_neigh_test = np.where(nneigh == struct[i])[0]
            number_neigh = find(nneigh == struct[i])[0]
            number_neigh = np.array(number_neigh)
            length_neigh = len(number_neigh)
            if len(struct) > 0:
                for j in range(0, length_neigh):
                    data_neigh.append(number_neigh[j])
            k = k + length_neigh
        data_neigh = np.array(data_neigh)

        struct = np.setdiff1d(data_neigh, struct_record)
        length_struct = len(struct)

        struct_record = list(struct_record)
        for i in range(0, length_struct):
            struct_record.append(struct[i])
        struct_record = np.array(struct_record)

        data_TR = data
        lable_TR = label_data
        data = list(data)
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)

        clf.fit(data_TR, lable_TR)
        label_data = clf.predict(data)

    clf.fit(data, label_data)
    predict_label_train = clf.predict(train)
    predict_label_train = []

    predict_label_train = np.array(predict_label_train)

    return predict_label_train

def SSC_DensityPeaks_FTRL(train, label_train, t, K, nneigh, decay_choice, contribute_error_rate, classifier): # , test, label_test #, clf, n, decay_choice, contribute_error_rate
    data = []
    label_data = []
    label_U = []

    classifier = classifier

    data.append(train[t-1,]) # t-1
    data = np.array(data)
    data = data.reshape(data.shape[1],data.shape[2])

    label_data.append(label_train[t-1,]) # t-1
    label_data = np.array(label_data)
    label_data = label_data.reshape(label_data.shape[1], label_data.shape[0])
    struct = t-1

    struct_record = struct
    length_data = len(data)
    length_struct = len(struct)
    diff_array = np.array([i for i in range(0,data.shape[0],1)])
    t_U = np.setdiff1d(diff_array, t-1)
    label_U.append(label_train[t_U])
    label_U = np.array(label_U)
    label_U = label_U.reshape(label_U.shape[1], label_U.shape[0])

    '''Select first to point to the next node'''
    data_neigh = []
    predict_1 = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        for i in range(0, len(struct),1 ):
            data_neigh.append(nneigh[int(struct[i])])
        data_neigh = np.array(data_neigh)

        struct = np.setdiff1d(data_neigh, struct_record)
        length_struct = len(struct)

        for i in range(0, length_struct):
            struct_record = np.append(struct_record, struct[i])
        struct_record = struct_record.reshape(struct_record.shape[0],1)

        data_TR = data
        lable_TR = label_data
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)

        n = len(data_TR)
        for row in range(n):
            indices = [i for i in range(data_TR.shape[1])]
            x = data_TR[row]
            y = lable_TR[row]
            p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
            if p < 0.3:
                p = 0
            else:
                p = 1
            predict_1.append(p)
    struct = struct_record
    length_struct = len(struct)
    del data_neigh

    data_neigh = []
    predict_2 = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        k = 0
        for i in range(0, len(struct), 1):
            number_neigh = np.where(nneigh == struct[i])[0]
            number_neigh = np.array(number_neigh)
            length_neigh = len(number_neigh)
            if len(struct) > 0:
                for j in range(0, length_neigh):
                    data_neigh.append(number_neigh[j])
            k = k + length_neigh
        data_neigh = np.array(data_neigh)

        struct = np.setdiff1d(data_neigh, struct_record)
        length_struct = len(struct)

        struct_record = list(struct_record)
        for i in range(0, length_struct):
            struct_record.append(struct[i])
        struct_record = np.array(struct_record)

        data_TR = data
        lable_TR = label_data
        data = list(data)
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)

        n = len(data_TR)
        for row in range(n):
            indices = [i for i in range(data_TR.shape[1])]
            x = data_TR[row]
            y = lable_TR[row]
            p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
            if p < 0.3:
                p = 0
            else:
                p = 1
            predict_2.append(p)

    predict_label_train = []
    n = len(train)
    for row in range(n):
        indices = [i for i in range(train.shape[1])]
        x = train[row]
        y = label_train[row]
        p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        if p < 0.3:
            p = 0
        else:
            p = 1
        predict_label_train.append(p)

    predict_label_train = np.array(predict_label_train)
    return predict_label_train

def SSC_DensityPeaks_SVC(train, label_train, t, nneigh, clf): # , test, label_test #, clf, n, decay_choice, contribute_error_rate

    data = []
    label_data = []
    label_U = []

    clf = clf

    data.append(train[t-1,]) # t-1
    data = np.array(data)
    data = data.reshape(data.shape[1],data.shape[2])

    label_data.append(label_train[t-1,]) # t-1
    label_data = np.array(label_data)
    label_data = label_data.reshape(label_data.shape[1], label_data.shape[0])
    struct = t-1

    struct_record = struct
    length_data = len(data)
    length_struct = len(struct)
    diff_array = np.array([i for i in range(0,data.shape[0],1)])
    t_U = np.setdiff1d(diff_array, t-1)
    label_U.append(label_train[t_U])
    label_U = np.array(label_U)
    label_U = label_U.reshape(label_U.shape[1], label_U.shape[0])

    data_neigh = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        for i in range(0, len(struct),1 ):
            data_neigh.append(nneigh[int(struct[i])])
        data_neigh = np.array(data_neigh)

        struct = np.setdiff1d(data_neigh, struct_record)
        length_struct = len(struct)

        for i in range(0, length_struct):
            struct_record = np.append(struct_record, struct[i])
        struct_record = struct_record.reshape(struct_record.shape[0],1)

        data_TR = data
        lable_TR = label_data
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)

        try:
            clf.fit(data_TR, lable_TR)
            label_data = clf.predict(data)
        except:
            continue

    struct = struct_record
    length_struct = len(struct)
    del data_neigh

    def find(condition):
        res = np.nonzero(condition)
        return res
    data_neigh = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        k = 0
        for i in range(0, len(struct), 1):
            number_neigh_test = np.where(nneigh == struct[i])[0]
            number_neigh = find(nneigh == struct[i])[0]
            number_neigh = np.array(number_neigh)
            length_neigh = len(number_neigh)
            if len(struct) > 0:
                for j in range(0, length_neigh):
                    data_neigh.append(number_neigh[j])
            k = k + length_neigh
        data_neigh = np.array(data_neigh)

        struct = np.setdiff1d(data_neigh, struct_record)
        length_struct = len(struct)

        struct_record = list(struct_record)
        for i in range(0, length_struct):
            struct_record.append(struct[i])
        struct_record = np.array(struct_record)

        data_TR = data
        lable_TR = label_data
        data = list(data)
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)

        try :
            clf.fit(data_TR, lable_TR)
            label_data = clf.predict(data)
        except:
            continue

    try:
        label_data = label_data.astype('int')
        clf.fit(data, label_data)
    except:
        clf.fit(train[:40], label_train[:40])
    predict_label_train = clf.predict(train)

    return predict_label_train

def SSC_DensityPeaks_SVC_ensemble(train_1, label_train_1, train_2, label_train_2, t, nneigh_x, nneigh_z, clf1, clf2): # , test, label_test #, clf, n, decay_choice, contribute_error_rate

    train_x = train_1
    label_train_x = label_train_1

    train_z = train_2
    label_train_z = label_train_2

    classifier_x = clf1
    classifier_z = clf2

    data_x = []
    label_data_x = []
    label_U_x = []

    data_x.append(train_x[t-1,]) # t-1
    data_x = np.array(data_x)
    data_x = data_x.reshape(data_x.shape[1], data_x.shape[2])

    label_data_x.append(label_train_x[t-1,]) # t-1
    label_data_x = np.array(label_data_x)
    label_data_x = label_data_x.reshape(label_data_x.shape[1], label_data_x.shape[0])
    struct_x = t-1

    struct_record_x = struct_x
    length_data_x = len(data_x)
    length_struct_x = len(struct_x)
    diff_array_x = np.array([i for i in range(0, data_x.shape[0], 1)])
    t_U_x = np.setdiff1d(diff_array_x, t-1)
    if len(t_U_x) != 0:
        label_U_x.append(label_train_x[t_U_x])
        label_U_x = np.array(label_U_x)
        label_U_x = label_U_x.reshape(label_U_x.shape[1], label_U_x.shape[0])

    '''Select first to point to the next node'''
    data_neigh_x = []
    while (len(struct_x) > 0):
        data_neigh_x = list(data_neigh_x)
        for i in range(0, len(struct_x),1 ):
            data_neigh_x.append(nneigh_x[int(struct_x[i])])
        data_neigh_x = np.array(data_neigh_x)

        struct_x = np.setdiff1d(data_neigh_x, struct_record_x)
        length_struct_x = len(struct_x)

        for i in range(0, length_struct_x):
            struct_record_x = np.append(struct_record_x, struct_x[i])
        struct_record_x = struct_record_x.reshape(struct_record_x.shape[0], 1)

        data_TR_x = data_x
        lable_TR_x = label_data_x
        data_list_x = list(data_x)
        label_data_list_x = list(label_data_x)
        for j in range(0, length_struct_x):
            data_list_x.append(train_x[int(struct_x[j])])
            label_data_list_x.append(label_train_x[j])
        data_x = np.array(data_list_x)
        label_data_x = np.array(label_data_list_x)
        length_data_x = len(data_x)

        try:
            classifier_x.fit(data_TR_x, lable_TR_x)
            label_data_x = classifier_x.predict(data_x)
        except:
            continue

    struct_x = struct_record_x
    length_struct_x = len(struct_x)
    del data_neigh_x

    data_neigh_x = []
    while (len(struct_x) > 0):
        data_neigh_x = list(data_neigh_x)
        k_x = 0
        for i in range(0, len(struct_x), 1):
            number_neigh_x = np.where(nneigh_x == struct_x[i])[0]
            number_neigh_x = np.array(number_neigh_x)
            length_neigh_x = len(number_neigh_x)
            if len(struct_x) > 0:
                for j in range(0, length_neigh_x):
                    data_neigh_x.append(number_neigh_x[j])
            k_x = k_x + length_neigh_x
        data_neigh_x = np.array(data_neigh_x)


        struct_x = np.setdiff1d(data_neigh_x, struct_record_x)
        length_struct_x = len(struct_x)

        struct_record_x = list(struct_record_x)
        for i in range(0, length_struct_x):
            struct_record_x.append(struct_x[i])
        struct_record_x = np.array(struct_record_x)

        data_TR_x = data_x
        lable_TR_x = label_data_x
        data_x = list(data_x)
        data_list_x = list(data_x)
        label_data_list_x = list(label_data_x)
        for j in range(0, length_struct_x):
            data_list_x.append(train_x[int(struct_x[j])])
            label_data_list_x.append(label_train_x[j])
        data_x = np.array(data_list_x)
        label_data_x = np.array(label_data_list_x)
        length_data_x = len(data_x)  # 求数据长度

        try:
            classifier_x.fit(data_TR_x, lable_TR_x)
            label_data_x = classifier_x.predict(data_x)
        except:
            continue

    data_z = []
    label_data_z = []
    label_U_z = []

    data_z.append(train_z[t-1,]) # t-1
    data_z = np.array(data_z)
    data_z = data_z.reshape(data_z.shape[1], data_z.shape[2])

    label_data_z.append(label_train_z[t-1,]) # t-1
    label_data_z = np.array(label_data_z)
    label_data_z = label_data_z.reshape(label_data_z.shape[1], label_data_z.shape[0])
    struct_z = t-1

    struct_record_z = struct_z
    length_data_z = len(data_z)
    length_struct_z = len(struct_z)
    diff_array_z = np.array([i for i in range(0, data_z.shape[0], 1)])
    t_U_z = np.setdiff1d(diff_array_z, t-1)
    if len(t_U_z) != 0:
        label_U_z.append(label_train_z[t_U_z])
        label_U_z = np.array(label_U_z)
        label_U_z = label_U_z.reshape(label_U_z.shape[1], label_U_z.shape[0])

    '''Select first to point to the next node'''
    data_neigh_z = []
    while (len(struct_z) > 0):
        data_neigh_z = list(data_neigh_z)
        for i in range(0, len(struct_z),1 ):
            data_neigh_z.append(nneigh_z[int(struct_z[i])])
        data_neigh_z = np.array(data_neigh_z)
        struct_z = np.setdiff1d(data_neigh_z, struct_record_z)
        length_struct_z = len(struct_z)

        for i in range(0, length_struct_z):
            struct_record_z = np.append(struct_record_z, struct_z[i])
        struct_record_z = struct_record_z.reshape(struct_record_z.shape[0], 1)

        data_TR_z = data_z
        lable_TR_z = label_data_z
        data_list_z = list(data_z)
        label_data_list_z = list(label_data_z)
        for j in range(0, length_struct_z):
            data_list_z.append(train_z[int(struct_z[j])])
            label_data_list_z.append(label_train_z[j])
        data_z = np.array(data_list_z)
        label_data_z = np.array(label_data_list_z)
        length_data_z = len(data_z)
        try:
            classifier_z.fit(data_TR_z, lable_TR_z)
            label_data_z = classifier_z.predict(data_z)
        except:
            continue

    struct_z = struct_record_z
    length_struct_z = len(struct_z)
    del data_neigh_z

    data_neigh_z = []
    while (len(struct_z) > 0):
        data_neigh_z = list(data_neigh_z)
        k_z = 0
        for i in range(0, len(struct_z), 1):
            number_neigh_z = np.where(nneigh_z == struct_z[i])[0]
            number_neigh_z = np.array(number_neigh_z)
            length_neigh_z = len(number_neigh_z)
            if len(struct_z) > 0:
                for j in range(0, length_neigh_z):
                    data_neigh_z.append(number_neigh_z[j])
            k_z = k_z + length_neigh_z
        data_neigh_z = np.array(data_neigh_z)

        struct_z = np.setdiff1d(data_neigh_z, struct_record_z)
        length_struct_z = len(struct_z)

        struct_record_z = list(struct_record_z)
        for i in range(0, length_struct_z):
            struct_record_z.append(struct_z[i])
        struct_record_z = np.array(struct_record_z)

        data_TR_z = data_z
        lable_TR_z = label_data_z
        data_z = list(data_z)
        data_list_z = list(data_z)
        label_data_list_z = list(label_data_z)
        for j in range(0, length_struct_z):
            data_list_z.append(train_z[int(struct_z[j])])
            label_data_list_z.append(label_train_z[j])
        data_z = np.array(data_list_z)
        label_data_z = np.array(label_data_list_z)
        length_data_z = len(data_z)

        try:
            classifier_z.fit(data_TR_z, lable_TR_z)
            label_data_z = classifier_z.predict(data_z)
        except:
            continue

    '''
        Ensemble Learning
    '''
    label_data_x = label_data_x.astype('int')
    label_data_z = label_data_z.astype('int')
    if len(data_x) != 0:
        if len(set(label_data_x)) != 1:
            classifier_x.fit(data_x, label_data_x)
    if len(data_z) != 0:
        if len(set(label_data_z)) != 1:
            classifier_z.fit(data_z, label_data_z)

    def sigmoid(x):
        if x >= 0:
            return 1.0 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

    lamda = 0.5
    predict_label = []
    length = len(train_1)
    # print("length", length)
    for n in range(length):
        train_1_1 = train_1[n, ]
        train_2_1 = train_2[n, ]
        try:
            predict_label_train_1 = classifier_x._predict_proba_lr(np.array(train_1_1).reshape(1, -1))
            predict_label_train_1 = np.argmax(predict_label_train_1)
        except IndexError:
            predict_label_train_1 = np.random.randint(0, 2, 1)
        except :
            predict_label_train_1 = np.random.randint(0, 2, 1)

        try:
            predict_label_train_2 = classifier_z._predict_proba_lr(np.array(train_2_1).reshape(1, -1))
            predict_label_train_2 = np.argmax(predict_label_train_2)
        except IndexError:
            predict_label_train_2 = np.random.randint(0, 2, 1)
        except :
            predict_label_train_2 = np.random.randint(0, 2, 1)

        predict_label_train = sigmoid(lamda * predict_label_train_1 + (1.0 - lamda) * predict_label_train_2)
        if predict_label_train > 0.5:
            predict_label_train = 1
        else:
            predict_label_train = 0
        predict_label.append(predict_label_train)

    predict_label = np.array(predict_label)
    predict_label = predict_label.reshape(predict_label.shape[0])
    return predict_label

def SSC_DensityPeaks_KNN_online(n, train, label_train, t, nneigh, decay_choice, contribute_error_rate): # , test, label_test
    data = []
    label_data = []
    label_U = []
    predict = []

    classifier = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs = train.shape[1])

    data.append(train[t-1,]) # t-1
    data = np.array(data)
    data = data.reshape(data.shape[1],data.shape[2])

    label_data.append(label_train[t-1,]) # t-1
    label_data = np.array(label_data)
    label_data = label_data.reshape(label_data.shape[1], label_data.shape[0])
    struct = t-1

    struct_record = struct
    length_data = len(data)
    length_struct = len(struct)
    diff_array = np.array([i for i in range(0,data.shape[0],1)])
    t_U = np.setdiff1d(diff_array, t-1)
    label_U.append(label_train[t_U])
    label_U = np.array(label_U)
    label_U = label_U.reshape(label_U.shape[1], label_U.shape[0])

    data_neigh = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        for i in range(0, len(struct),1 ):
            data_neigh.append(nneigh[int(struct[i])])
        data_neigh = np.array(data_neigh)

        struct = np.setdiff1d(data_neigh, struct_record)
        length_struct = len(struct)

        for i in range(0, length_struct):
            struct_record = np.append(struct_record, struct[i])
        struct_record = struct_record.reshape(struct_record.shape[0],1)

        data_TR = data
        lable_TR = label_data
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)

        for row in range(n):
            indices = [i for i in range(train.shape[1])]
            x = train[row]
            y = label_train[row]
            p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)

    struct = struct_record
    length_struct = len(struct)
    del data_neigh

    def find(condition):
        res = np.nonzero(condition)
        return res
    data_neigh = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        k = 0
        for i in range(0, len(struct), 1):

            number_neigh_test = np.where(nneigh == struct[i])[0]
            number_neigh = find(nneigh == struct[i])
            number_neigh = np.array(number_neigh)
            length_neigh = len(number_neigh)
            if len(struct) > 0:
                for j in range(0, length_neigh):
                    data_neigh.append(number_neigh[j])
            k = k + length_neigh
        data_neigh = np.array(data_neigh)

        struct = np.setdiff1d(data_neigh, struct_record)
        length_struct = len(struct)

        struct_record = list(struct_record)
        for i in range(0, length_struct):
            struct_record.append(struct[i])
        struct_record = np.array(struct_record)

        data_TR = data
        lable_TR = label_data
        data = list(data)
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)

        for row in range(data_TR.shape[0]):
            indices = [i for i in range(train.shape[1])]
            x = data_TR[row]
            y = lable_TR[row]
            p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
            error = [int(np.abs(y - p) > 0.5)]

    for row in range(train.shape[0]):
        indices = [i for i in range(train.shape[1])]
        x = train[row]
        y = label_train[row]
        p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]
        predict.append(p)

    predict_label_train = np.array(predict)

    return predict_label_train
