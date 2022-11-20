import numpy as np
from source.onlinelearning.ftrl_adp import FTRL_ADP

def ensemble(n, X_input, Z_input, Y_label , decay_choice, contribute_error_rate):
    errors=[]
    lamda_array = []

    classifier_X = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])
    classifier_Z = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = Z_input.shape[1])

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        if np.isnan(Y_label[row]):
            continue
        else:
            x = X_input[row]
            y = Y_label[row]
            p_x, decay_x, loss_x, w_x = classifier_X.fit(indices, x, y ,decay_choice,contribute_error_rate)

            z = Z_input[row]
            p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)

            p = sigmoid(lamda * np.dot(w_x,x) + ( 1.0 - lamda ) * np.dot(w_z,z))

            x_loss += loss_x
            z_loss += loss_z
            lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))

            lamda_array.append(lamda)

            error = [int(np.abs(y - p) > 0.5)]
            errors.append(error)
    lamda_array.savetxt()
    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return ensemble_error

def ensemble_Xmask(n, X_input, Z_input, Y_label, Y_label_masked, decay_choice, contribute_error_rate):
    predict_x = []
    predict_y = []
    lamda_array = []
    errors=[]

    classifier_X = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = len(X_input[-1]))
    classifier_Z = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = len(Z_input[-1]))

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001
    for row in range(n):
        indices_x = [i for i in range(len(X_input[row]))]
        indices_z = [i for i in range(len(Z_input[row]))]
        if row in Y_label_masked:
            x = X_input[row]
            z = Z_input[row]
            y = Y_label[row]
            y_not = 100  # 停止更新，返回两个数值（一定要注意这个位置）
            p_x, w_x = classifier_X.fit(indices_x, x, y_not, decay_choice, contribute_error_rate)
            p_z, w_z = classifier_Z.fit(indices_z, z, y_not, decay_choice, contribute_error_rate)

            p = sigmoid(lamda * np.dot(w_x, x) + (1.0 - lamda) * np.dot(w_z, z))

            error = [int(np.abs(y - p) > 0.5)]
            errors.append(error)

        else:
            # 进行更新，并且会更新lambda
            x = X_input[row]
            y = Y_label[row]
            p_x, decay_x, loss_x, w_x = classifier_X.fit(indices_x, x, y ,decay_choice,contribute_error_rate)

            z = Z_input[row]
            p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices_z, z, y, decay_choice, contribute_error_rate)

            p = sigmoid(lamda * np.dot(w_x, x) + ( 1.0 - lamda ) * np.dot(w_z, z))

            x_loss += loss_x
            z_loss += loss_z
            lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))

            error = [int(np.abs(y - p) > 0.5)]
            errors.append(error)
    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return ensemble_error

def ensemble_Xmask_trap(n, X_input, Z_input, Y_label, Y_label_masked, decay_choice, contribute_error_rate):
    predict_x = []
    predict_y = []
    errors=[]

    classifier_X = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])
    classifier_Z = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = Z_input.shape[1])

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001
    for row in range(n):
        indices = [i for i in range(len(X_input[row]))]
        if row in Y_label_masked:
            x = np.array(X_input[row]).data
            z = np.array(Z_input[row]).data
            y = Y_label[row]
            y_not = 100
            p_x, w_x = classifier_X.fit(indices, x, y_not, decay_choice, contribute_error_rate)
            p_z, w_z = classifier_Z.fit(indices, z, y_not, decay_choice, contribute_error_rate)

            p = sigmoid(lamda * np.dot(w_x, x) + (1.0 - lamda) * np.dot(w_z, z))

            error = [int(np.abs(y - p) > 0.5)]
            errors.append(error)

        else:
            x = X_input[row]
            y = Y_label[row]
            p_x, decay_x, loss_x, w_x = classifier_X.fit(indices, x, y ,decay_choice,contribute_error_rate)

            z = Z_input[row]
            p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)

            p = sigmoid(lamda * np.dot(w_x, x) + ( 1.0 - lamda ) * np.dot(w_z, z))

            x_loss += loss_x
            z_loss += loss_z
            lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))

            error = [int(np.abs(y - p) > 0.5)]
            errors.append(error)
    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return ensemble_error


def ensemble_Y(n, X_input, Z_input, Y_label, Y_label_fill_x, decay_choice, contribute_error_rate):
    errors=[]
    lamda_array = []

    classifier_X = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])
    classifier_Z = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = Z_input.shape[1])

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label_fill_x[row]
        p_x, decay_x, loss_x, w_x = classifier_X.fit(indices, x, y ,decay_choice,contribute_error_rate)

        z = Z_input[row]
        p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)

        p = sigmoid(lamda * np.dot(w_x, x) + ( 1.0 - lamda ) * np.dot(w_z, z))

        x_loss += loss_x
        z_loss += loss_z
        lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))
        lamda_array.append(lamda)

        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)
    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return ensemble_error, lamda_array

def ensemble_Y_trap(n, X_input, Z_input, Y_label, Y_label_fill_x, decay_choice, contribute_error_rate):
    errors=[]
    lamda_array = []

    classifier_X = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = len(X_input[-1]))
    classifier_Z = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = len(Z_input[-1]))

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001
    for row in range(n):
        indices_x = [i for i in range(len(X_input[row]))]
        indeces_z = [i for i in range(len(Z_input[row]))]
        x = np.array(X_input[row]).data
        y = Y_label_fill_x[row]
        p_x, decay_x, loss_x, w_x = classifier_X.fit(indices_x, x, y ,decay_choice,contribute_error_rate)

        z = np.array(Z_input[row]).data
        p_z, decay_z, loss_z, w_z = classifier_Z.fit(indeces_z, z, y, decay_choice, contribute_error_rate)

        p = sigmoid(lamda * np.dot(w_x, x) + ( 1.0 - lamda ) * np.dot(w_z, z))

        x_loss += loss_x
        z_loss += loss_z
        lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))
        lamda_array.append(lamda)

        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)
    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return ensemble_error, lamda_array

def logistic_loss(p,y):
    return (1 / np.log(2.0)) * (-y * np.log(p) - (1 - y) * np.log(1 - p))

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))