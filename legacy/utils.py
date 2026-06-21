
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

from datetime import datetime

class Configuration():
    def __init__(self):
        # Specify number of EMG channes to be considered
        self.channel_no = 8
        self.users = 4

        # Specify dimension of EMG segment and overlap
        self.segment_dim = 2048   # EMG segment dimension
        self.increment_dim = 512     # EMG overlap dimension
        self.trainingBlocks = 1

        # Class label order MLP was trianed
        # Create a list
        self.class_label = ['unmarked data','hand at rest','fist','wrist flexion',
                       'wrist extension','radial deviations','ulnar deviations','extended palm']

        #self.csv_file = './Data/combined_files.csv'
        self.csv_file = './Data/combined_files.csv'
        self.data = './Data/UCI'

        self.classifier = 'tensorflow'


# Check input variables,

def feature_extract(channel_data, no_channels, segment_dim):
    ar_no = 6
    n = segment_dim
    F_immg = sum(abs(channel_data))
    F_mean = F_immg / no_channels
    F_MAV = abs(F_mean)
    F_var = np.var(channel_data)
    F_ms = np.square(channel_data).mean()
    F_rms = math.sqrt(F_ms)
    F_logrms = math.log(F_rms)
    F_kurt = stats.kurtosis(channel_data, fisher=False)
    F_skew = stats.skew(channel_data)
    AR_coef = autoreg(channel_data, ar_no, n)
    AR_coef = AR_coef.tolist()
    feature_vec = AR_coef + [F_immg, F_logrms, F_kurt, F_skew, F_rms, F_var, F_MAV]

    return feature_vec

def autoreg(ar_in, coef, seg_length):
    # estimate covariance flag
    pt = True

    approach = 'fb'
    window = 'now'
    data_offset = 0
    est_covar = True
    max_size = 250e3
    # Decimals??
    ar_in = np.around(ar_in, decimals=4)
    ar_scalar = ar_in[0]
    #     print(type(ar_in))
    if isinstance(ar_scalar, float):
        ar_in = ar_in[:]
    else:
        print('ar_in type does not match')
        return None

    yor = [[None]]
    yor[0] = ar_in
    Ne = len(yor)
    Ncaps = seg_length

    est_covar = pt

    # Perform estimation
    yOff = 0

    for kexp in range(Ne):
        yor[0][kexp] = yor[0][kexp] - yOff

    y = yor  # Keep the original y for later computation of e

    # TODO - not sure
    if approach == 'yw' and win == 'ppw':
        win = 'ppw'

    # override pt for the other approaches
    pt1 = True

    # Now compute the regression matrix
    if pt1:
        Ncap = Ncaps
        nmax = coef
        M = math.floor(max_size / coef)
        R1 = np.zeros((0, coef + 1))
        #         print("R1",R1.shape)
        if approach == 'fb':
            fb = True
            R2 = np.zeros((0, coef + 1))
            yb = [[]]
            for i in range(Ne):
                yb_nested = yb[0]
                yb_nested.append([None])
                # [[[None], [None], [None], [None]]]
            for kexp in range(Ne):
                y_reverse = y[0][::-1]
                yb[0][kexp] = np.conjugate(y_reverse)
        else:
            fb = False

    for kexp in range(0, Ne):
        yy = y[kexp]
        #         print("yy", yy)
        for k in range(nmax, Ncap - 1, M):
            min_list_1 = k + M
            min_element = min(Ncap, min_list_1)
            # k+1 => k
            jj = [i for i in range(k, min_element)]
            #             print("JJ", type(jj), jj)
            phi = np.zeros((len(jj), coef))
            if fb == True:
                phib = np.zeros((len(jj), coef))
            for k1 in range(1, coef + 1):
                int = 0
                for jj_ele in jj:
                    #                     print(jj_ele - k1, =',')
                    phi[int][k1 - 1] = -yy[jj_ele - k1]
                    int += 1
            #                 print("phi", phi)
            if fb == True:
                for k2 in range(1, coef + 1):
                    int = 0
                    for jj_ele in jj:
                        phib[int][k2 - 1] = -yb[0][kexp][jj_ele - k2]
                        int += 1
            #                     print("phib", phib)

            if fb == True:
                qr_1 = [R2]
                qr_2 = np.concatenate((phi, phib))
                qr_3 = []
                qr_4 = []
                qr_3 += [yy[jj[0]:]]
                #                 print("qr_3", (qr_3))
                qr_4_a = yb[0][kexp]
                qr_4 += [qr_4_a[jj[0]:]]
                #                 print("qr_4", (qr_4))
                qr_34 = np.concatenate((qr_3, qr_4), axis=None)
                qr_34 = (qr_34[np.newaxis]).T
                #                 print(qr_2.shape)
                #                 print(qr_34.shape)
                qr = np.append(qr_2, qr_34, axis=1)
                q, r = np.linalg.qr(qr)
                R22 = np.triu(r)
                nRr, nRc = R22.shape
                R23 = R22[:min(nRr, nRc), :]

            QR_1 = (yy[jj[0]:][np.newaxis]).T
            QR = np.append(phi, QR_1, axis=1)
            Q, R = np.linalg.qr(QR)
            R12 = np.triu(R)
            nRr, nRc = R12.shape
            R13 = R12[:min(nRr, nRc), :]

    R01 = R13
    R02 = R23
    covR = []
    covR = R01[:coef, :coef]
    P = np.linalg.pinv(covR)

    if approach != 'burg' or approach != 'gl':
        if fb == False:
            AR_coef = (np.matmul(P, R01[:coef, coef])).T
        else:
            AR_pinv = np.linalg.pinv(R02[:coef, :coef])
            AR_coef = (np.matmul(AR_pinv, R02[:coef, coef])).T
    #             print("AR_coef", AR_coef)
    return AR_coef

# Most frequent result of segment
def segment_mov(time_, segment_dim, emg_y):
    y_int_train = emg_y.astype(int)
    segmented_mov_list = []

    mov_samplewindow = y_int_train[time_: time_ + segment_dim]
    counts = np.bincount(mov_samplewindow)
    mov = np.argmax(counts)

    return mov


def sklearn_training(fv_x_train, fv_y_train, fv_x_test, fv_y_test):
    # create clssifier from model
    # 44, 4
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 8), max_iter=1000, activation='relu',
                        solver='adam', verbose=False, random_state=763)
    # fir training data into model
    mlp.fit(fv_x_train, fv_y_train)

    predictions_train = mlp.predict(fv_x_train)
    print("TRAIN accuracy_score", accuracy_score(predictions_train, fv_y_train))
    predictions_test = mlp.predict(fv_x_test)
    print("TEST accuracy_score", accuracy_score(predictions_test, fv_y_test))

    print(classification_report(predictions_train, fv_y_train))
    print(classification_report(predictions_test, fv_y_test))

    return predictions_test, fv_y_test


# MLP hyper-parameter


def MLP_parameter(fv_x_train, fv_y_train, fv_x_test, fv_y_test):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report

    mlp = MLPClassifier(max_iter=1000)

    parameter_space = {
        'hidden_layer_sizes': [(2000, 300, 10), (50, 50, 50), (50, 50, 8), (1000, 500)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(fv_x_train, fv_y_train)

    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    y_true, y_pred = fv_y_train, clf.predict(fv_x_train)

    print('Results on the test set:')
    print(classification_report(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))


# # MLP Tensorflow

# # Train Tensorflow

def tensorflow_training(fv_x_train, fv_y_train, fv_x_test, fv_y_test):
    def loss(model, x, y, training):
        y_ = model(x, training=training)

        return loss_object(y_true=y, y_pred=y_)

    # Optimize model
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    keras.backend.clear_session()

    # LDA data: np array to tensor
    # fv_x_train, fv_y_train, fv_x_test, fv_y_test
    def np2tensor(data):
        data = K.constant(data)
        return data

    batch_size = 32

    x_train = np2tensor(fv_x_train)
    y_train = np2tensor(fv_y_train)
    x_test = np2tensor(fv_x_test)
    y_test = np2tensor(fv_y_test)
    # x_y_train_ = tf.constant(fx_fy_train, dtype=tf.float32)

    # print(x_y_train)
    features = x_train
    labels = y_train

    x_y_train = tf.data.Dataset.from_tensor_slices((x_train,
                                                    y_train)).batch(batch_size)

    print(x_y_train)

    x_y_test = tf.data.Dataset.from_tensor_slices((x_test,
                                                   y_test)).batch(batch_size)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation=tf.nn.relu, input_shape=(4,)),
        tf.keras.layers.Dense(50, activation=tf.nn.relu),
        tf.keras.layers.Dense(8)
    ])

    predictions = model(features)
    # returns a logit for each class
    predictions[:5]

    # convert these logits to a probability for each class
    tf.nn.softmax(predictions[:5])

    # print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
    # print("    Labels: {}".format(labels))

    # training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # learning rate decude from 0.01 -> 0.001
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    loss_value, grads = grad(model, features, labels)

    print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                              loss_value.numpy()))

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                              loss(model, features, labels, training=True).numpy()))

    # for plotting
    train_loss_results = []
    train_accuracy_results = []

    # 201
    num_epochs = 601

    start_time3 = datetime.now()

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop
        for x, y in x_y_train:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            epoch_accuracy.update_state(y, model(x, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))

    test_accuracy = tf.keras.metrics.Accuracy()
    prediction_list = []
    y_list = []
    target_names = ["0", "1", "2", "3", "4", "5", "6", "7"]

    for (x, y) in x_y_test:
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int64)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    # print(classification_report(y_list, prediction_list, target_names=target_names))

    end_time3 = datetime.now()
    print('Duration: {}'.format(end_time3 - start_time3))

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()