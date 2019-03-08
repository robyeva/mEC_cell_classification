__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'
__license__ = 'gpl-3.0'

from tables import openFile
import os

import numpy as np
import scipy as sp
from scipy.linalg import eig
import matplotlib.pyplot as plt

from sklearn import svm, cross_validation

path = ''


def project_data(data, vector_space):
    """Projects data matrix to the PC space

    :param data: numpy.ndarray [N_samples x N features]
    Matrix of normalized data

    :param vector_space: numpy.ndarray [N_features x N_PC]
            contains PC on which you want to project the data (N_PC <= N_features)

    :returns: numpy.ndarray [N_samples x N_PC]
            Data projected onto the PC space

    """
    return sp.dot(data, vector_space)


def pca(data):
    """performs PCA on already preprocessed data

    :param data: numpy.ndarray [N_samples x N_features]
    Matrix of normalized data

    :return:

        eigenvalues_pc: numpy.ndarray  [N_features]
                        Array of sorted eigenvalues (explained variance)

        principal_components: numpy.ndarray [N_features x N_features]
                            projection matrix, each column is an eigenvector

        sorted_index: numpy.ndarray [N_features]
        array of indices (from most important to less) used for reconstructing an order of most important features

    """
    num_of_samples = data.shape[0]
    covariance_matrix = sp.dot(data.T, data) / num_of_samples
    eigenvalues_pc, principal_components = eig(covariance_matrix)
    sorted_index = sp.flipud(sp.argsort(eigenvalues_pc))
    eigenvalues_pc = eigenvalues_pc[sorted_index]
    principal_components = principal_components[:, sorted_index]

    return eigenvalues_pc, principal_components, sorted_index


def PCA_on_all_features(filename='mECdatafile'):
    """
    Plot data projection onto first 2 PCs - using labeled, mEC L2 cells with 9 features.

    Recorded features are:
    - RMP (resting membrane potential)
    - Input resistance
    - latency to first spike
    - spike duration at rheobase
    - adaptation ratio (last ISI / ISI1)
    - ISI1 / ISI2
    - dAP (depolarizing afterpotential)
    - sag duration (-750 pA) (half-width of the sag potential)
    - sag potential amplitude (-750 pA)

    Cell labels are:
    - stellate cells (reelin positive)
    - pyramidal cells (calbindin positive)

    :param filename: str
    Filename of stored data (PyTables)

    """

    # get data stored in PyTables file
    f = openFile(os.path.join(path, filename), mode="r")
    array = f.getNode("/feature_names")
    feature_names = array.read()
    print feature_names
    array = f.getNode("/data_features")
    data_features = array.read()
    # 0 = pyramidal, 1 = stellate
    array = f.getNode("/data_label")
    data_label = array.read()

    f.close()

    # Relying on equivalence (reelin+ = stellate, calbindin+ = pyramidal)
    calb_idx = np.where(data_label == 0)[0]
    reel_idx = np.where(data_label == 1)[0]
    num_calb = np.shape(calb_idx)[0]
    num_reel = np.shape(reel_idx)[0]

    # data normalization (preprocessing for PCA)
    mean_feat = np.mean(data_features, axis=0)
    std_feat = np.std(data_features, axis=0)
    data_features = data_features - mean_feat
    data_features = data_features / std_feat

    data_variance, proj_matrix, _ = pca(data_features)
    # data_proj = project_data(data_features, proj_matrix)

    # Select PC1 - PC2
    pcvect = proj_matrix[:, :2]
    proj_data = project_data(data_features, pcvect)
    variance_two_PC = np.sum(data_variance[:2]) / np.sum(data_variance)

    print 'Explained variance by PC1 and PC2', variance_two_PC

    # ===================== plot labeled data projection on PC1-PC2 ======================#
    fig = plt.figure(figsize=[12, 10])
    ax = fig.add_subplot(111)
    ax.scatter(proj_data[calb_idx, 0], proj_data[calb_idx, 1], c='FireBrick', marker='^', lw=0, s=45,
               label='L2 calbindin + (n = %d)' % num_calb)
    ax.scatter(proj_data[reel_idx, 0], proj_data[reel_idx, 1], c='RoyalBlue',
               marker='*', lw=0, s=70,
               label='L2 reelin + (n = %d)' % num_reel)

    ax.set_xlabel('PC 1 [a.u.]', fontsize=16, fontweight='bold')
    ax.set_ylabel('PC 2 [a.u.]', fontsize=16, fontweight='bold')
    ax.set_title('Data projection onto the first two principal components', fontsize=16)
    ax.legend(loc='upper left', scatterpoints=1, frameon=False, prop={'size': 14})
    ax.axis('equal')

    fig.savefig(os.path.join(path, 'PC1_PC2_mix_%s.eps' % filename), dpi=1000, format='eps', bbox_inches='tight')


def separating_sag_duration_value_SVM(filename='data_w_features_unstained'):
    """
    Uses linear 1-d SVM with sag duration (at -750 pA) as feature to determine the best value separating the classes
    (stained pyramidal and stellate cells, L2 mEC)

    :param filename: str
    Filename of stored data (PyTables). File should contain some unstained data (to be classified)

    :return:

        T_weighted: float
                    Threshold sag duration value to distinguish between the two classes
    """

    # get data stored in PyTables file
    f = openFile(os.path.join(path, filename), mode="r")
    array = f.getNode("/feature_names")
    feature_names = array.read()
    print feature_names
    array = f.getNode("/data_features")
    data_features = array.read()
    # 0 = pyramidal, 1 = stellate
    array = f.getNode("/data_label")
    data_label = array.read()
    f.close()

    unclassified_idx = np.where(data_label == -2)[0]
    num_unclass = np.shape(unclassified_idx)[0]
    print 'Number of unclassified cells', num_unclass

    # remove unclassified
    data_features = np.delete(data_features, unclassified_idx, axis=0)
    data_label = np.delete(data_label, unclassified_idx, axis=0)

    num_cells = np.shape(data_label)[0]
    calb_idx = np.where(data_label == 0)[0]
    reel_idx = np.where(data_label == 1)[0]

    num_calb = np.shape(calb_idx)[0]
    num_reel = np.shape(reel_idx)[0]

    print 'Cells class:'
    print 'reelin + ', num_reel
    print 'calb + ', num_calb
    print 'cell total: ', num_reel + num_calb + num_unclass

    analysis_index = np.append(calb_idx, reel_idx)
    analyse_label = data_label[analysis_index]

    # feature used for 1d SVM
    analyse_sag_duration = data_features[analysis_index, -1]
    # used for plotting, not in SVM
    analyse_sag_amplitude = data_features[analysis_index, -2]

    analyse_sag_duration = np.hstack((np.ones((num_cells, 1)), analyse_sag_duration[:, np.newaxis]))
    # NOTE: labels are:  0 = pyramidal, 1 = stellate
    # rescale analyse_label to -1, +1 (needed for SVM)
    analyse_label[analyse_label == 0] = -1

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(analyse_sag_duration, analyse_label,
                                                                         test_size=0.4, random_state=0)
    wCVclf = svm.SVC(kernel='linear', class_weight='balanced').fit(X_train, y_train)
    print wCVclf.score(X_test, y_test), 'CV score'

    # weighted SVM to accont for N_samples imbalance
    wclf = svm.SVC(kernel='linear', C=1.0, class_weight='balanced')
    wscores_10 = cross_validation.cross_val_score(wclf, analyse_sag_duration, analyse_label, cv=10)
    print "weight Accuracy: %0.2f (+/- %0.2f)" % (wscores_10.mean(), wscores_10.std() / 2)
    waccur_mean, waccur_std = wscores_10.mean(), wscores_10.std()

    wclf.fit(analyse_sag_duration, analyse_label)
    ww = wclf.coef_[0]
    ampl_min = np.min(analyse_sag_amplitude) - 10
    ampl_max = np.max(analyse_sag_amplitude) + 10
    xx = np.linspace(ampl_min, ampl_max)
    wa = -ww[0] / ww[1]
    wyy = wa * xx - wclf.intercept_[0] / ww[1]
    print ' weighted threshold: ', np.mean(wyy)

    # =========== with stained cells only
    fig = plt.figure(figsize=[12, 10])
    ax = fig.add_subplot(111)
    # threshold line
    ax.plot(xx, wyy, 'k--', label='with weights')

    where_stel = np.where(analyse_label == 1)[0]
    num_stel = np.shape(where_stel)[0]

    where_pyr = np.where(analyse_label == -1)[0]
    num_pyr = np.shape(where_pyr)[0]

    ax.scatter(analyse_sag_amplitude[where_pyr], analyse_sag_duration[where_pyr, 1], c='FireBrick', marker='^', lw=0, s=45,
               label='L2 calbindin + (n = %d)' % num_pyr)
    ax.scatter(analyse_sag_amplitude[where_stel], analyse_sag_duration[where_stel, 1],
               c='RoyalBlue', marker='*', lw=0, s=70,
               label='L2 reelin + (n = %d)' % num_stel)
    ax.set_xlabel('sag amplitude [mV]', fontsize=16, fontweight='bold')
    ax.set_ylabel('sag duration [ms]', fontsize=16, fontweight='bold')
    ax.set_title(
        'File: %s \n Best sag duration value (svm): with weights = %.2f ms \n Accuracy (10-CV): with weights = %.2f' % (
        filename, np.mean(wyy), waccur_mean), fontsize=13)
    ax.legend(loc='upper right', scatterpoints=1, frameon=False, prop={'size': 12})
    ax.set_xlim([ampl_min, ampl_max])
    fig.savefig(os.path.join(path, 'separating_sag_duration_value_%s.eps' % filename), dpi=2000, format='eps',
                bbox_inches='tight')

    return np.mean(wyy)


def classify_unstained_using_theshold(T, filename='data_w_features_unstained', tol=1, plotting=True):
    """
    Use sag duration value (-750 pA) to assign unstained cells to either class (pyramidal or stellate cells)
    # if sag duration >= T + tol --> label: 0 = pyramidal
    # if sag duration < T - tol ---> label: 1 = stellate

    :param T: float
              Threshold sag duration value to distinguish between the two classes (1d linear SVM)

    :param filename: str
                    Filename of stored data (PyTables). File should contain some unstained data (to be classified)

    :param tol: float
                Tolerance for decision boundary around threshold value

    :param plotting: bool
                    If True, plot a summary plot to visualize class assigned to unstained cells
    """

    # get data stored in PyTables file
    f = openFile(os.path.join(path, filename), mode="r")
    # the title contains the IDs of unstained cells
    title = f._gettitle()
    array = f.getNode("/feature_names")
    feature_names = array.read()
    print feature_names
    array = f.getNode("/data_features")
    data_features = array.read()
    # 0 = pyramidal, 1 = stellate
    array = f.getNode("/data_label")
    data_label = array.read()
    f.close()

    calb_idx = np.where(data_label == 0)[0]
    reel_idx = np.where(data_label == 1)[0]

    num_calb = np.shape(calb_idx)[0]
    num_reel = np.shape(reel_idx)[0]

    unclassified_idx = np.where(data_label == -2)[0]
    unclass_features = data_features[unclassified_idx, :]
    print 'Number unclassified cells: ', np.shape(unclassified_idx)

    if plotting:
        classified_pyr = np.array([], dtype='int')
        classified_stel = np.array([], dtype='int')

    sag_duration_unclass = unclass_features[:, -1]
    # to recognize what has been wrongly classified, we need to store the ID of the cells
    cellID_pyr = []
    cellID_stel = []
    # also store sag duration value to check how close it is wrt to the threshold
    sd_pyr = []
    sd_stel = []

    for idx, el in enumerate(sag_duration_unclass):
        if int(el) >= int(T) + tol:
            # change label! --- It's pyramid
            data_label[unclassified_idx[idx]] = 0
            cell_id = title[unclassified_idx[idx]]
            cellID_pyr.append(cell_id)
            sd_pyr.append(sag_duration_unclass[idx])
            if plotting:
                classified_pyr = np.append(classified_pyr, unclassified_idx[idx])

        elif int(el) <= int(T) - tol:
            # change label ! --- It's stellate
            data_label[unclassified_idx[idx]] = 1
            cell_id = title[unclassified_idx[idx]]
            cellID_stel.append(cell_id)
            sd_stel.append(sag_duration_unclass[idx])
            if plotting:
                classified_stel = np.append(classified_stel, unclassified_idx[idx])

    num_now_classified_pyr = np.shape(classified_pyr)[0]
    num_now_classified_stel = np.shape(classified_stel)[0]

    print 'Number of cells within boundary (cannot be classified): ', np.where(data_label == -2)[0]

    pyr_idx = np.where(data_label == 0)[0]
    stel_idx = np.where(data_label == 1)[0]
    num_stel = np.shape(stel_idx)[0]
    num_pyr = np.shape(pyr_idx)[0]
    print 'After classification'
    print 'pyr ', num_pyr
    print 'stell ', num_stel

    if plotting:
        sag_duration = data_features[:, -1]
        sag_amplitude = data_features[:, -2]

        ampl_min = np.min(sag_amplitude) - 10
        ampl_max = np.max(sag_amplitude) + 10
        xx = np.linspace(ampl_min, ampl_max)
        yy = np.ones_like(xx) * T

        fig = plt.figure(figsize=[12, 10])
        ax = fig.add_subplot(111)
        ax.plot(xx, yy, 'k--', label='with weights')
        ax.scatter(sag_amplitude[calb_idx], sag_duration[calb_idx], c='FireBrick', marker='^', lw=0, s=45,
                   label='L2 calbindin + (n = %d)' % num_calb)
        ax.scatter(sag_amplitude[reel_idx], sag_duration[reel_idx], c='RoyalBlue',
                   marker='*', lw=0, s=70, label='L2 reelin + (n = %d)' % num_reel)
        # now plot the ones with no staining but now classified
        ax.scatter(sag_amplitude[classified_pyr], sag_duration[classified_pyr], c='Black', marker='^', lw=0, s=45,
                   label='L2 classified pyr (n = %d)' % num_now_classified_pyr)
        ax.scatter(sag_amplitude[classified_stel], sag_duration[classified_stel], c='Black', marker='*', lw=0, s=70,
                   label='L2 classified stel (n = %d)' % num_now_classified_stel)
        # plot cells which could not be classified
        if len(np.where(data_label == -2)[0]) > 0:
            ax.scatter(sag_amplitude[np.where(data_label == -2)[0]], sag_duration[np.where(data_label == -2)[0]],
                       c='DarkGray', marker='o', lw=0, s=45, label='L2 unclassified cells (n = %d)'
                                                                %len(np.where(data_label == -2)[0]))

        ax.set_xlabel('sag amplitude [mV]', fontsize=16, fontweight='bold')
        ax.set_ylabel('sag duration [ms]', fontsize=16, fontweight='bold')
        ax.set_title('Classification results - %s' % filename, fontsize=16)
        ax.legend(loc='upper right', scatterpoints=1, frameon=False, prop={'size': 14})
        ax.set_xlim([ampl_min, ampl_max])
        fig.savefig(os.path.join(path, 'classified_sag_duration__%s.eps' % (filename)), dpi=300,
                    format='eps', bbox_inches='tight')