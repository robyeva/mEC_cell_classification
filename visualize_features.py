__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'
__license__ = 'gpl-3.0'

import numpy as np
import os
import xlrd
from tables import openFile

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import axes3d

path = ''

def get_data_w_features(xcl_file='data_file.xlsx', save_filename='mECdatafile',
                        take_unstained=False):
    """
    Read .xlsx file with recorded elecrophysiological data and store cells, relevant features and labels in a PyTables file.
    Stored are only cells for which all features were recoreded.

    :param xcl_file: str
    Source datafile with recorded cells (L2 mEC), their features and labels.

    Recorded features of interest are:
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

    :param save_filename: str
    Name to assign to PyTables file where features are saved

    :param take_unstained: bool
    If True, also stores unlabeled cells with all features

    """
    xcl_file = os.path.join(path, xcl_file)
    book = xlrd.open_workbook(xcl_file)

    # data sheet
    my_sheet = book.sheet_by_index(0)
    num_rows = my_sheet.nrows - 1  # first is column name

    data_features = np.zeros((num_rows, 9))
    # if it was stained for calbindin (+1 it was positive, 0 not tested) and +2 if it was reel+
    data_stain = np.zeros(num_rows)
    # description: 0 = pyramidal, 1 = stellate
    data_label = -np.ones(num_rows)

    want_list = np.array([], dtype='int')
    take_out_idx = -1
    if take_unstained:
        data_id = []

    feature_names = ['RMP', 'Rin', 'latency_to_first_spike', 'spike_duration', 'adaptation_ratio', 'ISI1/ISI2',
                     'dAP', 'sag_amplitude (-750pA)', 'sag_duration (-750pA)']
    features_col = np.array([8, 18, 19, 9, 13, 21, 20, 15, 14])
    for aux_idx in features_col:
        content = my_sheet.cell(0, aux_idx)
        print content.value

    for j in range(1, num_rows - 1):
        for idx_f in features_col:

            cell = my_sheet.cell(j, idx_f)
            labcell = cell.value

            if type(labcell) is str or type(labcell) is unicode:
                take_out_idx = j    # missing values
            else:
                data_features[j, np.where(features_col == idx_f)[0]] = cell.value

        # staining value
        cell = my_sheet.cell(j, 6)
        labcell = cell.value
        if labcell.find('cal+') != -1:
            data_stain[j] = +1
        elif labcell.find('rel+') != -1:
            data_stain[j] = +2

        if data_stain[j] == 2:
            data_label[j] = 1   # reel+ stellate
        elif data_stain[j] == 1:
            data_label[j] = 0   # calb+ (and reel-) pyramid

        else:
            if take_unstained:
                # cells which have features but no label
                if data_stain[j] == 0:
                    data_label[j] = -2
            else:
                take_out_idx = j

        if take_out_idx != j:
            want_list = np.append(want_list, j)
            if take_unstained:
                # also take the cell ID and append it to the list
                cell = my_sheet.cell(j, 4)
                labcell = cell.value
                data_id.append(labcell)

    # cut incomplete samples
    data_features = data_features[want_list, :]
    data_stain = data_stain[want_list]
    data_label = data_label[want_list]

    if take_unstained:
        f = openFile(os.path.join(path, save_filename), mode="w", title=data_id)
    else:
        f = openFile(os.path.join(path, save_filename), mode="w", title='data_with_all_9features')
    for name, el in zip(['feature_names','data_features', 'data_label', 'data_stain'],
                        [feature_names, data_features, data_label, data_stain]):
        g = f.createArray('/', '%s' % name, el)
        print g

    f.close()


def get_features_as_Fuchs2016(xcl_file='data_file.xlsx', save_filename='asFuchs_stellate', get_stellate=True):
    """
    Read .xlsx file with data and store cells, relevant features (depends on get_stellate parameter) and labels in
    a PyTables file. Stored are only cells for which all features were recoreded (considering only 3 features increases
    number of stored cells w.r.t. those in get_data_w_features).

    :param xcl_file: str
    Source datafile with recorded cells (L2 mEC), their features and labels.

    :param save_filename: str
    Name to assign to PyTables file where features are saved

    :param get_stellate: bool
    if True, select stellate cells for which the following features are available:
    - latency to first spike
    - ISI 1/ ISI2
    - dAP (depolarizing afterpotential)
    if False, select pyramidal cells for which the following features are available:
    - sag amplitude (-750 pA)
    - latency to first spike
    - dAP (depolarizing afterpotential)

    """

    xcl_file = os.path.join(path, xcl_file)
    book = xlrd.open_workbook(xcl_file)

    my_sheet = book.sheet_by_index(0)
    num_rows = my_sheet.nrows - 1  # first is name (not counted)

    # features
    data_features = np.zeros((num_rows, 3))     # only 3 features wanted
    # if it was stained for calbindin (+1 it was positive, 0 not tested) and +2 if it was reel+
    data_stain = np.zeros(num_rows)
    # description: 0 = pyramidal, 1 = stellate
    data_label = -np.ones(num_rows)

    want_list = np.array([], dtype='int')
    take_out_idx = -1

    if get_stellate:
        feature_names = ['latency_to_first_spike', 'ISI1/ISI2', 'dAP']
        features_col = np.array([19, 21, 20])
    else:
        feature_names = ['latency_to_first_spike', 'dAP', 'sag_amplitude (-750pA)']
        features_col = np.array([19, 20, 15])

    for j in range(1, num_rows - 1):
        for idx_f in features_col:
            cell = my_sheet.cell(j, idx_f)
            labcell = cell.value

            if type(labcell) is str or type(labcell) is unicode:
                take_out_idx = j
            else:
                data_features[j, np.where(features_col == idx_f)[0]] = cell.value

        # staining value
        cell = my_sheet.cell(j, 6)
        labcell = cell.value

        if labcell.find('cal+') != -1:
            data_stain[j] = +1
        elif labcell.find('rel+') != -1:
            data_stain[j] = +2

        if data_stain[j] == 2:
            data_label[j] = 1  # reel+ stellate
            if not get_stellate:
                take_out_idx = j
        elif data_stain[j] == 1:
            data_label[j] = 0  # calb+ (and reel-) pyramid
            if get_stellate:
                take_out_idx = j
        else:
            take_out_idx = j

        if take_out_idx != j:
            want_list = np.append(want_list, j)

    data_features = data_features[want_list, :]
    data_stain = data_stain[want_list]
    data_label = data_label[want_list]

    if get_stellate:
        data_title = 'features_for_stellate_cells'
    else:
        data_title = 'features_for_pyramidal_cells'

    f = openFile(os.path.join(path, save_filename), mode="w", title=data_title)
    for name, el in zip(['feature_names', 'data_features', 'data_label', 'data_stain'],
                        [feature_names, data_features, data_label, data_stain]):
        g = f.createArray('/', '%s' % name, el)
        print g

    f.close()


def plot_all_features(filename='mECdatafile'):
    """
    Plot overview of data features depending on class label (stellate or pyramidal cell in L2, mEC)

    :param filename: str
    Filename of stored data (PyTables)

    """
    f = openFile(os.path.join(path, filename), mode="r")
    # observations, 9 features
    array = f.getNode("/feature_names")
    feature_names = array.read()
    print feature_names
    array = f.getNode("/data_features")
    data_features = array.read()
    # 0 = pyramidal, 1 = stellate
    array = f.getNode("/data_label")
    data_label = array.read()
    f.close()

    pyr_idx = np.where(data_label == 0)[0]
    stel_idx = np.where(data_label == 1)[0]

    print 'Data overview:'
    print 'pyramidal cells ', len(pyr_idx)
    print 'stellate cells', len(stel_idx)

    # ======================= plot features! =================================#

    fig = plt.figure(figsize=[20, 12])
    rcParams.update({'font.size': 16, 'font.weight': 'bold', 'font.serif': 'Arial'})
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs0 = gridspec.GridSpec(3, 3)

    titles_list = ['Resting membrane \n potential [mV]', 'Input \n resistance [M$ \mathbf{\Omega}$]',
                   'Latency to \n first spike [ms]', 'Spike duration at \n rheobase [ms]',
                   'Adaptation \n ratio (last ISI / ISI1)', 'Adaptation \n ratio (ISI1 / ISI2)', 'dAP amplitude [mV]',
                   'Sag potential amplitude [mV]',
                   'Sag potential half-width [ms]']
    y_titles = [1.015, 1.015, 1.015, 1.015, 1.015, 1.015, 1.015, 1.07, 1.07]

    for feature_idx in np.arange(9):
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[feature_idx])
        ax = plt.Subplot(fig, gs00[0])
        fig.add_subplot(ax)
        data_stel = data_features[stel_idx, feature_idx]
        data_pyr = data_features[pyr_idx, feature_idx]

        ax.scatter(np.zeros(np.shape(stel_idx)[0]), data_stel,
                    color='RoyalBlue', marker='_', s=200, label='stellate')
        stel_mean, stel_std = np.mean(data_stel), np.std(data_stel)
        ax.errorbar(0, stel_mean, yerr=stel_std, capsize=15, fmt='_', markersize=40, mec='MidnightBlue', barsabove=True,
                     color='MidnightBlue', ecolor='MidnightBlue', elinewidth=2.7, mew=3.0)
        ax.scatter(np.ones(np.shape(pyr_idx)[0]), data_pyr, color='FireBrick', marker='_',
                    s=200, label='pyramidal')
        pyr2_mean, pyr2_std = np.mean(data_pyr), np.std(data_pyr)
        ax.errorbar(1, pyr2_mean, yerr=pyr2_std, capsize=15, fmt='_', markersize=40, mec='MidnightBlue', barsabove=True,
                     color='MidnightBlue', ecolor='MidnightBlue', elinewidth=2.7, mew=3.0)

        ax.set_title(titles_list[feature_idx], y=y_titles[feature_idx])
        ax.xaxis.tick_bottom()
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Stel', 'Pyr'])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        for label in ax.get_yticklabels()[::2]:
            label.set_visible(False)

    fig.savefig(os.path.join(path, 'all_features_%s.eps' % filename), dpi=2000, format='eps', bbox_inches='tight')


def plot_features_as_Fuchs2016(filename='asFuchs_stellate', plot_stellate=True):
    """
    Plot 3d visualization of data to compare with Fuchs et al., 2016, Fig. 1B/C

    :param filename: str
    Filename of stored data (PyTables)

    :param plot_stellate: bool
    if True, plot features of stellate cells, use pyramidal cells otherwise

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

    if plot_stellate:
        col = 'RoyalBlue'
        data_idx = np.where(data_label == 1)[0]
        xlab = 'dAP amplitude [mV]'
        ylab = 'ISI 1 / ISI 2'
        zlab = 'Latency to first spike [log10, ms]'
        idx_featues = [2, 1, 0]
        my_label = 'Stel (reel +) \n (n = %d)' % len(data_idx)
        my_marker = '*'
    else:
        data_idx = np.where(data_label == 0)[0]
        col = 'FireBrick'
        xlab = 'Sag potential amplitude [log10, mV]'
        ylab = 'Latency to first spike [log10, ms]'
        zlab = 'dAP amplitude [mV]'
        idx_featues = [2, 0, 1]
        my_label = 'Pyr (calb +) \n (n = %d)' % len(data_idx)
        my_marker = '^'

    fig = plt.figure()
    rcParams.update({'font.size': 14, 'font.weight': 'bold', 'font.serif': 'Arial'})
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_features[:, idx_featues[0]], data_features[:, idx_featues[1]], zs=data_features[:, idx_featues[2]],
               zdir='z', marker=my_marker, lw=0, s=70, c=col, depthshade=True, label=my_label)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    for label in ax.get_yticklabels()[::2]:
        label.set_visible(False)
    for label in ax.get_xticklabels()[::2]:
        label.set_visible(False)
    for label in ax.get_zticklabels()[::2]:
        label.set_visible(False)

    ax.legend(loc='center left', scatterpoints=1, frameon=False, prop={'size': 14})
    fig.savefig(os.path.join(path, 'compare_w_Fuchs2016_%s_stellate_%s.eps' % (filename, str(plot_stellate))), dpi=2000,
                format='eps', bbox_inches='tight')
