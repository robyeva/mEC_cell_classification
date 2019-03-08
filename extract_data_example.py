__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'
__license__ = 'gpl-3.0'


from visualize_features import get_data_w_features, plot_all_features, get_features_as_Fuchs2016, plot_features_as_Fuchs2016
from classify_cells import PCA_on_all_features, separating_sag_duration_value_SVM, classify_unstained_using_theshold

# ===== save relevant data features in PyTables file
get_data_w_features(xcl_file='data_file.xlsx', save_filename='data_w_features', take_unstained=False)

# ===== plot feature overview
plot_all_features()

# ===== plot selected features as Fuchs et al., 2016
get_features_as_Fuchs2016(save_filename='asFuchs_pyramids', get_stellate=False)
plot_features_as_Fuchs2016(filename='asFuchs_stellate', plot_stellate=True)

# ===== cell classification using PCA
PCA_on_all_features()

# ===== classify unstained cells based on sag duration
get_data_w_features(save_filename='data_w_features_unstained', take_unstained=True)
T_weighted = separating_sag_duration_value_SVM()
classify_unstained_using_theshold(T_weighted, filename='data_w_features_unstained', tol=1)
