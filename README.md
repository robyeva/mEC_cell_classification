# mEC_cell_classification
Analysis of data recorded from medial entorhinal cortex (mEC), layer 2. Cell classification based on relevant electrophysiological features

Data are presented in Winterer, J., Maier, N., Wozny, C., Beed, P., Breustedt, J., Evangelista, R., ... & Schmitz, D. (2017). Excitatory microcircuits within superficial layers of the medial entorhinal cortex. Cell reports, 19(6), 1110-1116.

These files contain the code for reading and visualizing the data, and classifying cell in two different groups. 
Excitatory cells in layer 2 of mEC can belong to either the pyramidal cell or stellate cell class. Pyramidal cells can be stained with calbindin, stellate cells with reelin. Not all cells in the dataset are stained.

For stained cells, electrophysiological properties are plotted and PCA is performed to check for class separability. Unstained cells are assigned a class based on the result of a linear 1d SVM classifier which uses the most relevant feature (half-width of sag potential, or, in short, sag duration).  


For questions and additional information about the dataset: roberta.evangelista (at) posteo.de
