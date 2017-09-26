from shared.train import runNetwork
from network_custom5 import buildCustom5Network

verbose = 1
epochs = 10

runNetwork("Custom5",
           dataset_name="EMNIST-Balanced",
           data_filename="emnist-balanced.mat",
           build_function=buildCustom5Network,
           output_labels="labels.txt",
           epochs=epochs,
           batch_size=64,
           verbose=verbose)