from shared.train import runNetwork
from network_lenet import buildLeNet
from network_custom1 import buildCustom1Network
from network_custom2 import buildCustom2Network
from network_custom3 import buildCustom3Network

verbose = 1
epochs = 1

runNetwork("LeNet",
           dataset_name="EMNIST-Balanced",
           data_filename="emnist-balanced.mat",
           build_function=buildLeNet,
           output_labels="labels.txt",
           epochs=epochs,
           batch_size=64,
           verbose=verbose)

runNetwork("Custom1",
           dataset_name="EMNIST-Balanced",
           data_filename="emnist-balanced.mat",
           build_function=buildCustom1Network,
           output_labels="labels.txt",
           epochs=epochs,
           batch_size=64,
           verbose=verbose)

runNetwork("Custom2",
           dataset_name="EMNIST-Balanced",
           data_filename="emnist-balanced.mat",
           build_function=buildCustom2Network,
           output_labels="labels.txt",
           epochs=epochs,
           batch_size=64,
           verbose=verbose)

runNetwork("Custom3",
           dataset_name="EMNIST-Balanced",
           data_filename="emnist-balanced.mat",
           build_function=buildCustom3Network,
           output_labels="labels.txt",
           epochs=epochs,
           batch_size=64,
           verbose=verbose)