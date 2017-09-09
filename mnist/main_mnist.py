from network import buildNetwork
from shared.train import runNetwork

runNetwork("MNIST",
           dataset_name="EMNIST-MNIST",
           data_filename="emnist-mnist.mat",
           build_function=buildNetwork,
           output_labels="labels.txt",
           epochs=1,
           batch_size=64)

