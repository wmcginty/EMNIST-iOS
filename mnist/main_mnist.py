from network import buildNetworkAndTrain
from network import trainNetwork
from shared.convert import convertModel
from shared.persist import loadModelNamed
from shared.persist import saveModel
from shared.training_data import loadDataFromFile

print("Loading training data from emnist-mnist.mat ...")
training_data = loadDataFromFile("emnist-mnist.mat")
print("Training data loaded...")

model = loadModelNamed("MNIST")
if model is None:
    print ("Model could not be loaded. Building before training...")
    model = buildNetworkAndTrain(training_data)

else:
    print("Model loaded from file. Training...")
    trainNetwork(model, training_data=training_data, epochs=10)

print("Saving model...")
saveModel(model, "MNIST")

print("Converting to MLModel...")
convertModel(model, title="MNISTClassifier",
             description="An digit classifier trained on the MNIST data set.",
             class_labels="labels.txt")