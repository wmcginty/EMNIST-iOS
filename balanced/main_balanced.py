from network import buildNetworkAndTrain
from network import trainNetwork
from shared.convert import convertModel
from shared.persist import loadModelNamed
from shared.persist import saveModel
from shared.training_data import loadDataFromFile

print("Loading training data...")
training_data = loadDataFromFile("emnist-balanced.mat")
print("Training data loaded")

model = loadModelNamed("EMNIST")
if model is None:
    print ("Model could not be loaded. Building and training...")
    model = buildNetworkAndTrain(training_data, epochs=2)

else:
    print("Model loaded from file. Training...")
    trainNetwork(model, training_data=training_data, epochs=10)

print("Saving model...")
saveModel(model, "EMNIST")

print("Converting to .mlmodel ...")
convertModel(model, title="EMNISTClassifier",
             description="An alphanumeric classifier trained on the EMNIST data set.",
             class_labels="labels.txt")