from network_lenet import buildNetwork
from shared.train import trainNetwork
from shared.convert import convertModel
from shared.persist import loadModelNamed
from shared.persist import saveModel
from shared.training_data import loadDataFromFile

print("Loading training data...")
training_data = loadDataFromFile("emnist-balanced.mat")

model = loadModelNamed("EMNIST")
if model is None:

    #The length of 'mapping' is the number of classes in our training data
    _, _, mapping = training_data

    print ("Model could not be loaded. Building network...")
    model = buildNetwork(len(mapping))

print("Model loaded. Beginning training...")
trainNetwork(model, training_data=training_data, batch_size=64, epochs=25, runs=4)

print("Saving model (including weights)...")
saveModel(model, "EMNIST")

print("Converting to .mlmodel for iOS use...")
convertModel(model, title="EMNISTClassifier",
             description="An alphanumeric classifier trained on the EMNIST data set.",
             class_labels="labels.txt")