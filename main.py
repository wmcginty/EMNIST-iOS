from training_data import loadDataFromFile
from network import buildNetworkAndTrain
from network import trainNetwork
from persist import saveModel
from persist import loadModelNamed
from convert import convertModel

print("Loading training data from emnist-byclass.mat ...")
training_data = loadDataFromFile("emnist-byclass.mat")
print("Training data loaded.")

model = loadModelNamed("EMNIST")
if model is None:
    print ("Model could not be loaded. Re-building.")
    model = buildNetworkAndTrain(training_data)

else:
    print("Model loaded from file. Training")
    trainNetwork(model, training_data=training_data, epochs=1)

print("Saving model.")
saveModel(model, "EMNIST")

convertModel(model, title="EMNISTClassifier",
             description="An alphanumeric classifier trained on the EMNIST data set.")