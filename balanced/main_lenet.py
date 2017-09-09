from network_lenet import buildNetwork
from shared.train import trainNetwork
from shared.convert import convertModel
from shared.persist import loadModelNamed
from shared.persist import saveModel
from shared.training_data import loadDataFromFile
from shared.train import testNetwork

modelName = "LeNet"
dataName = "emnist-balanced.mat"

print("Beginning train/test cycle for " + modelName +" on " + dataName)
training_data = loadDataFromFile(dataName)

model = loadModelNamed(modelName)
if model is None:
    _, _, mapping = training_data
    model = buildNetwork(len(mapping)) #The length of 'mapping' is the number of classes in our training data

trainNetwork(model, training_data=training_data, batch_size=64, epochs=10)
results = testNetwork(model, training_data=training_data)
print("Test score: " + results[0])
print("Test accuracy: " + results[1] * 100 + "%")

saveModel(model, modelName)
convertModel(model, title=modelName + results[1],
             description="An alphanumeric classifier trained on the EMNIST data set.",
             class_labels="labels.txt")