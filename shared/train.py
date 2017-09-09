from keras.utils import np_utils

from shared.convert import convertModel
from shared.persist import loadModelNamed
from shared.persist import saveModel
from shared.training_data import loadDataFromFile

def trainNetwork(model, training_data, epochs, batch_size=64, verbose=1):

    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping = training_data
    nb_classes = len(mapping)

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_data=(x_test, y_test))

def testNetwork(model, training_data, verbose=1):

    # Initialize data
    _, (x_test, y_test), mapping = training_data
    nb_classes = len(mapping)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    score = model.evaluate(x_test, y_test, verbose=verbose)
    return score[1]

def runNetwork(modelName, dataset_name, data_filename, build_function, output_labels, epochs=10, batch_size=64, verbose=1):

    print("Beginning train/test cycle for the " + modelName + " network on the " + dataset_name + " data set.")
    training_data = loadDataFromFile(data_filename)

    model = loadModelNamed(modelName)
    if model is None:
        _, _, mapping = training_data
        model = build_function(len(mapping)) #The length of 'mapping' is the number of classes in our training data

    trainNetwork(model, training_data=training_data, batch_size=batch_size, epochs=epochs, verbose=verbose)
    accuracy = testNetwork(model, training_data=training_data, verbose=verbose)
    print('Test accuracy: ', accuracy * 100)

    saveModel(model, modelName)

    classifierName = modelName + '_{0:.2f}a'.format(accuracy * 100)
    convertModel(model, title=classifierName,
                 description="An alphanumeric classifier trained on the " + dataset_name + " data set.",
                 class_labels=output_labels)
    print("Saved model (with weights) and exported to .mlmodel file.")