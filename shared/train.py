from keras.utils import np_utils

def trainNetwork(model, training_data, epochs, batch_size=64, runs=1, verbose=1):

    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping = training_data
    nb_classes = len(mapping)

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    for x in range(1, runs):
        print("Starting run no%d..." % (x))
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose,
                  validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=verbose)
        print('Test score:', score[0])
        print('Test accuracy:', score[1] * 100)