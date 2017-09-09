from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, BatchNormalization
from keras.models import Sequential

def buildCustom1Network(nb_classes, width=28, height=28):
    # Initialize data
    input_shape = (height, width, 1)

    # initialize the model
    model = Sequential()

    # first convolution / pool - 32x28x28 -> 32x14x14
    model.add(Convolution2D(32,
                            (5, 5),
                            padding="same",
                            input_shape=input_shape,
                            activation="relu"))

    model.add(Convolution2D(16,
                            (5, 5),
                            padding="same",
                            activation="relu"))

    model.add(Convolution2D(8,
                            (3, 3),
                            padding="same",
                            activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    # second convolution / pool - 32x14x14 -> 32x7x7
    model.add(Convolution2D(8,
                            (3, 3),
                            padding="same",
                            activation="relu"))

    model.add(Convolution2D(8,
                            (3, 3),
                            padding="same",
                            activation="relu"))


    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    # flatten, 2 fully connected layers
    model.add(Flatten())

    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.40))

    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.40))

    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
