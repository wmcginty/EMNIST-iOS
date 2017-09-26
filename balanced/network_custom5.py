from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential


def buildCustom5Network(nb_classes, width=28, height=28):
    # Initialize data
    input_shape = (height, width, 1)

    # initialize the model
    model = Sequential()

    # first conv conv conv pool
    model.add(Convolution2D(8,
                            (3, 3),
                            padding="same",
                            input_shape=input_shape,
                            activation="relu"))

    model.add(Convolution2D(8,
                            (3, 3),
                            padding="same",
                            activation="relu"))

    model.add(Convolution2D(8,
                            (3, 3),
                            padding="same",
                            activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second conv conv conv pool
    model.add(Convolution2D(16,
                            (3, 3),
                            padding="same",
                            activation="relu"))

    model.add(Convolution2D(16,
                            (3, 3),
                            padding="same",
                            activation="relu"))

    model.add(Convolution2D(16,
                            (3, 3),
                            padding="same",
                            activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # third conv conv conv pool
    model.add(Convolution2D(32,
                            (3, 3),
                            padding="same",
                            activation="relu"))

    model.add(Convolution2D(32,
                            (3, 3),
                            padding="same",
                            activation="relu"))

    model.add(Convolution2D(32,
                            (3, 3),
                            padding="same",
                            activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # flatten, 2 fully connected layers
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.40))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.40))

    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
