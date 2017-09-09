from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, BatchNormalization
from keras.models import Sequential


def buildNetwork(nb_classes, width=28, height=28):
    # Initialize data
    input_shape = (height, width, 1)

    # initialize the model
    model = Sequential()

    # first convolution / pool - 32x28x28 -> 32x24x24 -> 32x12x12
    model.add(Convolution2D(32,
                            (7, 7),
                            padding="same",
                            input_shape=input_shape,
                            activation="relu"))

    model.add(Convolution2D(32,
                            (5, 5),
                            padding="valid",
                            activation="relu"))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    # second convolution / pool - 32x12x12 -> 32x10x10 -> 32x8x8 -> 32x6x6
    model.add(Convolution2D(48,
                            (3, 3),
                            padding="valid",
                            activation="relu"))

    model.add(Convolution2D(48,
                            (3, 3),
                            padding="valid",
                            activation="relu"))

    model.add(Convolution2D(48,
                            (3, 3),
                            padding="valid",
                            activation="relu"))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    # flatten, 2 fully connected layers
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.40))

    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.40))

    model.add(BatchNormalization())

    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
