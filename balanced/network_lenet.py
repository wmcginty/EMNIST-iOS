from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, BatchNormalization
from keras.models import Sequential

def buildLeNet(nb_classes, width=28, height=28):
    # Initialize data
    input_shape = (height, width, 1)

    # initialize the model
    model = Sequential()

    # first convolution / pool
    model.add(Convolution2D(6,
                            (5, 5),
                            padding="valid",
                            input_shape=input_shape,
                            activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    # second convolution / pool
    model.add(Convolution2D(16,
                            (5, 5),
                            padding="valid",
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
