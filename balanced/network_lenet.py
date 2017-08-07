from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential

def buildNetwork(nb_classes, width=28, height=28):

    # Initialize data
    input_shape = (height, width, 1)

    #initialize the model
    model = Sequential()

    #first convolution - feature maps - 6x24x24 (kernel size = 5x5)
    model.add(Convolution2D(6,
                            (5, 5),
                            padding="valid",
                            input_shape=input_shape,
                            activation="relu"))

    #first pooling - 6x14x14 (pool size = 2, 2)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    #second convolution - 6x10x10 (kernel size = 5x5)
    model.add(Convolution2D(16,
                            (5, 5),
                            padding="valid",
                            activation="relu"))

    #second pooling - 6x5x5 (pool size = 2, 2)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.50))

    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.50))

    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
