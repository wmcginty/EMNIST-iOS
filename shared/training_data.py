from scipy.io import loadmat

def loadDataFromFile(mat_file_path, width=28, height=28, max_=None, verbose=False):

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}

    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_]
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_]
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

    # Reshape training data to be valid
    if verbose == True: _len = len(training_images)
    for i in range(len(training_images)):
        training_images[i] = reshape(training_images[i], width, height)

    # Reshape testing data to be valid
    if verbose == True: _len = len(testing_images)
    for i in range(len(testing_images)):
        testing_images[i] = reshape(testing_images[i], width, height)

    # Extend the arrays to (None, 28, 28, 1)
    training_images = training_images.reshape(training_images.shape[0], height, width, 1)
    testing_images = testing_images.reshape(testing_images.shape[0], height, width, 1)

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    return ((training_images, training_labels), (testing_images, testing_labels), mapping)

def reshape(img, width, height):
    # Used to rotate images (for some reason they are transposed on read-in)
    img.shape = (width,height)
    img = img.T
    img = list(img)
    img = [item for sublist in img for item in sublist]
    return img
