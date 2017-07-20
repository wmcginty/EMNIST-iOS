from keras.models import load_model
import h5py
import os.path

def saveModel(model, name):
    model.save(name + '.h5')  # creates a HDF5 file 'name.h5'
    del model

def loadModelNamed(name):
    try:
        return load_model(name + ".h5")
    except:
        return None