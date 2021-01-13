from tfport import *
import timport
import numpy as np
import cv2
from imutils import paths
from keras.models import load_model
import fussy_methods
from hdf5_dataset_writer import HDF5DatasetWriter
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
import progressbar
import random
import os
import sys

args = sys.argv

if len(args) == 1:
    M_DIR = './models'
else:
    M_DIR = args[1]

DATA_DIR = './dataset'
batchSize = 1
bufferSize = 1000

# load model paths
m_pts = list(paths.list_files(M_DIR))

# load dataset
print("[INFO] Loading images...")
imagePaths = list(paths.list_images(DATA_DIR))
random.shuffle(imagePaths)
print("[INFO] {} images loaded".format(len(imagePaths)))

# extract class labels and encode them
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

for m_pt in m_pts:
    model = load_model(m_pt)
    out_shape = model.layers[-1].output.shape[1] * model.layers[-1].output.shape[2] * model.layers[-1].output.shape[3]

    # initialize dataset writer
    dataset = HDF5DatasetWriter((len(imagePaths), out_shape), M_DIR + '/' + m_pt.split('/')[-1].replace('h5', 'hdf5')
                                , "features", bufferSize)
    dataset.storeClassLabels(le.classes_)

    # initialize progress bar
    widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

    # loop over images
    for i in np.arange(0, len(imagePaths), batchSize):
        batchPaths = imagePaths[i:i + batchSize]
        batchLabels = labels[i:i + batchSize]
        batchImages = []

        for (j, imagePath) in enumerate(batchPaths):
            # load and resize image
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)

            # preprocess image by expanding and subtracting mean RGB value
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)

            # add image to batch
            batchImages.append(image)

        # pass images thr network
        batchImages = np.vstack(batchImages)
        #features = fussy_methods.extract_features_from_parts(models, batchImages)
        features = model.predict(batchImages)
        # reshape features
        features = features.reshape((features.shape[0], out_shape))

        # add features and labels to dataset
        dataset.add(features, batchLabels)
        pbar.update(i)

    dataset.close()
    pbar.finish()
