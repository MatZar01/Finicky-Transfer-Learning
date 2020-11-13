from tfport import *
import timport
import numpy as np
import pickle
from imutils import paths
from keras.models import load_model
import fussy_methods
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import random
import time

DIR = './models_trained/80_perc'
CLS_PTH = DIR + '/fussy_model.cpickle'
M_DIR = DIR + '/model'
DATA_DIR = './evaluation/Kaggle'
batchSize = 1
bufferSize = 1000
categories = [DATA_DIR + '/Bckgs', DATA_DIR + '/Cracks']

# Load and sort extracted models
models_list = list(paths.list_files(M_DIR))

# load classifier
classifier = pickle.load(open(CLS_PTH, 'rb'))

keys = []
models = []
for model_path in models_list:
    keys.append(int(model_path.split('_')[-1].split('.')[0]))
    models.append(load_model(model_path))

# sort models by keys
models = [y for _, y in sorted(zip(keys, models))]

# count parameters
total_params = 0
for model in models:
    total_params += model.count_params()

print("[INFO] Total models parameters: {}".format(total_params))

# start timer
start_time = time.time()
iterator = 0

# add confusion matrix
TP = 0
TN = 0
FP = 0
FN = 0

for category in categories:
    # load dataset
    print("[INFO] Loading images...")
    imagePaths = list(paths.list_images(category))
    random.shuffle(imagePaths)
    print("[INFO] {} images loaded for {} class".format(len(imagePaths), categories.index(category)))

    # loop over images in both categories
    for image_path in imagePaths[:1000]:
        # load and resize image
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess image by expanding and subtracting mean RGB value
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # pass images thr network
        features = fussy_methods.extract_features_from_parts(models, image)
        # reshape features
        features = features.reshape((features.shape[0], 512 * 7 * 7))

        # make classification
        prediction = classifier.predict_proba(features)
        prediction = np.argmax(prediction[0])

        # assess prediction
        if categories.index(category) == 0 and prediction == 0:
            TN += 1
        if categories.index(category) == 1 and prediction == 1:
            TP += 1
        if categories.index(category) == 0 and prediction == 1:
            FP += 1
        if categories.index(category) == 1 and prediction == 0:
            FN += 1

        iterator += 1

        if iterator % 10 == 0:
            print('[INFO] {} out of {} images done in {} seconds'.format(iterator, len(imagePaths), time.time() - start_time))

# stop timer
stop_time = time.time()
exec_time = stop_time - start_time
print('[INFO] Execution time: {}s logged into log_metrics.txt'.format(exec_time))

# compute metrics
accuracy = (TP + TN) / iterator
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print('[INFO]\nAccuracy: {}\nPrecision: {}\nRecall: {}\nSaved to log'.format(accuracy, precision, recall))

# add log
log = open(DIR + '/log_{}.txt'.format(DATA_DIR.split('/')[-1]), 'w')
log.write('{} Dataset:\n'.format(DATA_DIR.split('/')[-1]))
log.write('[INFO] Execution time for {} images was {} seconds\n'.format(iterator, exec_time))
log.write('Accuracy: {}\nPrecision: {}\nRecall: {}\nSaved to log'.format(accuracy, precision, recall))
log.close()

