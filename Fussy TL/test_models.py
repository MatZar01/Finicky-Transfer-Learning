from tfport import *
import timport
import time
import numpy as np
import pickle
from imutils import paths
from keras.models import load_model
import fussy_methods
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
from keras.applications import VGG16
import sys

args = sys.argv

if len(args) == 1:
    SUB = './models'
else:
    SUB = './' + args[1]

# get initial parameter number
init_params = VGG16(weights='imagenet', include_top=False).count_params()

DATA_DIR = './evaluation/Kaggle'
categories = [DATA_DIR + '/Background', DATA_DIR + '/Crack']

# get model paths
models_pts = []
files_pts = list(paths.list_files(SUB))
for file in files_pts:
    if '.h5' in file:
        models_pts.append(file)

models_pts.sort()

# add log
LOG_PATH = './logss/log_{}.txt'.format(SUB[2:])
log_file = open(LOG_PATH, 'w')

# test size
i_min = 1000

# besties
best_acc = 0
best_model = ''
best_params = 0

acc = []
perc = []

for model_pt in models_pts:
    CLS_PTH = model_pt.replace('.h5', '.cpickle')
    model = load_model(model_pt)
    # load classifier
    classifier = pickle.load(open(CLS_PTH, 'rb'))

    # count parameters
    total_params = model.count_params()

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
        print("[INFO] {} images loaded for {} class".format(i_min, categories.index(category)))

        # loop over images in both categories
        for image_path in imagePaths[0:i_min]:
            # load and resize image
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)

            # preprocess image by expanding and subtracting mean RGB value
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)

            # pass images thr network
            features = model.predict(image)
            # reshape features
            features = features.reshape((features.shape[0], features.shape[1] * features.shape[2] * features.shape[3]))

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
                print('[INFO] {} out of {} images done in {} seconds'.format(iterator, 2 * i_min, time.time() - start_time))

    # stop timer
    stop_time = time.time()
    exec_time = stop_time - start_time
    print('[INFO] Execution time: {}s logged into {}'.format(exec_time, LOG_PATH.split('/')[-1]))

    # compute metrics
    accuracy = (TP + TN) / iterator
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0

    if accuracy > best_acc:
        best_acc = accuracy
        best_model = model_pt
        best_params = total_params

    print('[INFO] model {} metrics:'.format(model_pt))
    print('[INFO]\nAccuracy: {}\nPrecision: {}\nRecall: {}\nSaved to log'.format(accuracy, precision, recall))

    # write to txt log
    log_file.write('MODEL: {}\n'.format(model_pt))
    log_file.write('Paremeters: {}\n'.format(total_params))
    log_file.write('[INFO] Execution time for {} images was {} seconds\n'.format(iterator, exec_time))
    log_file.write('Accuracy: {}\nPrecision: {}\nRecall: {}\n\n'.format(accuracy, precision, recall))
    log_file.flush()

    acc.append(accuracy * 100)
    perc.append(total_params/init_params * 100)

plt.plot(perc, acc)
plt.xlabel('% of parameters in model')
plt.ylabel('Accuracy [%]')
plt.title('Accuracy per parameters comparison')
# save best
log_file.write('\n\nBEST MODEL: {}'.format(best_model))
log_file.write('\nPARAMETERS: {}'.format(best_params))
log_file.write('\nACCURACY: {}'.format(best_acc))

# save log
log_file.close()
plt.show()
print('All done!')
