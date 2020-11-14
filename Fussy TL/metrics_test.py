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
import time
import os
from pyexcel_ods import save_data
from collections import OrderedDict

# load evaluation data
DATA_DIR = './evaluation/Kaggle'
categories = [DATA_DIR + '/Bckgs', DATA_DIR + '/Cracks']

# get subfolders
MASTER = './models_trained'
SUB = [x[1] for x in os.walk(MASTER)][0]
SUB.sort()

# split dataset into 5 subsets
sub_size = 2
datasets = 5
i_min = [x * sub_size for x in list(range(0, datasets))]

# add global ods output
data_global = OrderedDict()
dict_data_global = {}
for i in range(0, datasets):
    dict_data_global.update({'{} dataset'.format(i + 1): []})
    dict_data_global['{} dataset'.format(i + 1)].append(['Network', 'Accuracy', 'Precision', 'Recall'])

for subfolder in SUB:
    # check if vanilla
    if 'vanilla' in subfolder:
        vanilla = True
    else:
        vanilla = False

    DIR = MASTER + '/' + subfolder
    CLS_PTH = DIR + '/fussy_model.cpickle'
    M_DIR = DIR + '/model'

    # add log
    log = open(DIR + '/log_{}.txt'.format(DATA_DIR.split('/')[-1]), 'w')

    # add local ods output
    data_local = OrderedDict()
    dict_data_local = {'{} metrics'.format(DATA_DIR.split('/')[-1]): []}
    dict_data_local['{} metrics'.format(DATA_DIR.split('/')[-1])].append(['Accuracy', 'Precision', 'Recall'])

    # Load and sort extracted models
    models_list = list(paths.list_files(M_DIR))

    # load classifier
    classifier = pickle.load(open(CLS_PTH, 'rb'))

    if not vanilla:
        keys = []
        models = []
        for model_path in models_list:
            keys.append(int(model_path.split('_')[-1].split('.')[0]))
            models.append(load_model(model_path))

        # sort models by keys
        models = [y for _, y in sorted(zip(keys, models))]
    else:
        models = [load_model(models_list[0])]

    # count parameters
    total_params = 0
    for model in models:
        total_params += model.count_params()

    print("[INFO] Total models parameters: {}".format(total_params))

    for i in range(0, len(i_min)):
        # start timer
        start_time = time.time()
        iterator = 0

        # add confusion matrix
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        # add log info
        dict_data_local['{} metrics'.format(DATA_DIR.split('/')[-1])].append(['Dataset', i + 1])

        for category in categories:
            # load dataset
            print("[INFO] Loading images...")
            imagePaths = list(paths.list_images(category))
            print("[INFO] {} images loaded for {} class".format(sub_size, categories.index(category)))

            # loop over images in both categories
            for image_path in imagePaths[i_min[i]:i_min[i] + sub_size]:
                # load and resize image
                image = load_img(image_path, target_size=(224, 224))
                image = img_to_array(image)

                # preprocess image by expanding and subtracting mean RGB value
                image = np.expand_dims(image, axis=0)
                image = imagenet_utils.preprocess_input(image)

                # pass images thr network
                if vanilla:
                    features = models[0].predict(image)
                else:
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
        print('[INFO] Execution time of dataset {}: {}s logged into log_{}.txt'.format(i, exec_time, DATA_DIR.split('/')[-1]))

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

        print('[INFO] model {} metrics:'.format(subfolder))
        print('[INFO] Dataset {}\nAccuracy: {}\nPrecision: {}\nRecall: {}\nSaved to log'.format(i, accuracy, precision, recall))

        # write to txt log
        log.write('[INFO] {} Dataset, part {}:\n'.format(DATA_DIR.split('/')[-1], i + 1))
        log.write('[INFO] Execution time for {} images was {} seconds\n'.format(iterator, exec_time))
        log.write('Accuracy: {}\nPrecision: {}\nRecall: {}\n\n'.format(accuracy, precision, recall))
        log.flush()

        # write to local ods log
        dict_data_local['{} metrics'.format(DATA_DIR.split('/')[-1])].append([accuracy, precision, recall])
        # write to global ods log
        dict_data_global['{} dataset'.format(i + 1)].append([subfolder, accuracy, precision, recall])

    data_local.update(dict_data_local)
    save_data(DIR + "/log_{}.ods".format(DATA_DIR.split('/')[-1]), data_local)

    # save log
    log.close()

# save global log
data_global.update(dict_data_global)
save_data(MASTER + '/log_global.ods', data_global)

print('All done!')
