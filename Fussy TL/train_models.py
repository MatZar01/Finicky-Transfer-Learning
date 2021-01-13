#!python3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import h5py
import os
from imutils import paths
import sys

args = sys.argv

if len(args) == 1:
    DIR = './models'
else:
    DIR = args[1]

all_pts = list(paths.list_files(DIR))
db_pts = []
for pt in all_pts:
    if '.hdf5' in pt:
        db_pts.append(pt)

jobs = 1

for db_pt in db_pts:
    modelPath = db_pt.replace('hdf5', 'cpickle')

    # open database
    db = h5py.File(db_pt, "r")
    # and set the training / testing split index
    i = int(db["labels"].shape[0] * 0.75)

    # train Logistic Regression classifier
    print("Tuning hyperparameters...")
    params = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]}
    model = GridSearchCV(LogisticRegression(max_iter=2000), params, cv=4, n_jobs=jobs, verbose=5)
    model.fit(db["features"][:i], db["labels"][:i])
    print("Best hyperparameter: {}".format(model.best_params_))

    # evaluate model
    print("Evaluating...")
    preds = model.predict(db["features"][i:])
    print(classification_report(db["labels"][i:], preds, target_names=db["label_names"]))

    # save model to diskpyt 
    print("Saving model...")
    f = open(modelPath, "wb")
    f.write(pickle.dumps(model.best_estimator_))
    f.close()

    db.close()
