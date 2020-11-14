#!python3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import h5py
import os

DIR = './models_trained/20_perc'
databasePath = DIR + '/features.hdf5'
modelPath = DIR + '/fussy_model.cpickle'
if os.path.exists(modelPath):
    os.remove(modelPath)
jobs = 1

# open database
db = h5py.File(databasePath, "r")
# and set the training / testing split index
i = int(db["labels"].shape[0] * 0.75)

# train Logistic Regression classifier
print("Tuning hyperparameters...")
params = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
model = GridSearchCV(LogisticRegression(max_iter=700), params, cv=4, n_jobs=jobs, verbose=20)
model.fit(db["features"][:i], db["labels"][:i])
print("Best hyperparameter: {}".format(model.best_params_))

# evaluate model
print("Evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds, target_names=db["label_names"]))

# save model to disk
print("Saving model...")
f = open(modelPath, "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

db.close()

