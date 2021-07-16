'''
This is slightly modified version of TensorFlows pruning method available in full at:

https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras?hl=en

It is to be used with full CNN models for comparison with FTL pruning capabilities.
'''


import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
import cv2
import h5py
import tempfile
from tensorflow.keras.optimizers import SGD
#%%
model_init = load_model('./model_init.h5')

# laod training dataaset
db_train = h5py.File('./train.hdf5')
X_train = db_train['images']
y_train_init = db_train['labels']

X_train = np.array(X_train)
y_train_init = np.array(y_train_init)

from sklearn.preprocessing import OneHotEncoder
data = np.asarray([[0], [1], [2], [3]])
enc = OneHotEncoder(sparse=False)
oneh = enc.fit(data)
y_train = enc.transform(y_train_init.reshape(-1, 1))

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_init, keras_file, include_optimizer=False)
#%%
# fine tuning with pruning
import tensorflow_model_optimization as tfmot

# Compute end step to finish pruning after 2 epochs.
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

batch_size = 16
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set.

num_images = X_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model_init, **pruning_params)

# `prune_low_magnitude` requires a recompile.
opt = SGD(lr=0.0001, decay=0.01/75, momentum=0.9, nesterov=True)

model_for_pruning.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_for_pruning.summary()

#%%
#Fine tune with pruning for two epochs.
logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(X_train, y_train,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)
#%%
# Create 3x smaller models from pruning
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

pruned_keras_file = './after_fit.h5'
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

pruned_tflite_file = './1_3.tflite'

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)

#%%
def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
#%%
#Create a 10x smaller model from combining pruning and quantization

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

quantized_and_pruned_tflite_file = './1_10.tflite'

with open(quantized_and_pruned_tflite_file, 'wb') as f:
  f.write(quantized_and_pruned_tflite_model)

print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file)))


#%%
from sklearn.preprocessing import OneHotEncoder
data = np.asarray([[0], [1], [2], [3]])
enc = OneHotEncoder(sparse=False)
oneh = enc.fit(data)
oneh_y = enc.transform(y_train.reshape(-1, 1))
n = 122
print(model_init.predict(np.expand_dims(X_train[n, :, :, :], axis=0)))
print()
print(np.argmax(model_init.predict(np.expand_dims(X_train[n, :, :, :], axis=0))))
print(y_train[n])
print(oneh_y[n])
