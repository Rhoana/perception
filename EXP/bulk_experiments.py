from sklearn.model_selection import RepeatedStratifiedKFold 
from keras import models
from keras import layers
from keras import optimizers
import keras.applications
from keras.utils.np_utils import to_categorical

import cPickle as pickle
import numpy as np
import os
import sys
import time

import ClevelandMcGill as C


EXPERIMENT = sys.argv[1] #'Figure12'
DATASET = sys.argv[2] # 1
FEATUREGENERATOR = sys.argv[3] #'LeNet'
FRAMED = sys.argv[4] # 'Framed'

#
#
#
print 'Running', EXPERIMENT, 'on dataset', DATASET, FRAMED, 'with', FEATUREGENERATOR


EPOCHS = 10
FRAMED_BOOL = (FRAMED == 'Framed')
NO_SPLITS = 10
NO_REPEATS = 4
BATCH_SIZE = 32

FEATUREGENERATORS = {'MLP': 'Just MLP',
                     'LeNet': 'LeNet',
                     'VGG19': keras.applications.VGG19,
                     'Xception': keras.applications.Xception,
                     'DenseNet201': keras.applications.DenseNet201}


#
#
# We store now a folder structure:
#   RESULTS/Experiment/Dataset/FeatureGenerator/Framed/EPOCHS_NOSPLITS_NOREPEATS_INDEX.p
#
# INDEX is the index of the current run
#
OUTPUT_DIRECTORY = os.path.join('../RESULTS_NEW/', EXPERIMENT, str(DATASET), FEATUREGENERATOR, FRAMED)
if not os.path.exists(OUTPUT_DIRECTORY):
  os.makedirs(OUTPUT_DIRECTORY)


#
#
# LOAD DATA
#
#
if EXPERIMENT == 'Figure12':
  images, framed_images, labels = C.Figure12.load(dataset=int(DATASET),preview=False)

  if FRAMED_BOOL:
    X = framed_images
  else:
    X = images

  X = np.stack((X,)*3, -1) # make sure we have RGB
  X = X.astype(np.float32)-.5 # normalize
  y = labels


#
#
# GENERATE FEATURES
#
#
feature_time = 0

if FEATUREGENERATOR != 'LeNet':
  # either pre-classifier or MLP
  if FEATUREGENERATOR != 'MLP':
    
    classifier = FEATUREGENERATORS[FEATUREGENERATOR](include_top=False, weights='imagenet', input_shape=(100,100,3))

    t0 = time.time()

    features = classifier.predict(X, verbose=True)

    feature_time = time.time() - t0

    oshape = classifier.output_shape

    X = features.reshape(X.shape[0], oshape[1]*oshape[2]*oshape[3])

  else:
    # MLP case.. no feature generation here!
    oshape = X.shape
    # we now need to flatten the data
    X = X.reshape(X.shape[0], oshape[1]*oshape[2]*oshape[3])

INDEX = 0
kfold = RepeatedStratifiedKFold(n_splits=NO_SPLITS, n_repeats=NO_REPEATS)
for train, test in kfold.split(X, y):

  t0 = time.time()
  
  if FEATUREGENERATOR == 'LeNet':
    # the LeNet 5
    classifier = models.Sequential()
    classifier.add(layers.Convolution2D(20, (5, 5), border_mode="same", input_shape=(100, 100, 3)))
    classifier.add(layers.Activation("relu"))
    classifier.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    classifier.add(layers.Dropout(0.2))
    classifier.add(layers.Convolution2D(50, (5, 5), border_mode="same"))
    classifier.add(layers.Activation("relu"))
    classifier.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    classifier.add(layers.Dropout(0.2))

    oshape = classifier.output_shape
    classifier.add(layers.Flatten())

    MLP = classifier
  else:
    # create a new MLP
    MLP = models.Sequential()

  MLP.add(layers.Dense(256, activation='relu', input_dim=oshape[1]*oshape[2]*oshape[3]))
  MLP.add(layers.Dropout(0.5))
  MLP.add(layers.Dense(2, activation='softmax'))

  sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
  MLP.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

  history = MLP.fit(X[train], \
                    to_categorical(y[train]), \
                    epochs=EPOCHS, \
                    batch_size=BATCH_SIZE, \
                    validation_split=0.25,
                    verbose=True)

  print 'Training done!'

  scores = MLP.evaluate(X[test], to_categorical(y[test]), verbose=True)

  print 'Scoring done!'

  y_pred = MLP.predict(X[test])

  print 'Prediction done!'

  stats = dict(history.history)

  stats['test_loss'] = scores[0]
  stats['test_acc'] = scores[1]

  fit_time = time.time() - t0
  stats['time'] = feature_time + fit_time

  stats['y_test'] = y[test]
  stats['y_pred'] = y_pred
  
  print 'Created stats..'

  #
  # Every sample we create two files
  #   a) the stats file as a lil pickle
  #   b) and the network weights so we can re-run any experiment without retraining
  stats_outputfile = OUTPUT_DIRECTORY + '/' + str(EPOCHS) + '_' + str(NO_SPLITS) + '_' + str(NO_REPEATS) + '_' + str(INDEX) + '_results.p'
  with open(stats_outputfile, 'w') as f:
    pickle.dump(stats, f)
  mlp_outputfile = OUTPUT_DIRECTORY + '/' + str(EPOCHS) + '_' + str(NO_SPLITS) + '_' + str(NO_REPEATS) + '_' + str(INDEX) + '_mlp.hdf5'
  MLP.save(mlp_outputfile)

  print 'Stored', stats_outputfile
  print '...and', mlp_outputfile

  INDEX += 1





