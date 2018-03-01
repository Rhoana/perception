from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.ensemble import RandomForestRegressor

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


EXPERIMENT = sys.argv[1] # Figure1_Position_Common_Scale
DATASET = int(sys.argv[2]) # 1
CLASSIFIER = sys.argv[3] # 'LeNet'
NORMALIZATION_FACTOR = float(sys.argv[4]) # e.g. 80

#
#
#
print 'Running', EXPERIMENT, 'on dataset', DATASET, 'with', CLASSIFIER

EPOCHS = 10
NO_SPLITS = 10
NO_REPEATS = 4
BATCH_SIZE = 32


if os.path.abspath('~').startswith('/n/'):
  # we are on the cluster
  PREFIX = '/n/regal/pfister_lab/PERCEPTION/'
else:
  PREFIX = '/home/d/PERCEPTION/'


#
# DATA LOADING
#

DATAPATH = PREFIX + 'DATA/' + EXPERIMENT + '/'

with open(DATAPATH + 'labels_'+str(DATASET)+'.p', 'r') as f:
  y = pickle.load(f)

if CLASSIFIER == 'RF':
  '''
  Load datapoints.
  '''
  with open(DATAPATH + 'datapoints_'+str(DATASET)+'.p', 'r') as f:
    X = pickle.load(f)

  # convert to numpy
  X_np = np.zeros((len(X), 3), dtype=np.uint8)
  for i,x in enumerate(X):
    X_np[i] = [x[0], x[1], x[3]]

  X = X_np

  print 'Loaded datapoints!'

elif CLASSIFIER == 'RF_Image' or CLASSIFIER == 'MLP' or CLASSIFIER == 'LeNet':
  '''
  Load images (1-channel). (This is already raveled!)
  '''
  X = np.load(DATAPATH + 'images_'+str(DATASET)+'.npy', mmap_mode='r')

elif CLASSIFIER == 'VGG19':
  '''
  Load VGG19 features. 
  '''
  X = np.load(DATAPATH + 'vgg19_features'+str(DATASET)+'.npy', mmap_mode='r')

elif CLASSIFIER == 'Xception':
  '''
  Load Xception features.
  '''
  X = np.load(DATAPATH + 'xception_features'+str(DATASET)+'.npy', mmap_mode='r')

#
#
# OUTPUT PATH
#
#
OUTPUT_DIRECTORY = os.path.join(PREFIX, 'RESULTS_NEW', EXPERIMENT, str(DATASET), CLASSIFIER)
if not os.path.exists(OUTPUT_DIRECTORY):
  os.makedirs(OUTPUT_DIRECTORY)


#
#
# CROSS VALIDATION
#
#
INDEX = 0
kfold = RepeatedStratifiedKFold(n_splits=NO_SPLITS, n_repeats=NO_REPEATS)
for train, test in kfold.split(X, y):

  #
  # Let's check if we have some partial results here
  #
  current_output = OUTPUT_DIRECTORY + '/' + str(EPOCHS) + '_' + str(NO_SPLITS) + '_' + str(NO_REPEATS) + '_' + str(INDEX) + '_results.p'
  if os.path.exists(current_output):
    print current_output, 'exists.. skipping this run!'
    INDEX += 1
    continue

  t0 = time.time()


  #
  # NOW THE DIFFERENT CLASSIFIERS
  #
  if CLASSIFIER == 'RF' or CLASSIFIER == 'RF_Image':

    #
    # setup random forest
    #

    rf = RandomForestRegressor(n_estimators=128)

    y_normalized = y[train] / NORMALIZATION_FACTOR
    rf.fit(X[train], y_normalized)

    stats = {}
    predictor = rf

  else:
    #
    # All other classifiers are ending in an MLP
    #
    
    if CLASSIFIER == 'LeNet':
      # the LeNet 5
      classifier = models.Sequential()
      classifier.add(layers.Convolution2D(20, (5, 5), padding="same", input_shape=(100, 100, 1)))
      classifier.add(layers.Activation("relu"))
      classifier.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
      classifier.add(layers.Dropout(0.2))
      classifier.add(layers.Convolution2D(50, (5, 5), padding="same"))
      classifier.add(layers.Activation("relu"))
      classifier.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
      classifier.add(layers.Dropout(0.2))

      oshape = classifier.output_shape
      classifier.add(layers.Flatten())

      print 'reshaping'
      X_ready = X[train].reshape(len(train), 100, 100, 1)
      print 'reshaping done!'

      MLP = classifier

    elif CLASSIFIER == 'MLP':
      oshape = X.shape[1] # since it is already raveled

      # create a new MLP
      MLP = models.Sequential()

      X_ready = X[train]


    elif CLASSIFIER == 'VGG19' or CLASSIFIER == 'Xception':
      oshape = X.shape[1] * X.shape[2] * X.shape[3]

      # create a new MLP
      MLP = models.Sequential()

      X_ready = X[train].reshape(len(train), oshape)

    MLP.add(layers.Dense(256, activation='relu', input_dim=oshape))
    MLP.add(layers.Dropout(0.5))
    MLP.add(layers.Dense(1, activation='linear')) # REGRESSION

    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    MLP.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse', 'mae']) # MSE for regression

    #
    # TODO we need to normalize by dividing by 80
    #
    y_normalized = y[train] / NORMALIZATION_FACTOR

    history = MLP.fit(X_ready, \
                      y_normalized, \
                      epochs=EPOCHS, \
                      batch_size=BATCH_SIZE, \
                      validation_split=0.25,
                      verbose=True)

    stats = dict(history.history)

    print 'Training done!'

    predictor = MLP

  #
  # Fitting is done!
  #
  fit_time = time.time() - t0

  y_pred = predictor.predict(X[test])
  print 'Prediction done!'  

  y_test_normalized = y[test] / NORMALIZATION_FACTOR
  
  #
  # update stats dict
  #
  stats['time'] = fit_time
  stats['y_test'] = y_test_normalized
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
  if CLASSIFIER != 'RF' and CLASSIFIER != 'RF_Image':
    predictor.save(mlp_outputfile)
  else:
    with open(mlp_outputfile.replace('hdf5','p'), 'w') as f:
      # RANDOM FORESTS ARE PICKLED
      pickle.dump(predictor, f)

  print 'Stored', stats_outputfile
  print '...and', mlp_outputfile

  INDEX += 1

#
# THE END
#
print 'All done!'
