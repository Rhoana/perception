import matplotlib
matplotlib.use('agg')
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt    

import os, glob, sys
import cPickle as pickle

import numpy as np

from keras.models import load_model
from keras import backend as K
import sklearn.metrics
import keras.applications
import itertools

import ClevelandMcGill as C
from util import Util


EXPERIMENT = sys.argv[1]#'C.Figure1.curvature'
CLASSIFIER = sys.argv[2]#'LeNet'
NETWORK_INDEX = sys.argv[3]#4


DATATYPE = eval(EXPERIMENT)


RESULTS_DIR = '/n/regal/pfister_lab/PERCEPTION/RESULTS/' + EXPERIMENT + '/'
OUTPUT_DIR = '/n/regal/pfister_lab/PERCEPTION/CROSSNETWORK/' + EXPERIMENT +'/'
if not os.path.exists(OUTPUT_DIR):
  # here can be a race condition
  try:
    os.makedirs(OUTPUT_DIR)
  except:
    print 'Race condition!', os.path.exists(OUTPUT_DIR)





all_labels = {'C.Figure1.position_common_scale': ['Position Y', '+ Position X', '+ Spotsize'], \
              'C.Figure1.position_non_aligned_scale': ['Scale', '+ Position Y', '+ Position X', '+ Spotsize'],\
              'C.Figure1.length': ['Length', '+ Position Y', '+ Position X', '+ Width'], \
              'C.Figure1.direction': ['Direction', '+ Position Y', '+ Position X'], \
              'C.Figure1.angle': ['Angle', '+ Position Y', '+ Position X'], \
              'C.Figure1.area': ['Area', '+ Position Y', '+ Position X'], \
              'C.Figure1.volume': ['Volume', '+ Position Y', '+ Position X'], \
              'C.Figure1.curvature': ['Curvature', '+ Position Y', '+ Position X', '+ Width'], \
              'C.Figure1.shading': ['Shading', '+ Position Y', '+ Position X']
              }
DATASETS = len(all_labels[EXPERIMENT])


networks = []

for x in range(DATASETS):
    networks.append(load_model(RESULTS_DIR+str(x)+'/'+CLASSIFIER+'/'+str(NETWORK_INDEX).zfill(2)+'_noise.h5'))





# generate images
N = 20000
X = [np.zeros((N, 100, 100), dtype=np.float32)] * DATASETS
y = [np.zeros((N), dtype=np.float32)] * DATASETS

for x in range(DATASETS):
    FLAGS = [False] * 10 # never more than 10 flags
    for f in range(x):
      FLAGS[f] = True            
    print 'Generating with', FLAGS
    
    for n in range(N):

        sparse, image, label, parameters = DATATYPE(FLAGS)

        image = image.astype(np.float32)
        image += np.random.uniform(0, 0.05,(100,100))
            
        X[x][n] = image
        y[x][n] = label

# normalize
for i in range(DATASETS):
    
    X_min = X[i].min()
    X_max = X[i].max()

    # scale in place
    X[i] -= X_min
    X[i] /= (X_max - X_min)
    X[i] -= .5

    y_min = y[i].min()
    y_max = y[i].max()

    y[i] -= y_min
    y[i] /= (y_max - y_min)


results = [[-1]*DATASETS]*DATASETS

for data in range(DATASETS):

  network_results = []

  for network in range(DATASETS):

    if CLASSIFIER == 'XCEPTION':

      # make 3d
      X_new = np.stack((X[data],)*3, -1)
      
      feature_generator = keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(100,100,3))
      X_test_3D_features = feature_generator.predict(X_new, verbose=True)
      
      feature_shape = X_test_3D_features.shape[1]*X_test_3D_features.shape[2]*X_test_3D_features.shape[3]
      
      reshaped_data = X_test_3D_features.reshape(len(X_test_3D_features), feature_shape)

    elif CLASSIFIER == 'VGG19':

      feature_generator = keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(100,100,3))
      X_test_3D_features = feature_generator.predict(X_new, verbose=True)
      
      feature_shape = X_test_3D_features.shape[1]*X_test_3D_features.shape[2]*X_test_3D_features.shape[3]
      
      reshaped_data = X_test_3D_features.reshape(len(X_test_3D_features), feature_shape)

    elif CLASSIFIER == 'LeNet':

      reshaped_data = X[data].reshape(len(X[data]), 100, 100, 1)

    elif CLASSIFIER == 'MLP':

      feature_shape = 100*100
      reshaped_data = X[data].reshape(len(X[data]), feature_shape)

    else:
      print 'WRONG CLASSIFIER!'
      sys.exit(1)

    y_pred = networks[network].predict(reshaped_data)
    y_test = y[data]

    MLAE = np.log2(sklearn.metrics.mean_absolute_error(y_pred*100, y_test*100)+.125)

    mae = sklearn.metrics.mean_absolute_error(y_pred, y_test)
    
    print 'predicted data', data, 'with network', network, 'MLAE', MLAE, 'MAE', mae
    
    network_results.append(MLAE)

    
  results[data] = network_results


pickle_file = OUTPUT_DIR + '/' + str(CLASSIFIER)+'_'+str(NETWORK_INDEX)+'.pdf'
with open(pickle_file, 'w') as f:
  pickle.dump(results, f)
print 'Stored', pickle_file



def plot_cross_network(cm, classes, title, cmap=plt.cm.Blues, c_name='', e_name='', c_index=-1):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Dataset')
    plt.xlabel(c_name+' Training')
    plt.savefig('../PAPER/SUPPLEMENTAL/gfx/cn_'+str(e_name)+'_'+str(c_name)+'_'+str(c_index)+'.pdf', bbox_inches='tight', pad_inches=0)
    
plt.figure()
plot_cross_network(np.array(results), classes=all_labels[EXPERIMENT],
                      title='Cross Network Evaluation (MLAE)', c_name=CLASSIFIER, e_name=EXPERIMENT, c_index=NETWORK_INDEX)

print 'Stored image!!'
