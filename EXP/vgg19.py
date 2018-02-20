from keras import models
from keras import layers
from keras import optimizers
from keras.utils.np_utils import to_categorical
import keras.applications.vgg19

import os

from sklearn.metrics import classification_report
from sklearn.preprocessing import binarize
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

from classifier import Classifier

class VGG19(Classifier):

  def __init__(self):
    '''
    '''

    Classifier.__init__(self)

    self.name = 'VGG19 (Imagenet pre-trained) + MLP'

    self._VGG19 = None
    self._MLP = None

    self._VGG19_features_for_kfold = None
    self._VGG19_features_calculated = False

  #
  # setup MLP to fit in sklearn framework
  #
  @staticmethod
  def create_mlp():

    MLP = models.Sequential()
    MLP.add(layers.Dense(256, activation='relu', input_dim=3 * 3 * 512))
    MLP.add(layers.Dropout(0.5))
    MLP.add(layers.Dense(2, activation='softmax'))

    MLP.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])    

    return MLP


  def setup(self, verbose=True):
    '''
    '''

    Classifier.setup(self)

    #
    # setup VGG
    #
    self._VGG19 = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(100,100,3))

    self._MLP = VGG19.create_mlp()

    print 'Configured VGG19 (imagenet weights) and a MLP..'


  def show(self):
    '''
    '''
    
    Classifier.show(self)

    self._VGG19.summary()

    if self._MLP:
      self._MLP.summary()


  def train(self, X_train, y_train, epochs=1, batch_size=32, validation_split=.25, verbose=True):
    '''
    '''

    Classifier.train(self)

    #
    # We generate features using VGG19 (pre-trained)
    #
    if verbose:
      print 'Generating features for X_train using VGG19..'
    features_train = self._VGG19.predict(X_train, verbose=verbose)

    #
    # .. and we then fit our MLP
    #
    if verbose:
      print 'Fitting the MLP..'
    self._MLP.fit(features_train.reshape(X_train.shape[0], 3 * 3 * 512), \
                  to_categorical(y_train), \
                  epochs=epochs, \
                  batch_size=batch_size, \
                  validation_split=validation_split, \
                  verbose=verbose)


  def test(self, X_test, y_test, verbose=True):
    '''
    '''

    Classifier.test(self)

    #
    # We generate features for X_test
    #
    if verbose:
      print 'Generating features for X_test using VGG19..'
    features_test = self._VGG19.predict(X_test, verbose=verbose)

    #
    # .. predict using MLP
    #
    if verbose:
      print 'Predict using the MLP..'
    y_predict = self._MLP.predict(features_test.reshape(X_test.shape[0], 3*3*512), verbose=verbose)

    print classification_report(y_test, binarize(y_predict, threshold=.5)[:,1])

    return y_predict


  def store(self, filename):
    '''
    '''

    Classifier.store(self)

    if not os.path.exists(filename):
      print 'Storing', filename
      self._MLP.save(filename)
    else:
      print 'File exists! NOT STORED!'


  def run_kfold(self, X_train, y_train, n_splits=10, n_repeats=10, epochs=1, batch_size=32, verbose=False):

    #
    # We generate features using VGG19 (pre-trained)
    #
    if not self._VGG19_features_calculated:
      if verbose:
        print 'Generating features for X_train using VGG19..'
      features_train = self._VGG19.predict(X_train, verbose=verbose)
      self._VGG19_features_for_kfold = features_train
      self._VGG19_features_calculated = True
    else:
      features_train = self._VGG19_features_for_kfold

    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

    X = features_train.reshape(X_train.shape[0], 3*3*512)
    y = y_train

    results = []

    for train, test in kfold.split(X, y):

      MLP = models.Sequential()
      MLP.add(layers.Dense(256, activation='relu', input_dim=3 * 3 * 512))
      MLP.add(layers.Dropout(0.5))
      MLP.add(layers.Dense(2, activation='softmax'))

      MLP.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

      history = MLP.fit(X[train], \
                        to_categorical(y[train]), \
                        epochs=epochs, \
                        batch_size=batch_size, \
                        validation_split=0.25,
                        verbose=verbose)

      scores = MLP.evaluate(X[test], to_categorical(y[test]), verbose=verbose)

      stats = dict(history.history)

      stats['test_loss'] = scores[0]
      stats['test_acc'] = scores[1]

      results.append(stats)

    return results

    # # we create the keras classifier in the sklearn sense
    # estimator = KerasClassifier(build_fn=VGG19.create_mlp, \
    #                             epochs=epochs, \
    #                             batch_size=batch_size, \
    #                             verbose=verbose)

    # kfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=31337)

    # results = cross_val_score(estimator, \
    #                           features_train.reshape(X_train.shape[0], 3*3*512), \
    #                           y_train, \
    #                           cv=kfold)

    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    # return results

