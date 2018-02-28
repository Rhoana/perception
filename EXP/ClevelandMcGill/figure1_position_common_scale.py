import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('../')
from util import Util

class Figure1_Position_Common_Scale:

  def __init__(self, variable_x=False, \
                     variable_spot_size=False, \
                     constrain = True):

    '''
    '''
    self.X = None
    self.Y = None
    self.ORIGIN = [0, 0]
    self.SPOT_SIZE = None

    self.label = None

    self._delta = 20 # we use a bit larger delta here, it needs to be bigger
    # than the max. spot size / 2. for sure

    self._size = (100,100)
    self._spot_sizes = [1, 3, 5, 7, 9, 11]

    #
    # variability switches
    #
    #  stage 1: var. Y
    #  stage 2: var. WIDTH
    #  stage 3: var. X
    #    
    self._variable_x = variable_x
    self._variable_spot_size = variable_spot_size
    self._constrain = constrain


  def create(self, verbose=False):
    '''
    Create a valid datapoint with variability settings in mind.
    '''

    parameters = 1

    #
    # Y is always variable since it encodes our value
    #
    Y, p = Util.parameter(self._delta, self._size[0]-self._delta + 1)
    parameters *= p

    X = self._size[1] / 2
    if self._variable_x:
      X, p = Util.parameter(self._delta, self._size[1]-self._delta + 1)
      parameters *= p

    spot_size = 5
    if self._variable_spot_size:
      spot_size = np.random.choice(self._spot_sizes)
      parameters *= len(self._spot_sizes)

    origin = [0, 0]

    # update the label
    self.X = X
    self.Y = Y
    self.ORIGIN = origin
    self.SPOT_SIZE = spot_size
    self.label = Y # label is the same as Y


  def to_sparse(self):
    '''
    Convert to sparse representation
      [X, Y, origin, spot_size], label
    '''
    return [self.X, self.Y, self.ORIGIN, self.SPOT_SIZE], self.label


  @staticmethod
  def from_sparse(sparse_representation, \
                  variable_x=False, \
                  variable_spot_size=False, \
                  constrain = True):
    '''
    From sparse:
      [X, Y, origin, spot_size]
    '''

    fig = Figure1_Position_Common_Scale(variable_x=variable_x, \
                                        variable_spot_size = variable_spot_size, \
                                        constrain = constrain)

    fig.X = sparse_representation[0]
    fig.Y = sparse_representation[1]
    fig.ORIGIN = sparse_representation[2]
    fig.SPOT_SIZE = sparse_representation[3]

    # update the label
    fig.label = fig.Y

    return fig

  def to_image(self):
    '''
    '''
    image = np.zeros(self._size, dtype=np.bool)

    half_ss = self.SPOT_SIZE / 2 # this always floors

    image[self.Y-half_ss:self.Y+half_ss+1, self.X-half_ss:self.X+half_ss+1] = 1

    return image

  def show(self):
    '''
    '''
    print 'Label', self.label
    plt.figure()
    plt.imshow(self.to_image())


  def render_many(self, datapoints, framed=False):
    '''
    Renders many sparse representations as images.
    '''

    N = len(datapoints)

    images = np.zeros((N, 100, 100), dtype=np.bool)
    labels = np.zeros((N), dtype=np.uint8)

    for n in range(N):

      current = Figure1_Position_Common_Scale.from_sparse(datapoints[n], \
                                                          variable_x=self._variable_x, \
                                                          variable_spot_size=self._variable_spot_size, \
                                                          constrain=self._constrain)

      images[n] = current.to_image()

      labels[n] = current.label

    return images, labels


  def show_many(self, N=100):
    '''
    Creates many random examples with the current variability settings
    and show them.
    '''
    datapoints, labels = self.make_many(N)

    for n in range(N):

      current = Figure1_Position_Common_Scale.from_sparse(datapoints[n], \
                                                          variable_x=self._variable_x, \
                                                          variable_spot_size=self._variable_spot_size, \
                                                          constrain=self._constrain)



      current.show()

    return datapoints, labels


  def make_many(self, N=100):
    '''
    Creates many random examples with the current variability settings.
    '''

    datapoints = []
    labels = []

    for n in range(N):

      current = Figure1_Position_Common_Scale(variable_x=self._variable_x, \
                                              variable_spot_size=self._variable_spot_size, \
                                              constrain=self._constrain)

      current.create()

      datapoint, label = current.to_sparse()

      datapoints.append(datapoint)
      labels.append(label)

    return datapoints, labels





















