import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('../')
from util import Util

class Figure1_Length(object):

  def __init__(self, variable_x=False, \
                     variable_y=False, \
                     variable_width=False, \
                     constrain = True):

    '''
    '''
    self.X = None
    self.Y = None
    self.LENGTH = None
    self.WIDTH = None

    self.label = None

    self._delta = 20 # we use a bit larger delta here, it needs to be bigger
    # than the max. spot size / 2. for sure

    self._size = (100,100)
    self._widths = [1, 3, 5, 7, 9, 11]

    #
    # variability switches
    #
    #  stage 1: var. X
    #  stage 2: var. Y
    #  stage 3: var. WIDTH
    #    
    self._variable_x = variable_x
    self._variable_y = variable_y
    self._variable_width = variable_width
    self._constrain = constrain


  def create(self, verbose=False):
    '''
    Create a valid datapoint with variability settings in mind.
    '''

    parameters = 1

    length, p = Util.parameter(1,41)
    parameters *= p

    Y = self._size[0] / 2
    if self._variable_y:
      Y, p = Util.parameter(self._delta, self._size[0]-self._delta + 1)
      parameters *= p

    X = self._size[1] / 2
    if self._variable_x:
      X, p = Util.parameter(self._delta, self._size[1]-self._delta + 1)
      parameters *= p

    width = 1
    if self._variable_width:
      width = np.random.choice(self._widths)
      parameters *= len(self._widths)

    # update the label
    self.X = X
    self.Y = Y
    self.LENGTH = length
    self.WIDTH = width
    self.label = length

    if verbose:
      print '# Parameters', parameters


  def to_sparse(self):
    '''
    Convert to sparse representation
      [X, Y, origin, spot_size], label
    '''
    return [self.X, self.Y, self.LENGTH, self.WIDTH], self.label


  @staticmethod
  def from_sparse(sparse_representation, \
                  variable_x=False, \
                  variable_y=False, \
                  variable_width=False, \
                  constrain = True):
    '''
    From sparse:
      [X, Y, length, width]
    '''

    fig = Figure1_Length(variable_x=variable_x, \
                         variable_y=variable_y, \
                         variable_width = variable_width, \
                         constrain = constrain)

    fig.X = sparse_representation[0]
    fig.Y = sparse_representation[1]
    fig.LENGTH = sparse_representation[2]
    fig.WIDTH = sparse_representation[3]

    # update the label
    fig.label = fig.LENGTH

    return fig

  def to_image(self):
    '''
    '''
    image = np.zeros(self._size, dtype=np.bool)

    half_width = self.WIDTH / 2 # this always floors
    half_length = self.LENGTH / 2

    image[self.Y-half_length:self.Y+half_length+1, self.X-half_width:self.X+half_width+1] = 1

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

      current = Figure1_Length.from_sparse(datapoints[n], \
                                            variable_x=self._variable_x, \
                                            variable_y=self._variable_y, \
                                            variable_width=self._variable_width, \
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

      current = Figure1_Length.from_sparse(datapoints[n], \
                                            variable_x=self._variable_x, \
                                            variable_y=self._variable_y, \
                                            variable_width=self._variable_width, \
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

      current = Figure1_Length(variable_x=self._variable_x, \
                               variable_y=self._variable_y, \
                               variable_width=self._variable_width, \
                               constrain=self._constrain)

      current.create()

      datapoint, label = current.to_sparse()

      datapoints.append(datapoint)
      labels.append(label)

    return datapoints, labels





















