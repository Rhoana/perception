import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('../')
from util import Util

from figure1_position_common_scale import Figure1_Position_Common_Scale

class Figure1_Position_Nonaligned_Scale(Figure1_Position_Common_Scale):

  def __init__(self, variable_y=False, \
                     variable_x=False, \
                     variable_spot_size=False, \
                     variable_origin=True, \
                     constrain = True):

    self._variable_y = variable_y
    self._variable_origin = True

    super(Figure1_Position_Nonaligned_Scale, self).__init__(variable_x=variable_x, \
                                                            variable_spot_size=variable_spot_size, \
                                                            constrain=constrain)


  def create(self, verbose=False):
    '''
    Create a valid datapoint with variability settings in mind.
    '''

    parameters = 1

    #
    # Y is always variable since it encodes our value
    #
    Y = self._size[0] / 2
    if self._variable_y:
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
    if self._variable_origin:
      origin_y, p = Util.parameter(0, self._delta/2)
      parameters *= p

      origin[0] = origin_y
      #
      # NOTE: we do not support origin_x changes (not applicable here!)
      #


    # update the label
    self.X = X
    self.Y = Y
    self.ORIGIN = origin
    self.SPOT_SIZE = spot_size
    self.label = Y - self.ORIGIN[0]# label is the same as Y

    if verbose:
      print '# Parameters', parameters

  @staticmethod
  def from_sparse(sparse_representation, \
                  variable_y=False, \
                  variable_x=False, \
                  variable_spot_size=False, \
                  variable_origin=True, \
                  constrain = True):
    '''
    From sparse:
      [X, Y, origin, spot_size]
    '''

    fig = Figure1_Position_Nonaligned_Scale(variable_y=variable_y, \
                                            variable_x=variable_x, \
                                            variable_spot_size = variable_spot_size, \
                                            variable_origin=variable_origin, \
                                            constrain = constrain)

    fig.X = sparse_representation[0]
    fig.Y = sparse_representation[1]
    fig.ORIGIN = sparse_representation[2]
    fig.SPOT_SIZE = sparse_representation[3]

    # update the label
    fig.label = fig.Y - fig.ORIGIN[0]

    return fig


  def render_many(self, datapoints, framed=False):
    '''
    Renders many sparse representations as images.
    '''

    N = len(datapoints)

    images = np.zeros((N, 100, 100), dtype=np.bool)
    labels = np.zeros((N), dtype=np.uint8)

    for n in range(N):

      current = Figure1_Position_Nonaligned_Scale.from_sparse(datapoints[n], \
                                                              variable_y=self._variable_y, \
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

      current = Figure1_Position_Nonaligned_Scale.from_sparse(datapoints[n], \
                                                              variable_y=self._variable_y, \
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

      current = Figure1_Position_Nonaligned_Scale(variable_y=self._variable_y, \
                                                  variable_x=self._variable_x, \
                                                  variable_spot_size=self._variable_spot_size, \
                                                  constrain=self._constrain)

      current.create()

      datapoint, label = current.to_sparse()

      datapoints.append(datapoint)
      labels.append(label)

    return datapoints, labels











