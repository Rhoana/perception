import numpy as np
import os
import sys

sys.path.append('../')
from util import Util

class Figure1:

  DELTA_MIN = 20
  DELTA_MAX = 80
  SIZE = (100, 100)

  @staticmethod
  def position_common_scale(flags=[False, False]):
    '''
    '''
    var_x = flags[0]
    var_spot = flags[1]


    sparse = None
    image = None
    label = None
    parameters = 1


    Y_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)
    X_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)

    Y, p = Util.parameter(Y_RANGE[0], Y_RANGE[1])
    parameters *= p

    X = Figure1.SIZE[1] / 2
    if var_x:
      X, p = Util.parameter(X_RANGE[0], X_RANGE[1])
      parameters *= p

    SPOT_SIZE = 5
    if var_spot:
      sizes = [1, 3, 5, 7, 9, 11]
      SPOT_SIZE = np.random.choice(sizes)
      parameters *= len(sizes)

    ORIGIN = 10

    sparse = [X, Y, SPOT_SIZE]

    image = np.zeros(Figure1.SIZE, dtype=np.bool)

    # draw axis
    image[Y_RANGE[0]:Y_RANGE[1], ORIGIN] = 1

    # draw spot
    half_spot_size = SPOT_SIZE / 2
    image[Y-half_spot_size:Y+half_spot_size+1, X-half_spot_size:X+half_spot_size+1] = 1

    label = Y - Figure1.DELTA_MIN

    return sparse, image, label, parameters



  @staticmethod
  def position_non_aligned_scale(flags=[False, False, False]):

    var_x = flags[0]
    var_y = flags[1]
    var_spot = flags[2]

    sparse = None
    image = None
    label = None
    parameters = 1

    OFFSET, p = Util.parameter(1, 11)
    # print 'OFFSET', OFFSET
    parameters *= p

    Y_RANGE = (Figure1.DELTA_MIN-OFFSET, Figure1.DELTA_MAX-OFFSET)
    X_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)

    Y = Figure1.SIZE[0] / 2
    if var_y:
      Y, p = Util.parameter(Y_RANGE[0], Y_RANGE[1])
      parameters *= p

    # print 'Y', Y

    X = Figure1.SIZE[1] / 2
    if var_x:
      X, p = Util.parameter(X_RANGE[0], X_RANGE[1])
      parameters *= p

    SPOT_SIZE = 5
    if var_spot:
      sizes = [1, 3, 5, 7, 9, 11]
      SPOT_SIZE = np.random.choice(sizes)
      parameters *= len(sizes)

    ORIGIN = 10

    sparse = [X, Y, SPOT_SIZE]

    image = np.zeros(Figure1.SIZE, dtype=np.bool)

    # draw axis
    image[Y_RANGE[0]:Y_RANGE[1], ORIGIN] = 1

    # draw spot
    half_spot_size = SPOT_SIZE / 2
    image[Y-half_spot_size:Y+half_spot_size+1, X-half_spot_size:X+half_spot_size+1] = 1

    label = Y + OFFSET - Figure1.DELTA_MIN


    return sparse, image, label, parameters


  @staticmethod
  def length(flags=[False, False, False]):

    var_x = flags[0]
    var_y = flags[1]
    var_width = flags[2]


    sparse = None
    image = None
    label = None
    parameters = 1

    length, p = Util.parameter(1,41)
    parameters *= p


    return sparse, image, label, parameters


  @staticmethod
  def direction(var_x=False, var_y=False):


    sparse = None
    image = None
    label = None
    parameters = 1


    return sparse, image, label, parameters



  @staticmethod
  def angle(var_x=False, var_y=False):


    sparse = None
    image = None
    label = None
    parameters = 1


    return sparse, image, label, parameters


  @staticmethod
  def area(var_x=False, var_y=False):



    sparse = None
    image = None
    label = None
    parameters = 1


    return sparse, image, label, parameters



  @staticmethod
  def volume(var_x=False, var_y=False):



    sparse = None
    image = None
    label = None
    parameters = 1


    return sparse, image, label, parameters




  @staticmethod
  def curvature(var_x=False, var_y=False):



    sparse = None
    image = None
    label = None
    parameters = 1


    return sparse, image, label, parameters



  @staticmethod
  def shading(var_x=False, var_y=False):



    sparse = None
    image = None
    label = None
    parameters = 1


    return sparse, image, label, parameters


