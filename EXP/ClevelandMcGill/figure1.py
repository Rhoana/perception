import math
import numpy as np
import os
import skimage.draw
import sys

sys.path.append('../')
from util import Util

class Figure1:

  DELTA_MIN = 20
  DELTA_MAX = 80
  SIZE = (100, 100)

  @staticmethod
  def position_common_scale(flags=[False, False], preset=None):
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

    if preset:
      Y = preset

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

    sparse = [Y, X, SPOT_SIZE]

    image = np.zeros(Figure1.SIZE, dtype=np.bool)

    # draw axis
    image[Y_RANGE[0]:Y_RANGE[1], ORIGIN] = 1

    # draw spot
    half_spot_size = SPOT_SIZE / 2
    image[Y-half_spot_size:Y+half_spot_size+1, X-half_spot_size:X+half_spot_size+1] = 1

    label = Y - Figure1.DELTA_MIN

    return sparse, image, label, parameters



  @staticmethod
  def position_non_aligned_scale(flags=[False, False, False], preset=None):

    var_y = flags[0]
    var_x = flags[1]
    var_spot = flags[2]

    sparse = None
    image = None
    label = None
    parameters = 1

    OFFSET, p = Util.parameter(-9, 11)
    # print 'OFFSET', OFFSET
    parameters *= p

    if preset:
      OFFSET = preset

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

    sparse = [Y, X, SPOT_SIZE]

    image = np.zeros(Figure1.SIZE, dtype=np.bool)

    # draw axis
    image[Y_RANGE[0]:Y_RANGE[1], ORIGIN] = 1

    # draw spot
    half_spot_size = SPOT_SIZE / 2
    image[Y-half_spot_size:Y+half_spot_size+1, X-half_spot_size:X+half_spot_size+1] = 1

    label = Y + OFFSET - Figure1.DELTA_MIN


    return sparse, image, label, parameters


  @staticmethod
  def length(flags=[False, False, False], preset=None):

    var_y = flags[0]
    var_x = flags[1]
    var_width = flags[2]


    sparse = None
    image = None
    label = None
    parameters = 1

    Y_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)
    X_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)

    LENGTH, p = Util.parameter(1, Y_RANGE[1]-Y_RANGE[0]+1) # 1..60
    parameters *= p

    if preset:
      LENGTH = preset

    MAX_LENGTH = Y_RANGE[1]-Y_RANGE[0]
    # print 'Max length', MAX_LENGTH

    X = math.floor(Figure1.SIZE[1] / 2)
    if var_x:
      X, p = Util.parameter(X_RANGE[0], X_RANGE[1])
      parameters *= p

    Y = Y_RANGE[0]
    if var_y:
      
      Y, p = Util.parameter(0, Figure1.SIZE[0]-MAX_LENGTH)
      # print 'Y',Y
      parameters *= p

    WIDTH = 1
    if var_width:
      sizes = [1, 3, 5, 7, 9, 11]
      WIDTH = np.random.choice(sizes)
      parameters *= len(sizes)

    sparse = [Y, X, LENGTH, WIDTH]

    image = np.zeros(Figure1.SIZE, dtype=np.bool)


    half_width = math.floor(WIDTH / 2) # this always floors
    
    # print(Y,LENGTH,X,half_width,WIDTH)
    image[Y:Y+LENGTH, X-half_width:X+half_width+1] = 1


    label = LENGTH

    return sparse, image, label, parameters


  @staticmethod
  def direction(flags=[False, False], preset=None):

    var_y = flags[0]
    var_x = flags[1]


    sparse = None
    image = None
    label = None
    parameters = 1

    Y_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)
    X_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)

    DOF = 360
    DIRECTION = np.random.randint(DOF)
    parameters *= DOF

    if preset:
      DIRECTION = preset


    theta = -(np.pi / 180.0) * DIRECTION
    LENGTH = Figure1.DELTA_MIN



    X = Figure1.SIZE[1] / 2
    if var_x:
      X, p = Util.parameter(X_RANGE[0], X_RANGE[1])
      parameters *= p

    Y = Figure1.SIZE[0] / 2
    if var_y:
      Y, p = Util.parameter(Y_RANGE[0], Y_RANGE[1])
      parameters *= p

    END = (Y - LENGTH * np.cos(theta), X - LENGTH * np.sin(theta))

    sparse = [Y, X, DIRECTION, LENGTH]

    image = np.zeros(Figure1.SIZE, dtype=np.bool)

    # draw direction
    rr, cc = skimage.draw.line(Y, X, int(np.round(END[0])), int(np.round(END[1])))
    image[rr, cc] = 1

    # Draw origin spot
    half_spot_size = 1 #equals spot size of 3
    image[Y-half_spot_size:Y+half_spot_size+1, X-half_spot_size:X+half_spot_size+1] = 1

    label = DIRECTION

    return sparse, image, label, parameters



  @staticmethod
  def angle(flags=[False,False], preset=None):


    var_y = flags[0]
    var_x = flags[1]


    sparse = None
    image = None
    label = None
    parameters = 1


    Y_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)
    X_RANGE = (Figure1.DELTA_MIN, Figure1.DELTA_MAX)

    DOF = 90
    ANGLE = np.random.randint(1, DOF+1)
    parameters *= DOF

    if preset:
      ANGLE=preset

    LENGTH = Figure1.DELTA_MIN

    X = Figure1.SIZE[1] / 2
    if var_x:
      X, p = Util.parameter(X_RANGE[0], X_RANGE[1])
      parameters *= p

    Y = Figure1.SIZE[0] / 2
    if var_y:
      Y, p = Util.parameter(Y_RANGE[0], Y_RANGE[1])
      parameters *= p

    image = np.zeros(Figure1.SIZE, dtype=np.bool)

    # first line
    first_angle = np.random.randint(360)
    theta = -(np.pi / 180.0) * first_angle
    END = (Y - LENGTH * np.cos(theta), X - LENGTH * np.sin(theta))
    rr, cc = skimage.draw.line(Y, X, int(np.round(END[0])), int(np.round(END[1])))
    image[rr, cc] = 1

    second_angle = first_angle+ANGLE
    theta = -(np.pi / 180.0) * second_angle
    END = (Y - LENGTH * np.cos(theta), X - LENGTH * np.sin(theta))
    rr, cc = skimage.draw.line(Y, X, int(np.round(END[0])), int(np.round(END[1])))
    image[rr, cc] = 1

    sparse = [Y, X, ANGLE, first_angle]

    label = ANGLE

    return sparse, image, label, parameters


  @staticmethod
  def area(flags=[False, False], preset=None):



    var_y = flags[0]
    var_x = flags[1]


    sparse = None
    image = None
    label = None
    parameters = 1


    Y_RANGE = (40,60)#(Figure1.DELTA_MIN, Figure1.DELTA_MAX)
    X_RANGE = (40,60)#(Figure1.DELTA_MIN, Figure1.DELTA_MAX)

    DOF = 40
    RADIUS = np.random.randint(1, DOF+1)
    parameters *= DOF

    if preset:
      RADIUS = preset

    X = Figure1.SIZE[1] / 2
    if var_x:
      X, p = Util.parameter(X_RANGE[0], X_RANGE[1])
      parameters *= p

    Y = Figure1.SIZE[0] / 2
    if var_y:
      Y, p = Util.parameter(Y_RANGE[0], Y_RANGE[1])
      parameters *= p

    image = np.zeros(Figure1.SIZE, dtype=np.bool)

    rr, cc = skimage.draw.ellipse_perimeter(Y, X, RADIUS, RADIUS)
    image[rr, cc] = 1

    sparse = [Y, X, RADIUS]

    label = np.pi * RADIUS * RADIUS


    return sparse, image, label, parameters



  @staticmethod
  def volume(flags=[False, False], preset=None):



    var_y = flags[0]
    var_x = flags[1]


    sparse = None
    img = None
    label = None
    parameters = 1


    DOF = 20
    DEPTH = np.random.randint(1, DOF+1)
    parameters *= DOF

    if preset:
      DEPTH = preset

    Y_RANGE = (Figure1.DELTA_MIN+DOF, Figure1.DELTA_MAX-DOF)
    X_RANGE = (Figure1.DELTA_MIN+DOF, Figure1.DELTA_MAX-DOF)


    def obliqueProjection( point ):
        angle = -45.
        alpha = (np.pi / 180.0) * angle;
        
        P = [[1, 0, (1/2.)*np.sin(alpha)],
             [0, 1, (1/2.)*np.cos(alpha)],
             [0, 0, 0]]

        ss = np.dot(P, point);
        
        return [int(np.round(ss[0])), int(np.round(ss[1]))]


    X = Figure1.SIZE[1] / 2
    if var_x:
      X, p = Util.parameter(X_RANGE[0], X_RANGE[1])
      parameters *= p

    Y = Figure1.SIZE[0] / 2
    if var_y:
      Y, p = Util.parameter(Y_RANGE[0], Y_RANGE[1])
      parameters *= p

    # print X,Y,DEPTH

    img = np.zeros(Figure1.SIZE, dtype=np.bool)

    front_bottom_left = (Y, X)

    front_bottom_right = (front_bottom_left[0], front_bottom_left[1]+DEPTH)
    rr, cc = skimage.draw.line(front_bottom_left[0], front_bottom_left[1], front_bottom_right[0], front_bottom_right[1])
    img[rr,cc] = 1

    front_top_left = (front_bottom_left[0]-DEPTH, front_bottom_left[1])
    front_top_right = (front_bottom_right[0]-DEPTH, front_bottom_right[1])
    rr, cc = skimage.draw.line(front_top_left[0], front_top_left[1], front_top_right[0], front_top_right[1])
    img[rr,cc] = 1

    rr, cc = skimage.draw.line(front_top_left[0], front_top_left[1], front_bottom_left[0], front_bottom_left[1])
    img[rr,cc] = 1

    rr, cc = skimage.draw.line(front_top_right[0], front_top_right[1], front_bottom_right[0], front_bottom_right[1])
    img[rr,cc] = 1

    back_bottom_right = obliqueProjection([front_bottom_right[0], front_bottom_right[1], DEPTH])
    back_top_right = (back_bottom_right[0]-DEPTH, back_bottom_right[1])
    back_top_left = (back_top_right[0], back_top_right[1]-DEPTH)

    rr, cc = skimage.draw.line(front_bottom_right[0], front_bottom_right[1], back_bottom_right[0], back_bottom_right[1])
    img[rr,cc] = 1

    rr, cc = skimage.draw.line(back_bottom_right[0], back_bottom_right[1], back_top_right[0], back_top_right[1])
    img[rr,cc] = 1

    rr, cc = skimage.draw.line(back_top_right[0], back_top_right[1], back_top_left[0], back_top_left[1])
    img[rr,cc] = 1

    rr, cc = skimage.draw.line(back_top_left[0], back_top_left[1], front_top_left[0], front_top_left[1])
    img[rr,cc] = 1

    rr, cc = skimage.draw.line(back_top_right[0], back_top_right[1], front_top_right[0], front_top_right[1])
    img[rr,cc] = 1


    sparse = [Y, X, DEPTH]

    label = DEPTH ** 3


    return sparse, img, label, parameters




  @staticmethod
  def curvature(flags=[False,False,False], preset=None):



    var_y = flags[0]
    var_x = flags[1]
    var_width = flags[2]


    sparse = None
    img = None
    label = None
    parameters = 1


    DOF = 80
    DEPTH = np.random.randint(1, DOF+1)
    parameters *= DOF

    if preset:
      DEPTH = preset

    # print 'DEPTH', DEPTH

    Y_RANGE = (Figure1.DELTA_MAX-20, Figure1.DELTA_MAX)
    X_RANGE = (0, Figure1.DELTA_MIN*2)

    X = Figure1.DELTA_MIN
    if var_x:
      X, p = Util.parameter(X_RANGE[0], X_RANGE[1])
      parameters *= p

    Y = Figure1.DELTA_MAX
    if var_y:
      Y, p = Util.parameter(Y_RANGE[0], Y_RANGE[1])
      parameters *= p

    WIDTH = 60
    if var_width:
      WIDTH, p = Util.parameter(20, 60)
      # we only support even width
      WIDTH = int(math.ceil(WIDTH / 2.) * 2)
      parameters *= (p/2) # only half of the parameters 

    # print 'WIDTH', WIDTH

    start = (Y, X)
    mid = (DEPTH, X+WIDTH/2)
    end = (Y, X+WIDTH)

    img = np.zeros(Figure1.SIZE, dtype=np.bool)

    rr, cc = skimage.draw.bezier_curve(start[0], start[1], mid[0], mid[1], end[0], end[1], 1)
    img[rr, cc] = 1
    t = 0.5

    P10 = (mid[0] - start[0], mid[1] - start[1])
    P21 = (end[0] - mid[0], end[1] - mid[1])
    dBt_x = 2*(1-t)*P10[1] + 2*t*P21[1]
    dBt_y = 2*(1-t)*P10[0] + 2*t*P21[0]
    dBt2_x = 2*(end[1] - 2*mid[1] + start[1])
    dBt2_y = 2*(end[0] - 2*mid[0] + start[0])
    curvature = np.abs((dBt_x*dBt2_y - dBt_y*dBt2_x) / ((dBt_x**2 + dBt_y**2)**(3/2.)))

    sparse = [Y, X, DEPTH, WIDTH]

    label = np.round(curvature, 3)

    return sparse, img, label, parameters



  @staticmethod
  def shading(flags=[False, False], preset=None):


    var_y = flags[0]
    var_x = flags[1]


    sparse = None
    img = None
    label = None
    parameters = 1

    Y_RANGE = (0, 20)
    X_RANGE = (0, 20)

    DOF = 100
    COVERED = np.random.randint(1, DOF+1)
    parameters *= DOF


    if preset:
      COVERED = preset

    img = np.zeros((100,100), dtype=np.bool)

    X = 0
    if var_x:
      X, p = Util.parameter(X_RANGE[0], X_RANGE[1])
      parameters *= p

    Y = 0
    if var_y:
      Y, p = Util.parameter(Y_RANGE[0], Y_RANGE[1])
      parameters *= p

    step = max(1, 100-COVERED)

    img = np.zeros(Figure1.SIZE, dtype=np.bool)

    for i in range(0,100):
        for j in range(0,100):
        
            if (i+j+X) % step == 0: #or i == 100-step:
                img[i,j] = True
            if (i-j+Y) % step == 0: #or i == 100-step:
                img[i,j] = True

    sparse = [Y, X, COVERED]

    label = COVERED

    return sparse, img, label, parameters


