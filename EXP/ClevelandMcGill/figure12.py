import numpy as np
import os
import skimage.draw
import sys

sys.path.append('../')
from util import Util

class Figure12:

  SIZE = (100, 100)

  @staticmethod
  def generate_datapoint():
    '''
    '''
    # ok james, what do you think of this:
    # we stick to 100x100 images
    # then we leave a delta of 20 to move the bars in Y
    # the bars  then have a max value of 60
    # this projects them to the range of 10..70 or 30..90 in the extreme cases which leaves the 
    # 10 pixel border around which makes sure our 5x5 lenet picks up all pixels

    # then, of the 60 bar value we say max 20% == 12 pixels white and max 80% black == 48 pixels.

    # then the least JND is for white: 1/12 == 8% 
    # and for black 1/48 == 20% and 1-p for the best case

    # do you think thats enough?

    # james: this is just bloody fantastic
    # (not sure if he really said anything like that but i imagine he did LOL)

    parameters = 1

    # VALUE A
    value_A, p = Util.parameter(49, 61)
    parameters *= p

    # VALUE B
    value_B = value_A
    while value_B == value_A:
      value_B, p = Util.parameter(49, 61)
    parameters *= p

    # POS A
    x_A = 20
    y_A, p = Util.parameter(10, 30)
    parameters *= p

    # POS B
    x_B = 60
    y_B = y_A
    while y_B == y_A:
      y_B, p = Util.parameter(10, 30)
    parameters *= p

    data = [x_A, y_A, value_A, x_B, y_B, value_B]

    labels = [value_A, value_B]

    return data, labels, parameters

  @staticmethod 
  def data_to_bars(data):
    '''
    '''
    return Figure12.data_to_image(data, framed=False)

  @staticmethod 
  def data_to_framed_rectangles(data):
    '''
    '''
    return Figure12.data_to_image(data, framed=True)


  @staticmethod
  def data_to_image(data, framed=True):
    '''
    '''
    image = np.zeros((100,100), dtype=np.uint8)
    
    x_A, y_A, value_A, x_B, y_B, value_B = data

    barwidth = 20
    max_height = 60


    image[y_A:y_A+max_height,x_A:x_A+1] = 2
    image[y_A:y_A+max_height,x_A+barwidth:x_A+barwidth+1] = 2
    image[y_A:y_A+1,x_A:x_A+barwidth] = 2
    image[y_A+max_height:y_A+max_height+1,x_A:x_A+barwidth+1] = 2
    image[y_A+max_height-value_A+1:y_A+max_height+1,x_A:x_A+barwidth+1] = 1
    
    image[y_B:y_B+max_height,x_B:x_B+1] = 2
    image[y_B:y_B+max_height,x_B+barwidth:x_B+barwidth+1] = 2
    image[y_B:y_B+1,x_B:x_B+barwidth] = 2
    image[y_B+max_height:y_B+max_height+1,x_B:x_B+barwidth+1] = 2    
    image[y_B+max_height-value_B+1:y_B+max_height+1,x_B:x_B+barwidth+1] = 1
    
    if framed:
      image[image == 2] = 1
    else:
      image[image == 2] = 0

    return image




