import numpy as np
import os
import skimage.draw
import sys

sys.path.append('../')
from util import Util

class Weber:

  SIZE = (100, 100)


  @staticmethod
  def base10():
    '''
    '''
    randomnumber = np.random.randint(1,11)

    image = Weber.generate(base=10, to_add=randomnumber)

    return image, randomnumber


  @staticmethod
  def base100():
    '''
    '''
    randomnumber = np.random.randint(1,11)

    image = Weber.generate(base=100, to_add=randomnumber)

    return image, randomnumber


  @staticmethod
  def base1000():
    '''
    '''
    randomnumber = np.random.randint(1,11)

    image = Weber.generate(base=1000, to_add=randomnumber)

    return image, randomnumber


  @staticmethod
  def generate(base=10, to_add=10):

    image = np.zeros((100,100), dtype=np.bool)

    for p in range(base):
      
      image[np.random.randint(100), np.random.randint(100)] = 1


    added = 0
    while added < to_add:
      
      y, x = np.random.randint(100), np.random.randint(100)
      
      if image[y,x] == 1:
        continue

      image[y, x] = 1
      added += 1

    return image


