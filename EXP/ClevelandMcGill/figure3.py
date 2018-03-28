import numpy as np
import os
import skimage.draw
import sys

sys.path.append('../')
from util import Util

class Figure3:

  SIZE = (100, 100)


  @staticmethod
  def generate_datapoint():
    '''
    '''


    # from codementum.org/cleveland-mcgill/
    #
    #
    # randomize data according to Cleveland84
    # specifically:
    #  - 5 numbers
    #  - add to 100
    #  - none less than 3
    #  - none greater than 39
    #  - differences greater than .1
    #
    def randomize_data():
      max = 36;
      min = 3;
      diff = 0.1;

      d = []

      while len(d) < 5:
        randomnumber=np.ceil(np.random.random()*36 + 3);
        found=False;
        for i in range(len(d)):
          if not ensure_difference(d, randomnumber):
            found = True
            break
            
        if not found:
          d.append(randomnumber)


      return d;


    def ensure_difference(A, c):
      result = True;
      for i in range(len(A)):
        if c > (A[i] - 3) and c < (A[i] + 3):
          result = False

      return result

    sum = -1
    while(sum != 100):
      data = randomize_data()
      sum = data[0] + data[1] + data[2] + data[3] + data[4]

    labels = np.zeros((5), dtype=np.float32)
    for i,d in enumerate(data):
      labels[i] = d/float(np.max(data))

    #
    #
    # ATTENTION, HERE WE NEED TO ORDER THE LABELS ACCORDING
    # OUR CONVENTION
    #
    # NOW, THE MARKED ELEMENT IS AT POSITION 0 BUT ONLY IN THE
    # LABELS. THIS MEANS WE CAN NOW GO LEFT TO RIGHT (ROLLING) IN THE BARCHART
    # STARTING FROM THE MARKED ONE. AND SIMILARLY, IN THE PIE CHART WE CAN GO
    # COUNTER-CLOCKWISE STARTING FROM THE MARKED ONE.
    #
    labels = np.roll(labels, 5-np.where(labels==1)[0])

    return data, list(labels)


  @staticmethod
  def data_to_barchart(data):
    '''
    '''
    barchart = np.zeros((100,100), dtype=np.bool)

    for i,d in enumerate(data):

      if i==0:
        start = 2
      else:
        start = 0
        
      left_bar = start+3+i*3+i*16
      right_bar = 3+i*3+i*16+16
      
      rr, cc = skimage.draw.line(99, left_bar, 99-int(d), left_bar)
      barchart[rr, cc] = 1
      rr, cc = skimage.draw.line(99, right_bar, 99-int(d), right_bar)
      barchart[rr, cc] = 1
      rr,cc = skimage.draw.line(99-int(d), left_bar, 99-int(d), right_bar)
      barchart[rr, cc] = 1
      
      if d == np.max(data):
        # mark the max
        barchart[90:91, left_bar+8:left_bar+9] = 1

    return barchart

  @staticmethod
  def data_to_piechart(data):
    '''
    '''
    piechart = np.zeros((100,100), dtype=np.bool)
    RADIUS = 30
    rr,cc = skimage.draw.circle_perimeter(50,50,RADIUS)
    piechart[rr,cc] = 1
    random_direction = np.random.randint(360)
    theta = -(np.pi / 180.0) * random_direction
    END = (50 - RADIUS * np.cos(theta), 50 - RADIUS * np.sin(theta))
    rr, cc = skimage.draw.line(50, 50, int(np.round(END[0])), int(np.round(END[1])))
    piechart[rr, cc] = 1

    for i,d in enumerate(data):

      current_value = data[i]
      current_angle = (current_value / 100.) * 360.
      # print current_value, current_angle
      # print 'from', random_direction, 'to', current_angle
      theta = -(np.pi / 180.0) * (random_direction-current_angle)
      END = (50 - RADIUS * np.cos(theta), 50 - RADIUS * np.sin(theta))
      rr, cc = skimage.draw.line(50, 50, int(np.round(END[0])), int(np.round(END[1])))
      piechart[rr,cc] = 1
      
      if d == np.max(data):
        # this is the max spot
        theta = -(np.pi / 180.0) * (random_direction-current_angle/2.)
        END = (50 - RADIUS/2 * np.cos(theta), 50 - RADIUS/2 * np.sin(theta))
        rr, cc = skimage.draw.line(int(np.round(END[0])), int(np.round(END[1])), int(np.round(END[0])), int(np.round(END[1])))
        piechart[rr,cc] = 1
      
      random_direction -= current_angle

    return piechart

  @staticmethod
  def data_to_piechart_aa(data):
    '''
    '''
    piechart = np.zeros((100,100), dtype=np.float32)
    RADIUS = 30
    rr,cc,val = skimage.draw.circle_perimeter_aa(50,50,RADIUS)
    piechart[rr,cc] = val
    random_direction = np.random.randint(360)
    theta = -(np.pi / 180.0) * random_direction
    END = (50 - RADIUS * np.cos(theta), 50 - RADIUS * np.sin(theta))
    rr, cc, val = skimage.draw.line_aa(50, 50, int(np.round(END[0])), int(np.round(END[1])))
    piechart[rr, cc] = val

    for i,d in enumerate(data):

      current_value = data[i]
      current_angle = (current_value / 100.) * 360.
      # print current_value, current_angle
      # print 'from', random_direction, 'to', current_angle
      theta = -(np.pi / 180.0) * (random_direction-current_angle)
      END = (50 - RADIUS * np.cos(theta), 50 - RADIUS * np.sin(theta))
      rr, cc,val = skimage.draw.line_aa(50, 50, int(np.round(END[0])), int(np.round(END[1])))
      piechart[rr,cc] = val
      
      if d == np.max(data):
        # this is the max spot
        theta = -(np.pi / 180.0) * (random_direction-current_angle/2.)
        END = (50 - RADIUS/2 * np.cos(theta), 50 - RADIUS/2 * np.sin(theta))
        rr, cc,val = skimage.draw.line_aa(int(np.round(END[0])), int(np.round(END[1])), int(np.round(END[0])), int(np.round(END[1])))
        piechart[rr,cc] = val
      
      random_direction -= current_angle

    return piechart

