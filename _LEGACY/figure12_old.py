import cPickle as pickle
import numpy as np
import os
import time

class Figure12:

  @staticmethod
  def create_datapoint_(size, max_height, width, constrain=True, variability=False):
    '''
    '''
    delta = 5 # only if we constrain the differences between A and B
    # this enables cases which are difficult for the human eye
    
    if variability:
      A_x = np.random.randint(20-delta, 20+delta)
    A_x = 20
    
    if constrain:
      A_height = np.random.randint(max_height-delta, max_height)
      A_y = np.random.randint(delta, size[0] - max_height - delta)
    else:
      A_height = np.random.randint(1, max_height)
      A_y = np.random.randint(size[0] - max_height - 1)
    
    B_x = 60
    
    if constrain:
      B_height = np.random.randint(max_height-delta, max_height)
      B_y = np.random.randint(delta, size[0] - max_height - delta)
    else:
      B_height = np.random.randint(1, max_height)
      B_y = np.random.randint(size[0] - max_height - 1)
    
    return [A_x, A_y, A_height, B_x, B_y, B_height]

  @staticmethod
  def create_datapoint(size=(100,100), max_height=65, width=20, constrain=True, variability=False):
    '''
    '''
    A_x, A_y, A_height, B_x, B_y, B_height = Figure12.create_datapoint_(size, max_height, width, constrain)
    
    while (A_height == B_height) or (constrain and (np.abs(A_y - B_y) < 10)):
      # constrain means that the difference between the y position has to be larger than 10 pixel
      # this only enables that the human eye can't really perceive the difference
      # we always eliminate cases where the two bars are the same height since we can not label these as 0 or 1
      # (it is really not part of the experiment)
      A_x, A_y, A_height, B_x, B_y, B_height = Figure12.create_datapoint_(size, max_height, width, constrain)

    if A_height > B_height:
      label = 0
    elif B_height > A_height:
      label = 1
    
    return [A_x, A_y, A_height, B_x, B_y, B_height], label
    
  @staticmethod
  def create_image(datapoint, size=(100,100), max_height=65, width=20):
    '''
    Creates framed image.
    '''
    image = np.zeros(size, dtype=np.int8)
    A_x, A_y, A_height, B_x, B_y, B_height = datapoint
    
    image[A_y:A_y+max_height,A_x:A_x+1] = 2
    image[A_y:A_y+max_height,A_x+width:A_x+width+1] = 2
    image[A_y:A_y+1,A_x:A_x+width] = 2
    image[A_y+max_height:A_y+max_height+1,A_x:A_x+width+1] = 2
    image[A_y+max_height-A_height:A_y+max_height+1,A_x:A_x+width+1] = 1
    
    image[B_y:B_y+max_height,B_x:B_x+1] = 2
    image[B_y:B_y+max_height,B_x+width:B_x+width+1] = 2
    image[B_y:B_y+1,B_x:B_x+width] = 2
    image[B_y+max_height:B_y+max_height+1,B_x:B_x+width+1] = 2    
    image[B_y+max_height-B_height:B_y+max_height+1,B_x:B_x+width+1] = 1
    
    return image

  @staticmethod
  def create_images_from_datapoints(datapoints, labels):
    '''
    '''
    N = len(labels)
    
    images = np.zeros((N, 100, 100), dtype=np.bool)
    framed_images = np.zeros((N, 100, 100), dtype=np.bool)
    labels = np.array(labels, dtype=np.bool)

    for i,datapoint in enumerate(datapoints):
      framed_image = Figure12.create_image(datapoint)

      # remove frame
      image = framed_image.copy()
      image[image == 2] = 0
      images[i] = image.astype(np.bool)

      # harden frame
      framed_image[framed_image == 2] = 1
      framed_images[i] = framed_image.astype(np.bool)

    return images, framed_images, labels

  @staticmethod
  def setup(N=100000, directory=None):
    '''
    '''
    datapoints = []
    labels = []

    if not directory or (directory and not os.path.exists(directory)):
      # we didn't save yet or do not want to save
        
      for n in range(N):
        datapoint, label = Figure12.create_datapoint()
        datapoints.append(datapoint)
        labels.append(label)
        
      if directory:
        os.makedirs(directory)
        
        # now directly store this badboy
        with open(os.path.join(directory, 'datapoints.p'), 'w') as f:
          pickle.dump(datapoints, f)

        with open(os.path.join(directory, 'labels.p'), 'w') as f:
          pickle.dump(labels, f)
        
        print 'Stored', directory
    
    else:
      # we just want to load
      print 'Loading from', directory
      
      with open(os.path.join(directory, 'datapoints.p'), 'r') as f:
        datapoints = pickle.load(f)

      with open(os.path.join(directory, 'labels.p'), 'r') as f:
        labels = pickle.load(f)
    
    images, framed_images, labels = Figure12.create_images_from_datapoints(datapoints, labels)
    
    return images, framed_images, labels
