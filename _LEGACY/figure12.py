import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('../')
from util import Util

class Figure12:

  def __init__(self, variable_y=False, \
                     variable_width=False, \
                     variable_x=False, \
                     constrain=True):
    '''
  	'''

    # the flag for the visual aid
    self.framed = False

    self.A_x = None
    self.A_y = None
    self.A_value = None
    self.A_width = None

    self.B_x = None
    self.B_y = None
    self.B_value = None
    self.B_width = None

    self.label = None

    # measured from Cleveland McGill paper via pixel ruler
    self._size = (100,100)
    self._max_height = 65
    self._default_width = 20

    self._delta = 5 # only if we constrain the differences between A and B
    # this enables cases which are difficult for the human eye

    #
    # variability switches
    #
    #  stage 1: var. Y
    #  stage 2: var. WIDTH
    #  stage 3: var. X
    #
    self._variable_y = variable_y
    self._variable_width = variable_width
    self._variable_x = variable_x
    self._constrain = constrain


  def create(self, verbose=False):
    '''
    Create a valid datapoint which fulfills constrains
    and the variability settings.
    '''
    self.create_(verbose)

    while (self.A_value == self.B_value): #or (self._variable_y and self._constrain and (np.abs(self.A_y - self.B_y) < 10)):
      # constrain means that the difference between the y position has to be larger than 10 pixel
      # this only enables that the human eye can't really perceive the difference
      # we always eliminate cases where the two bars are of the same value since we can not label these as 0 or 1
      # (it is really not part of the experiment)
      self.create_(verbose)

    # update the label
    if self.A_value > self.B_value:
      self.label = 0
    elif self.B_value > self.A_value:
      self.label = 1


  def parametrize_rectangle(self, X, Y, WIDTH):
    '''
    '''

    parameters = 1

    #
    # X can be variable
    #
    if self._variable_x:
      X, p = Util.parameter(X, X, delta=self._delta)
      parameters *= p

    #
    # Y can be variable
    #
    if self._variable_y:
      #
      # Y starts from the top
      #
      # we constrain values to 5..-5 to not have the image border be touched
      Y, p = Util.parameter(self._delta, self._size[0] - self._max_height - self._delta)
      parameters *= p

    #
    # We constrain the value to 60..65
    #
    VALUE, p = Util.parameter(self._max_height - self._delta, self._max_height)
    # but we also have the full range flag which has the range 1..65
    if not self._constrain:
      VALUE, p = Util.parameter(1, self._max_height)
    parameters *= p

    #
    # Width can be variable
    #
    if self._variable_width:
      WIDTH, p = Util.parameter(WIDTH, WIDTH, delta=self._delta)
      parameters *= p


    return [X, Y, WIDTH, VALUE], parameters


  def create_(self, verbose):
    '''
    '''

    # parametrize rectangle A
    values_A, parameters_A = self.parametrize_rectangle(20, 20, 20)
    self.A_x, self.A_y, self.A_width, self.A_value = values_A

    # parametrize rectangle B
    values_B, parameters_B = self.parametrize_rectangle(60, 20, 20)
    self.B_x, self.B_y, self.B_width, self.B_value = values_B

    if verbose:
      print '# Parameters', parameters_A*parameters_B



  def to_sparse(self):
    '''
    Convert to sparse representation
      [A_x, A_y, A_width, A_value, B_x, B_y, B_width, B_value], label
    '''
    return [self.A_x, \
            self.A_y, \
            self.A_width, \
            self.A_value, \
            self.B_x, \
            self.B_y, \
            self.B_width, \
            self.B_value], self.label


  @staticmethod
  def from_sparse(sparse_representation,
                  variable_y=False, \
                  variable_width=False, \
                  variable_x=False, \
                  constrain=True):
    '''
    '''
    fig = Figure12(variable_y=variable_y, \
                   variable_width=variable_width, \
                   variable_x=variable_x, \
                   constrain=constrain)

    fig.A_x = sparse_representation[0]
    fig.A_y = sparse_representation[1]
    fig.A_width = sparse_representation[2]
    fig.A_value = sparse_representation[3]

    fig.B_x = sparse_representation[4]
    fig.B_y = sparse_representation[5]
    fig.B_width = sparse_representation[6]
    fig.B_value = sparse_representation[7]

    # update the label
    if fig.A_value > fig.B_value:
      fig.label = 0
    elif fig.B_value > fig.A_value:
      fig.label = 1

    return fig


  def to_image(self):
    '''
    '''
    image = np.zeros(self._size, dtype=np.int8)
    
    image[self.A_y:self.A_y+self._max_height,self.A_x:self.A_x+1] = 2
    image[self.A_y:self.A_y+self._max_height,self.A_x+self.A_width:self.A_x+self.A_width+1] = 2
    image[self.A_y:self.A_y+1,self.A_x:self.A_x+self.A_width] = 2
    image[self.A_y+self._max_height:self.A_y+self._max_height+1,self.A_x:self.A_x+self.A_width+1] = 2
    image[self.A_y+self._max_height-self.A_value+1:self.A_y+self._max_height+1,self.A_x:self.A_x+self.A_width+1] = 1
    
    image[self.B_y:self.B_y+self._max_height,self.B_x:self.B_x+1] = 2
    image[self.B_y:self.B_y+self._max_height,self.B_x+self.B_width:self.B_x+self.B_width+1] = 2
    image[self.B_y:self.B_y+1,self.B_x:self.B_x+self.B_width] = 2
    image[self.B_y+self._max_height:self.B_y+self._max_height+1,self.B_x:self.B_x+self.B_width+1] = 2    
    image[self.B_y+self._max_height-self.B_value+1:self.B_y+self._max_height+1,self.B_x:self.B_x+self.B_width+1] = 1
    
    if self.framed:
      image[image == 2] = 1
    else:
      image[image == 2] = 0

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
    labels = np.zeros((N), dtype=np.bool)

    for n in range(N):

      current = Figure12.from_sparse(datapoints[n], \
                                     variable_y=self._variable_y, \
                                     variable_width=self._variable_width, \
                                     variable_x=self._variable_x, \
                                     constrain=self._constrain)

      current.framed = framed

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

      current = Figure12.from_sparse(datapoints[n], \
                                     variable_y=self._variable_y, \
                                     variable_width=self._variable_width, \
                                     variable_x=self._variable_x, \
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

      current = Figure12(variable_y=self._variable_y, \
                         variable_width=self._variable_width, \
                         variable_x=self._variable_x, \
                         constrain=self._constrain)

      current.create()

      datapoint, label = current.to_sparse()

      datapoints.append(datapoint)
      labels.append(label)

    return datapoints, labels


  @staticmethod
  def load(DATA_DIR='../DATA/Figure12/', dataset=1, preview=True, images=True):
    '''
    '''
    with open(os.path.join(DATA_DIR, 'datapoints_'+str(dataset)+'.p'), 'r') as f:
      datapoints = pickle.load(f)
      
    with open(os.path.join(DATA_DIR, 'labels_'+str(dataset)+'.p'), 'r') as f:
      labels = pickle.load(f)  
      
    if not images:
      # return just the datapoints
      return datapoints, labels

    fig = Figure12()
    fig.create()
    images, labels = fig.render_many(datapoints, framed=False)
    framed_images, framed_labels = fig.render_many(datapoints, framed=True)    

    assert np.array_equal(labels, framed_labels) is True

    if preview:
      INDEX = np.random.randint(len(datapoints))
      print 'Datapoint', INDEX
      print 'Label', labels[INDEX]

      plt.imshow(images[INDEX])
      plt.figure()
      plt.imshow(framed_images[INDEX])

    return images, framed_images, labels
