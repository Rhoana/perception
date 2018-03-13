import matplotlib.pyplot as plt
import numpy as np

class Util:

  @staticmethod
  def parameter(start, end, delta=0):

    value = np.random.randint(start-delta, end+delta)
    parameters = len(range(start-delta, end+delta))

    return value, parameters

  @staticmethod
  def imshow_nicely(image, filename=None, new_figure=True):

    if new_figure:
      plt.figure()
      
    plt.imshow(image, cmap='Greys', interpolation='nearest')
    ax = plt.gca()
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_xticks(np.arange(-.5, 100, 10), minor=False);
    ax.set_yticks(np.arange(-.5, 100, 10), minor=False);
    ax.grid(which='major', color='gray', linestyle=':', linewidth='0.5')
    ax.set_axisbelow(True)

    if filename:

      plt.savefig(filename, bbox_inches='tight', transparent=True,)
    