import numpy as np

class Util:

  @staticmethod
  def parameter(start, end, delta=0):

    value = np.random.randint(start-delta, end+delta)
    parameters = len(range(start-delta, end+delta))

    return value, parameters
