import numpy

def mean_absolute_percentage_error(labels, predictions):
  error = numpy.abs((labels - predictions) / labels)
  return numpy.mean(error) * 100