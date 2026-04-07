import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import dft

def Decimator(signal, max=.3, min=None):
  if min is None:
    min = - max
  valid = np.ones_like(signal)
  valid[signal > max] = 0
  valid[signal < min] = 0
  return valid

def Reconstruct(decimator, g, pulse):
    Omega = fftfreq(pulse.shape[0])
    D = np.diag(decimator)
    F = dft(len(g))
    F_ = F[:, np.abs(Omega) < .1]

    b = D @ g
    A = D @ F_
    x = np.linalg.lstsq(A, b, rcond=-1)

    d = F_ @ x[0]
    d = np.real(d)

    return d

from scipy.signal import hilbert, convolve2d