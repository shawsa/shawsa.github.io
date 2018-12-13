#from ctypes import *
import ctypes
import numpy as np
import numpy.linalg as la
from numpy.ctypeslib import ndpointer

libtest = ctypes.cdll.LoadLibrary("libtest.so")
x, y = 3, 5
print("%d + %d = %d" % (x, y, libtest.my_sum(x,y)) )

n = 5
x = np.random.randn(n)
y = np.random.randn(n)

w = x+y

test = np.empty(n)
libtest.py_my_sum(x.ctypes.data, y.ctypes.data, n, test.ctypes.data)

