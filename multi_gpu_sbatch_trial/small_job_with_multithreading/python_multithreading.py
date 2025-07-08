#!/usr/bin/env python

# NumPy example with implicit threading

import numpy as np

n = 4096
M = np.random.rand(n, n)

# The following computation will use threading via the underlying
# high-performance BLAS library (Intel MKL), based on the variable
# OMP_NUM_THREADS
N = np.matmul(M, M)

print(N.size)
