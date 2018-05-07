from mpi4py import MPI
from datetime import datetime
import time
import numpy as np
import sys


S = int(sys.argv[1])
tau_max = int(sys.argv[2])
print('S is %f' %S)
print('tau_max is %f' %tau_max)

