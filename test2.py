from mpi4py import MPI
from datetime import datetime
import time
import numpy as np



comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'1st time send from rank': 0}
    time.sleep(2)
    req = comm.isend(data, dest=1, tag=11)
    print("rank 0 1st time send time is: %s" %str(datetime.now()))
    #
    data = {'2nd time send from rank': 0}
    time.sleep(2)
    req = comm.isend(data, dest=1, tag=11)
    print("rank 0 2nd time send time is: %s" %str(datetime.now()))
    #
    data = {'3rd time send from rank': 0}
    time.sleep(2)
    req = comm.isend(data, dest=1, tag=11)
    print("rank 0 3rd time send time is: %s" %str(datetime.now()))
    #
    data = {'4th time send from rank': 0}
    time.sleep(2)
    req = comm.isend(data, dest=1, tag=11)
    print("rank 0 4th time send time is: %s" %str(datetime.now()))
    #
    data = {'5th time send from rank': 0}
    time.sleep(2)
    req = comm.isend(data, dest=1, tag=11)
    print("rank 0 5th time send time is: %s" %str(datetime.now()))


elif rank == 1:
    time.sleep(15)
    req0 = comm.irecv(source=0, tag=11)
    data = req0.wait()
    print(data)
    print("rank 1 %s" %str(datetime.now()))
    #
    time.sleep(2)
    req0 = comm.irecv(source=0, tag=11)
    data = req0.wait()
    print(data)
    print("rank 1 %s" %str(datetime.now()))
    #
    time.sleep(2)
    req0 = comm.irecv(source=0, tag=11)
    data = req0.wait()
    print(data)
    print("rank 1 %s" %str(datetime.now()))
    #
    time.sleep(2)
    req0 = comm.irecv(source=0, tag=11)
    data = req0.wait()
    print(data)
    print("rank 1 %s" %str(datetime.now()))
    #
    time.sleep(2)
    req0 = comm.irecv(source=0, tag=11)
    data = req0.wait()
    print(data)
    print("rank 1 %s" %str(datetime.now()))

elif rank == 2:
    print("rank 2 %s" %str(datetime.now()))

elif rank == 3:
    print("rank 3 %s" %str(datetime.now()))




