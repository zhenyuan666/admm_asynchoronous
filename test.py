from mpi4py import MPI
from datetime import datetime
import time
import numpy as np



comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'rank': 0}
    time.sleep(2)
    req = comm.isend(data, dest=1, tag=11)
    print("rank 0 %s" %str(datetime.now()))

elif rank == 1:
    num_probed = 0
    is_received = np.zeros((3, 1))
    while num_probed < 2:
        is_received[0, 0] = comm.Iprobe(source=0, tag=11)
        is_received[1, 0] = comm.Iprobe(source=2, tag=11)
        is_received[2, 0] = comm.Iprobe(source=3, tag=11)
        num_probed =  sum(is_received)
    print(num_probed)
    print(is_received)
    if is_received[0, 0] == 1:
    	req0 = comm.irecv(source=0, tag=11)
    	data = req0.wait()
    	print(data)
    if is_received[1, 0] == 1:
    	req2 = comm.irecv(source=2, tag=11)
    	data = req2.wait()
    	print(data)
    if is_received[2, 0] == 1:
    	req3 = comm.irecv(source=3, tag=11)
    	data = req3.wait()
    	print(data)

    print("rank 1 %s" %str(datetime.now()))

elif rank == 2:
    data = {'rank': 2}
    time.sleep(10)
    req = comm.isend(data, dest=1, tag=11)
    print("rank 2 %s" %str(datetime.now()))

elif rank == 3:
    data = {'rank': 3}
    time.sleep(4)
    req = comm.isend(data, dest=1, tag=11)
    print("rank 3 %s" %str(datetime.now()))




