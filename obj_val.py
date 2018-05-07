from __future__ import division
import pdb,time,os
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky
from mpi4py import MPI
import scipy.io as sio
import sys

"""

"""

S = int(sys.argv[1])
tau_max = int(sys.argv[2])
machine_number = int(sys.argv[3])
num_iters = int(sys.argv[4])
m = int(sys.argv[5])
n = int(sys.argv[6])


def main():
    if machine_number == 0:
        main_folder = '/global/homes/z/zhenyuan/project/async/'
    else:
        main_folder = '/Users/zhenyuanliu/Dropbox/Spring2018/CS267/project/asychronous/data4/'

    # main_folder = '/Users/zhenyuanliu/Dropbox/Spring2018/CS267/project/asychronous/data4/'
    for i in range(5):
        success = obj_val(directory = main_folder, repeatition = i, max_iter = num_iters)
    # print('objective value is : ' + str(total_loss[0]))
    # print('norm of x_hat is: ' + str(norm(z)))


def obj_val(directory, repeatition, mylambda = .5, rho = 1., max_iter = 200, abs_tol = 1e-6, rel_tol = 1e-4):
    '''
    MPI
    '''
    read_start = time.time()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    N = size

    '''
    Read in Data
    '''
    print('generating A' + str(rank) + '.npy')
    gen_start = time.time()
    # x_true is the same for each rank
    np.random.seed(0)
    x_true = np.random.normal(0, 1, (n, 1))
    index_zero = np.random.choice(n, int(n * 0.9))
    x_true[index_zero, :] = 0
    #
    np.random.seed(rank)
    A = np.random.normal(0, 1, (m, n))
    v = np.random.normal(0, 1e-3, (m, 1));
    b = np.dot(A, x_true) + v;
    #
    # m,n = A.shape
    filename_z_hist = directory + 'z_hist_' + str(S) + '_' + str(tau_max)+ '_' + str(repeatition) + '.npy'
    z_hist = np.load(filename_z_hist)
    comm.Barrier()
    gen_end = time.time()


    if rank == 0:
        print('generating data time is %f\n' %(gen_end - gen_start))
        tic = time.time()

    #save a matrix-vector multiply
    Atb = A.T.dot(b)
    
    #initialize ADMM solver
    x = np.zeros((n,1))
    z = np.zeros((n,1))
    u = np.zeros((n,1))
    r = np.zeros((n,1))

    local_loss = np.zeros(1)
    total_loss = np.zeros(1)

    objval = []

    '''
    ADMM solver loop
    '''
    for k in range(max_iter):
        
        # diagnostics, reporting, termination checks
        z = z_hist[:, k]
        z = z.reshape((z.shape[0], 1))
        local_loss[0] = objective(A, b, mylambda * 1./N, z)
        comm.Barrier()
        comm.Allreduce([local_loss, MPI.DOUBLE], [total_loss, MPI.DOUBLE], op=MPI.SUM)
        #
        objval.append(total_loss[0])

    if rank == 0:
        toc = time.time() - tic
        print("\nElapsed time is %.2f seconds"%toc)
        '''
        Store results
        '''
        np.save(directory + 'objval_' + str(S) + '_' + str(tau_max)+ '_' + str(repeatition) + '.npy', objval)

    return 1

def objective(A, b, mylambda, z):
    return .5 * norm(A.dot(z) - b)**2 + mylambda * norm(z, 1)

if __name__=='__main__':
    main()
