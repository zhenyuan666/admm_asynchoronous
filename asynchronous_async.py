from __future__ import division
import pdb,time,os
import numpy as np
from numpy.linalg import norm,cholesky
from mpi4py import MPI
import scipy.io as sio
import sys

"""

"""

S_init = int(sys.argv[1])
tau_max = int(sys.argv[2])
machine_number = int(sys.argv[3])
num_iters = int(sys.argv[4])
m = int(sys.argv[5])
n = int(sys.argv[6])
S_cut = int(sys.argv[7])


def main():
    if machine_number == 0:
        main_folder = '/global/homes/z/zhenyuan/project/async/'
    else:
        main_folder = '/Users/zhenyuanliu/Dropbox/Spring2018/CS267/project/asychronous/data4/'
    success = lasso_admm(directory = main_folder, num_repeatition = 5, max_iter = num_iters)
    # print('objective value is : ' + str(total_loss[0]))
    # print('norm of x_hat is: ' + str(norm(z)))


def lasso_admm(directory, num_repeatition, mylambda = .5, rho = 1., max_iter = 200, abs_tol = 1e-6, rel_tol = 1e-4):
    """
     Lasso problem:

       minimize 1/2*|| A x - b ||_2^2 + mylambda || x ||_1
    Output:
            - z      : solution of the Lasso problem
            - objval : objective value
            - r_norm : primal residual norm
            - s_norm : dual residual norm
            - eps_pri: tolerance for primal residual norm
            - eps_pri: tolerance for dual residual norm
    """

    '''
    MPI
    '''
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    for repeatition in range(num_repeatition):
        N = size
        comm_time = []
        total_time = []
        inv_time = []
        S = S_init

        # a broadcasted variable by rank 0, True if the master node finishes the for loop
        finished = 0.

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
        comm.Barrier()
        gen_end = time.time()
        # print('rank is ' + str(rank) + 'x_true: ' + str(norm(x_true)))
        
        if rank == 0:
            print('data generation time is %f\n' %(gen_end - gen_start))
            tic = time.time()

            #save a matrix-vector multiply
            Atb = A.T.dot(b)
            
            #initialize ADMM solver
            x = np.zeros((n,1))
            z = np.zeros((n,1))
            u = np.zeros((n,1))
            r = np.zeros((n,1))
            z_hist = np.zeros((n, max_iter))

            send = np.zeros(3)
            recv = np.zeros(3)    
            local_loss = np.zeros(1)
            local_losses = np.zeros((N, 1))
            total_losses = np.zeros((max_iter, 1))
            num_update = np.zeros((N, max_iter))

            # cache the (Cholesky) factorization
            factor_start = time.time()
            L,U = factor(A, rho)
            factor_end = time.time()
            print('factor time is %f\n' %(factor_end - factor_start))

            print('\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s' %('iter',
                                                              'r norm', 
                                                              'eps pri', 
                                                              's norm', 
                                                              'eps dual', 
                                                              'objective'))
            objval     = []
            r_norm     = []
            s_norm     = []
            eps_pri    = []
            eps_dual   = []

            # tau_otherrank : keep track of how many time steps a given rank falls behind rank 0
            tau_otherrank = np.ones((N, 1))

            '''
            ADMM solver loop
            '''
            for k in range(max_iter):
                if k > S_cut:
                    S = min(S+3, N)
                    
                # last step
                if k == max_iter - 1:
                    S = N

                iter_start = time.time()

                # u-update
                u= u + x - z
                 
                # local x-update 
                q = Atb + rho * (z - u) #(temporary value)
                
                inv_time_start = time.time()
                if m >= n: # skinny matrix, use inverse of (A^T A + \rho I)
                    x = np.linalg.solve(U,np.linalg.solve(L, q))
                else:
                    ULAq = np.linalg.solve(U, np.linalg.solve(L, A.dot(q)))
                    x = (q * 1./rho)-((A.T.dot(ULAq))*1./(rho**2))
                inv_time_end = time.time()

                # x = linalg.solve(U, linalg.solve(L, q))

                # send[0] = r.T.dot(r)[0][0] # sum ||r_i||_2^2
                # send[1] = x.T.dot(x)[0][0] # sum ||x_i||_2^2
                # send[2] = u.T.dot(u)[0][0]/(rho**2) # sum ||y_i||_2^2

                # zprev = np.copy(z)

                # calculate the local loss
                local_loss[0] = objective(A, b, mylambda * 1. /N, z)
                local_losses[0, 0] = local_loss[0]
                num_update[0, k] = 1
                
                # wait until receive a minimum of S updates , and max(tau_1, tau_i...) <= tau_size
                num_probed = 1
                is_received = np.zeros((N, 1))
                is_received[0, 0] = 1 # rank 0 is always "received" because it's always accesssible

                # if tau_otherrank == 10, then force communication with that rank
                for ii in range(1, N):
                    if is_received[ii, 0] == 0:
                        if  tau_otherrank[ii, 0] == tau_max:
                            is_received[ii, 0] = 1
                            tau_otherrank[ii, 0] = 1
                num_probed =  sum(is_received)
                
                recv_start = time.time()

                # wait until at least S of the N x+u updates are received by the master node
                while num_probed < S:
                    for ii in range(1, N):
                        is_received[ii, 0] = comm.Iprobe(source = ii, tag=13)
                        num_probed =  sum(is_received)
                
                # print("iteration %f\t rank %f" %(k, rank))
                # print(is_received)
                x_add_u_received = np.zeros((n, 1))

                # finally retrieve the x+u from the num_probed ranks and take the average of them to update z
                to_receive = np.zeros((n+1, 1))

                for ii in range(1, N):
                    if is_received[ii, 0] == 1:
                        # reset tau_otherrank to 1
                        tau_otherrank[ii, 0] = 1
                        num_update[ii, k] = 1
                        req = comm.irecv(1000000, source = ii, tag = 13)
                        # 
                        to_receive = req.wait()
                        # if ii == 1:
                        #     print(type(to_receive))
                        temp1 = to_receive[0:-1, 0]
                        temp2 = to_receive[-1, 0]
                        local_losses[ii, 0] = temp2
                        x_add_u_received += temp1.reshape((temp1.shape[0], 1))
                    else:
                        tau_otherrank[ii, 0] += 1

                recv_end_1 = time.time()

                # don't forget the x+u in the rank 0
                x_add_u_received += x + u
                 
                # z-update
                z = soft_threshold(x_add_u_received * 1./num_probed, mylambda * 1./(N * rho))
                z_hist[:, k] = z.reshape((z.shape[0], ))
                print('z:' + str(norm(z)))
             
                # send z etc to the above num_probed nodes that contributed to the update of z
                to_send = np.zeros((n+1, 1))
                to_send[0:-1, 0] = z.reshape((z.shape[0], ))
                finished = k == max_iter - 1
                to_send[-1, :] = finished
                
                recv_end_2 = time.time()
                if k < max_iter -1:
                    for ii in range(1, N):
                        if is_received[ii, 0] == 1:
                            req = comm.isend(to_send, dest = ii, tag=13)
                            req.wait()

                # the final iteration, send/receive message to all ranks to stop
                else:
                    for ii in range(1, N):
                        req = comm.isend(to_send, dest = ii, tag=13)
                        req.wait()

                recv_end_3 = time.time()

                # # diagnostics, reporting, termination checks
                # # prior residual -> norm(x-z)
                # r_norm.append(np.sqrt(recv[0])/np.sqrt(N))
                # # dual residual -> norm(-rho*(z-zold))
                # s_norm.append(rho * norm(z - zprev))
                # eps_pri.append(np.sqrt(n) * abs_tol + 
                #                rel_tol * np.maximum(np.sqrt(recv[1])/np.sqrt(N), norm(z)))
                # eps_dual.append(np.sqrt(n) * abs_tol + rel_tol * np.sqrt(recv[2])/np.sqrt(N))
                # print('%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' %(k,\
                #                                                   r_norm[k],\
                #                                                   eps_pri[k],\
                #                                                   s_norm[k],\
                #                                                   eps_dual[k],\
                #                                                   objval[k]))

                # if r_norm[k] < eps_pri[k] and s_norm[k] < eps_dual[k] and k > 0:
                #     break
                # #Compute primal residual
                # r = x - z

                total_losses[k, 0] = sum(local_losses)
                #
                iter_end = time.time()

                comm_time.append(recv_end_1 - recv_start + recv_end_3 - recv_end_2)
                total_time.append(iter_end - iter_start)
                inv_time.append(inv_time_end - inv_time_start)
                #
                print("iteration %f\t total loss %f" %(k, total_losses[k, 0]))

            np.save(directory + '/z_hist_Sinit' + str(S_init) + '_N' + str(N) + '_tau' + str(tau_max)
             + '_m' + str(m) + '_n' + str(n) + '_' + str(repeatition), z_hist)
            np.save(directory + '/total_losses_Sinit' + str(S_init) + '_N' + str(N) + '_tau' + str(tau_max)
             + '_m' + str(m) + '_n' + str(n) + '_' + str(repeatition), total_losses)
            np.save(directory + '/num_update_Sinit' + str(S_init) + '_N' + str(N) + '_tau' + str(tau_max)
             + '_m' + str(m) + '_n' + str(n) + '_' + str(repeatition), num_update)
            np.save(directory + '/comm_time_Sinit' + str(S_init) + '_N' + str(N) + '_tau' + str(tau_max)
             + '_m' + str(m) + '_n' + str(n) + '_' + str(repeatition), comm_time)
            np.save(directory + '/total_time_Sinit' + str(S_init) + '_N' + str(N) + '_tau' + str(tau_max)
             + '_m' + str(m) + '_n' + str(n) + '_' + str(repeatition), total_time)
            np.save(directory + '/inv_time_Sinit' + str(S_init) + '_N' + str(N) + '_tau' + str(tau_max)
             + '_m' + str(m) + '_n' + str(n) + '_' + str(repeatition), inv_time)
            toc = time.time() - tic
            print("\nElapsed time is %.2f seconds"%toc)

        
        else:

            #save a matrix-vector multiply
            Atb = A.T.dot(b)
            
            #initialize ADMM solver
            x = np.zeros((n,1))
            z = np.zeros((n,1))
            u = np.zeros((n,1))
            r = np.zeros((n,1))

            send = np.zeros(3)
            recv = np.zeros(3)    
            local_loss = np.zeros(1)
            total_loss = np.zeros(1)

            # cache the (Cholesky) factorization
            L,U = factor(A, rho)

            # Saving state
            objval     = []
            r_norm     = []
            s_norm     = []
            eps_pri    = []
            eps_dual   = []

            '''
            ADMM solver loop
            '''
            k = 0
            while finished != 1.:
                # print("iteration %f\t rank %f" %(k, rank))
                # u-update
                u= u + x - z
                 
                # x-update 
                q = Atb + rho * (z - u) #(temporary value)

                if m >= n: # skinny matrix, use inverse of (A^T A + \rho I)
                    x = np.linalg.solve(U,np.linalg.solve(L, q))
                else:
                    ULAq = np.linalg.solve(U, np.linalg.solve(L, A.dot(q)))
                    x = (q * 1./rho)-((A.T.dot(ULAq))*1./(rho**2))

                w = x + u # w would be sent to calculate z

                # calculate the local loss
                local_loss[0] = objective(A, b, mylambda * 1./N, z)

                # pack things to be sent
                to_send = np.zeros((n+1, 1))
                to_send[0:-1, 0] = w.reshape((w.shape[0], ))
                to_send[-1, :] = local_loss[0]

                # send x+u to the master node non-blocking
                req = comm.isend(to_send, dest = 0, tag=13)
                # req.wait()
                
                # receive z from the master node blocking
                to_receive = np.zeros((n+1, 1))
                req0 = comm.irecv(1000000, source = 0, tag = 13)
                to_recieve = req0.wait()
                temp1 = to_recieve[0:-1, 0]
                temp2 = to_recieve[-1, 0]
                z = temp1.reshape((temp1.shape[0], 1))
                finished = temp2
                
                #Compute primal residual
                r = x - z
                k += 1
    return 1

def objective(A, b, mylambda, z):
    return .5 * norm(A.dot(z) - b)**2 + mylambda * norm(z, 1)

def factor(A, rho):
    m_local,n_local = A.shape
    if m_local >= n_local:
       L = cholesky(A.T.dot(A) + rho * np.identity(n_local))
    else:
       L = cholesky(np.identity(m_local) + 1./rho * (A.dot(A.T)))
    U = L.T
    return L,U

def soft_threshold(v, k):
    return np.maximum(0., v - k) - np.maximum(0., -v - k)

if __name__=='__main__':
    main()
