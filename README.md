# admm_asynchoronous

Despite the superior performance of many state-of-the-art algorithms, solving huge-scale convex optimization problems in serial is infeasible. Distributed optimization has thus become indispensable in solving such problems. In other problems, many agents seek to minimize the sum of the objective functions collectively, yet they only have access to local objective functions and decision variables. Distributed optimization is a necessary tool to solve such problems. This project explores one of the distributed optimization algorithm: Alternating Direction Method of Multipliers (ADMM)[1] in solving huge-scale LASSO problems. In addition to the standard (synchronous) ADMM algorithm, asynchronous (communication-avoiding) ADMM algorithms [2][3] are investigated as well.

In this project, we solve an enormous LASSO problem: $A$ has $m = 2,457,600$ samples and $n = 20,000$ features, with a size about 400 GB! The synchronous ADMM algorithms converges in 92 steps (746 seconds)! Such a big problem cannot be solved serially at all.

To deal with the convergence problem of the asynchronous ADMM algorithm, an hybrid ADMM algorithm is proposed in this project, in which $S$ increases to $N$ gradually. Specifically, $S$ increments by a preset number $\Delta S$ in each master iteration after the number of master iterations reaches a preset cutoff value $k_{cut}$. The hybrid algorithms blends the convergence guarantee of the synchronous ADMM algorithm and the fast speed of the asynchronous ADMM algorithms. 

To run the code in NERSC's Cori:
srun -N 4 -n 128 --cpu_bind=cores python asynchronous_async.py 128 5 0 200 4800 20000 200
128 is initial S, if S = N, synchronous; if S < N, asynchronous
5 is the \tau
0 is the remote computer (1 is the local Mac machine)
200 is the total number of iterations
4800 is m_i, the number of rows of the local A_i matrix
20000 is n, the number of colomns of the local A_i matrix, the same for all ranks
200 is k_{cutoff}, when k >= k_{cutoff}, the S will increment by 3 after each master iteration, if S < N


References:
[1] Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. Distributed opti- mization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1):1–122, 2011.
[2] Ruiliang Zhang and James T. Kwok. Asynchronous distributed admm for consensus opti- mization. In Proceedings of the 31st International Conference on International Conference on Machine Learning - Volume 32, ICML’14, pages II–1701–II–1709. JMLR.org, 2014.
[3] E. Wei and A. Ozdaglar. On the o(1/k) convergence of asynchronous distributed alternating direction method of multipliers. In 2013 IEEE Global Conference on Signal and Information Processing, pages 551–554, Dec 2013.
[4] http://mpi4py.scipy.org/docs/.
[5] http://www.nersc.gov/users/computational-systems/cori/.