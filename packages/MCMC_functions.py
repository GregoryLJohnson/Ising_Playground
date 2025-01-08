#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import scipy, numpy as np


# In[ ]:


def coldstart(L): 
    ''' coldstart(L)
    LxL Matrix full of aligned spins with s = +1 '''
    return np.ones((L,L)) 

def hotstart(L): 
    ''' hotstart(L)
    LxL Matrix of random spin '''
    return np.random.choice([-1,1],size=(L,L))

def antialigned(L):
    ''' antialigned(L)
    LxL Checkerboard (Using Timothy Budds code - only produces 2d lattice)'''
    if L % 2 == 0:
        return np.tile([[1,-1],[-1,1]],(L//2,L//2))
    else:
        return np.tile([[1,-1],[-1,1]],((L+1)//2,(L+1)//2))[:L,:L]


# In[ ]:


def auto_corr(x): 
    '''auto_corr(x)
    Computes the autocorrelation of an array x
    '''
    ## scipy signal.correlate is signifcantly faster than 
    #                 numpy's correlate which is significantly faster than hand-made code
    l = len(x)
    avg = np.mean(x)
    var = np.var(x)
    x = [i-avg for i in x]
    #acov = np.correlate(x,x,"full")[l-1:]
    auto_covariance = scipy.signal.correlate(x,x,"full")[l-1:]
    return auto_covariance/(l*var)

def exp_corr_len(x):
    '''exp_corr_len(x)
    Computes the exponential correlation length of an array x
    '''
    def comp(s):
        if s > np.exp(-1):
            return 0
        else:
            return 1
    partition = [comp(i) for i in x]
    return partition.index(1)  


def int_corr_time_set(x, t_lag): 
    '''int_corr_time_set(x, t_lag)
    Computes the integrated correlation length of an array x with 
    a maximum time lag given by t_lag.
    '''
    ## Integrated corr time. Since auto_corrs are computed all at once, 
    ##   compute correlations time all at once
    l = len(x)
    avg = np.mean(x)
    var = np.var(x)
    x = [i-avg for i in x]
    auto_covariance = scipy.signal.correlate(x, x, "full")[l-1: ]
    #auto_covariance = np.correlate(x, x, "full")[l-1: ]
    auto_correlation = [ 1 + 2*np.sum(auto_covariance[:i+1])/(l*var)  for i in range(t_lag-1)]
    return np.array([1] + auto_correlation)


# In[ ]:


def Metropolis_update(Latt, beta, H = 0, N_iterations = int(1e3) ):
    ''' Metropolis_update(Latt, beta, H = 0, N_iterations = int(1e3)) 
    
    The Metropolis algorithm for a square spin lattice nearest neighbor Ising model
    
    Parameters
    __________
    Latt : square int array
        A square lattice of spins taking values +1 or -1
    beta : float
        The inverse temperature. Must be a positive number.
    H : float
        The external magnetic field. Default is 0.
    N_iterations : int 
        The number of Monte Carlo steps to perform. Typically given as a number of sweeps, 
        i.e. an integer times the number of spins. Defaults to 1e3
        
    Returns
    _______
    The trace of magnetization and energy - an array of floats of size (N_iterations,2) - at each Monte Carlo step.
    
    '''
    
    J = 1
    L = len(Latt)
    
    m = np.sum(Latt) # the magnetization
    
    e = 0
    for i in range(L):
        for j in range(L):
            # move through lattice and compute s_i * neighbors
            # this double counts bounds, so divide by 2 afterwards
            e += -J*Latt[i, j] * (Latt[(i+1)%L, j] + Latt[(i-1)%L, j]
                                  + Latt[i, (j+1)%L]+ Latt[i, (j-1)%L])
    e = e/2 - H*m
    
    trace=np.zeros((N_iterations,2)) # trace of energy and magnetization
    
    for s in range(N_iterations):
        # choose a random spin to flip - convince youself of the dE expression, do it by hand for a small lattice
        i, j = np.random.randint(0, L, 2)
        de = -2 * (-J) * Latt[i, j] * (Latt[(i+1)%L, j] + Latt[(i-1)%L, j]
                                            + Latt[i, (j+1)%L]+ Latt[i, (j-1)%L])
        de = de + 2*H*Latt[i,j]

        if np.random.rand() < np.exp(-beta*de):
            Latt[i,j] *= -1 # flip the spin if random # < exp(-\beta de)
            e += de
            m += 2*Latt[i,j]
        
        trace[s] = [m,e]
    
    return trace


# In[ ]:


## Below is code adapted from Timothy Budd - only works for H=0

rng = np.random.default_rng()

def neighboring_sites(s,L):
    ''' neighboring_sites(s,L)
    Return the coordinates of the 4 sites adjacent to s on an LxL lattice.'''
    return [((s[0]+1)%L,s[1]),((s[0]-1)%L,s[1]),(s[0],(s[1]+1)%L),(s[0],(s[1]-1)%L)]

def Wolff_update(Latt, beta, H = 0, N_iterations = int(1e3)):
    ''' Wolff_update(Latt, beta, H = 0, N_iterations = int(1e3) ) 
    
    The Wolff cluster algorithm for a square spin lattice nearest neighbor Ising model
    
    Parameters
    __________
    Latt : square int array
        A square lattice of spins taking values +1 or -1
    beta : float
        The inverse temperature. Must be a positive number.
    H : float
        The external magnetic field. Default is 0.
    N_iterations : int 
        The number of Monte Carlo steps to perform. Typically given as a number of sweeps, 
        i.e. an integer times the number of spins. Defaults to 1e3.
        
    Returns
    _______
    The trace of magnetization and energy - an array of floats of size (N_iterations,2) - at each Monte Carlo step.
    
    '''
        
    J = 1
    L = len(Latt)
    p_add = 1 - np.exp(-2*beta)
    
    trace = np.zeros((N_iterations,2)) # trace of energy and magnetization
    
## Below is code for non-zero H $$

#     for s in range(N_iterations):
#         i, j = np.random.randint(0, L, 2)
#         proposal = Latt.copy()
#         seed = tuple([i,j])
#         spin = proposal[seed]
#         proposal[seed] = -spin
#         cluster_size = 1
#         unvisited = [seed]
#         while unvisited:
#             site = unvisited.pop()
#             for nbr in neighboring_sites(site,L):
#                 if proposal[nbr] == spin and rng.uniform() < p_add:
#                     proposal[nbr] = -spin
#                     unvisited.append(nbr)
#                     cluster_size += 1
        
#         de_h = 2*H*cluster_size
        
#         if np.random.rand() < min(1, np.exp(-beta * de_h )):
#             Latt = proposal

## The computation of energy and magnetization traces needs to be improved 

    for s in range(N_iterations):
        i, j = np.random.randint(0, L, 2)
        seed = tuple([i,j])
        spin = Latt[seed]
        Latt[seed] = -spin
        cluster_size = 1
        unvisited = [seed]
        while unvisited:
            site = unvisited.pop()
            for nbr in neighboring_sites(site,L):
                if Latt[nbr] == spin and rng.uniform() < p_add:
                    Latt[nbr] = -spin
                    unvisited.append(nbr)
                    cluster_size += 1
            
        m = np.sum(Latt) # the magnetization
        
        e = 0
        for i in range(L):
            for j in range(L):
                e += -J*Latt[i, j] * (Latt[(i+1)%L, j] + Latt[(i-1)%L, j]
                                      + Latt[i, (j+1)%L]+ Latt[i, (j-1)%L])
        e = e/2 - H*m
        
        trace[s] = [m,e]
        
        #print(Latt - coldstart(L))
        #print(Latt)
        #print(proposal)
    
    return trace


# In[ ]:


def Heatbath_update(Latt, beta, H = 0, N_iterations = int(1e3) ):
    ''' Heatbath_update(Latt, beta, H = 0, N_iterations = int(1e3)) 
    
    The Heatbath algorithm (Glauber dynamics) for a square spin lattice nearest neighbor Ising model
    
    Parameters
    __________
    Latt : square int array
        A square lattice of spins taking values +1 or -1
    beta : float
        The inverse temperature. Must be a positive number.
    H : float
        The external magnetic field. Default is 0.
    N_iterations : int 
        The number of Monte Carlo steps to perform. Typically given as a number of sweeps, 
        i.e. an integer times the number of spins. Defaults 10 1e3.
        
    Returns
    _______
    The trace of magnetization and energy - an array of floats of size (N_iterations,2) - at each Monte Carlo step.
    
    '''
    J = 1
    L = len(Latt)
    
    m = np.sum(Latt) # the magnetization
    
    e = 0
    for i in range(L):
        for j in range(L):
            # move through lattice and compute s_i * neighbors
            # this double counts bounds, so divide by 2 afterwards
            e += -J*Latt[i, j] * (Latt[(i+1)%L, j] + Latt[(i-1)%L, j]
                                  + Latt[i, (j+1)%L]+ Latt[i, (j-1)%L])
    e = e/2 - H*m
    
    trace=np.zeros((N_iterations,2)) # trace of energy and magnetization
    
    for s in range(N_iterations):
        # choose a random spin to flip - convince youself of the dE expression, do it by hand for a small lattice
        i, j = np.random.randint(0, L, 2)
        de = -2 * (-J) * Latt[i, j] * (Latt[(i+1)%L, j] + Latt[(i-1)%L, j]
                                            + Latt[i, (j+1)%L]+ Latt[i, (j-1)%L])
        de = de + 2*H*Latt[i,j]

        if np.random.rand() < 1/(1+np.exp(beta*de)):
            Latt[i,j] *= -1 # flip spin and increment energy and magnetization
            e += de
            m += 2*Latt[i,j]
        
        trace[s] = [m,e]
    
    return trace


# In[ ]:


def E_list_gen(L):
    ''' E_list_gen(L)
    Generates list of energies for even L '''
    
    if L % 2 != 0:
        raise ValueError("This list generation only works for even side lengths")
        
    E_list=[-2*L**2 + 4*k for k in range(0,L**2+1)]
    E_list.pop(1)
    E_list.pop(-2)
    E_list=np.array(E_list)  # energy states = N-1 
    
    # Emax = 2*L**2
    # [(e+Emax)//4 for e in E_list]
    
    return E_list


def Wang_Landau_update(L, N_sweeps = int(1e3), flatness = .2, logf = 1, fmod = 1/2, f_criteria = 1e-8):
    ''' Wang_Landau_update(L, N_sweeps = 10**3, flatness = .2, logf = 1, fmod = 1/2, f_criteria = 1e-8)  
    
    The Heatbath algorithm (Glauber dynamics) for a square spin lattice nearest neighbor Ising model
    
    Parameters
    __________
    L : int
        The linear size of a square Ising spin lattice
    N_sweeps : int
        The number of Monte Carlo sweeps to perform. A sweep corresponds to a Monte Carlo step for 
        each lattice spin. Defaults to 1e3.
    flatness: float
        The percent of flatness to achieve for the energy histrograms. The minimum bin and maximum bin 
        should only differ from the mean by a percent given by flatness. Defaults to .2.
    logf : float
        The inital value for the log of the modification Wang-Landau factor f. Defaults to 1.
    fmoad : float, typically a rational
        The modification of logf after each succesful iteration ( when the histograms is flat). 
        If f is replaced with f^{fmod}, logf becomes fmod*logf. Defaults to 1/2.
    f_criteria: float
        The lower bound on logf, after which the algorithm terminates. Defaults to 1e-8.
        
    Returns
    _______
    Two arrays - lngE_list, hE_list. 
        The float array lngE_list contains the logarithm of the density of states. 
    The integer array hE_list contains the final energy histogram. Intermediate energy histograms are not stored.
    '''
    
    # Initialize using fully aligned state
    E_list = E_list_gen(L)
    lngE_list = np.zeros_like(E_list, dtype = float)
    hE_list = np.zeros_like(E_list, dtype = int)
    state = coldstart(L)
    e1 = -2*L**2
    index1 = np.argwhere( E_list == e1)[0,0]

    while logf > f_criteria:
        # reset the histogram each iteration
        hE_list.fill(0)
        iteration = 0
        # run while loop while not(h_min < .8 h_mean and h_max > 1.2 h_mean)
        while np.min(hE_list) <= (1-flatness)*np.mean(hE_list) or np.max(hE_list) >= (1+flatness)*np.mean(hE_list):
            for s in range(N_sweeps*L**2):                
                i,j = rng.integers(0, L, 2)
                de = 2*state[i,j]*( state[i, (j+1)%L] + state[i, (j-1)%L] + 
                                   state[(i+1)%L, j] + state[ (i-1)%L, j] )
                e2 = e1 + de
                index2 = np.argwhere( E_list == e2)[0,0]
                
                if np.random.random() < np.exp(lngE_list[index1]-lngE_list[index2]):
                    state[i,j] *= -1
                    e1 = e2
                    index1 = index2
                else:
                    index1 = index1
                    
                lngE_list[index1] += logf
                hE_list[index1] += 1 
                iteration += 1
                
        print(" log(f) = ", logf, " and sweeps =", iteration//L**2)
        logf *= 1/2

    return lngE_list, hE_list

