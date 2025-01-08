#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import itertools, numpy as np


# In[ ]:


def config_gen(L):
    '''Function to generate all lattice configurations for a given linear size L '''
    perm = [list(seq) for seq in itertools.product("ab", repeat=L*L)]
    perm = np.array(perm).reshape(2**(L*L), L, L)
    perm = np.where(perm=="a", 1, -1)
    return perm

def binary_rep(M):
    ''' Function to map a spin configuration M to binary then an integer between 0 and 2**N-1  '''
    N=np.shape(M)[0]
    M=np.reshape(M,N*N)
    M=np.where(M==-1,0,1)
    return sum([M[i]*2**i for i in range(len(M))])

def Ham(Latt,H):
    ''' The Hamiltonian for a square LxL spin lattice Ising model with
    nearest neighbor interactions and magnetic field H. We take J=1 and use scaled temperature T=J/T '''
    J = 1
    L = len(Latt) 
    Sum_H = -H*np.sum(Latt)
    Sum_J = 0
    for i in range(L):
        for j in range(L):
            # move through lattice and compute s_i * neighbors
            Sum_J += -J * Latt[i, j] * (Latt[(i+1)%L, j] + Latt[(i-1)%L, j]
                                        + Latt[i, (j+1)%L]+ Latt[i, (j-1)%L])
    return ( Sum_J/2 + Sum_H )


# In[ ]:


def E_list(L):
    ''' Computes nearest neighbor energies for all configurations for a given N at H=0'''
    return [Ham(M,0) for M in config_gen(L)]

def deg_E(L): 
    ''' Counts occurences of each E in E_list(N), returns the set of (E, deg(E)) '''
    el = E_list(L)
    degen = sorted([ (x,el.count(x)) for x in set(el)])
    #return [ i for i,j in degen]
    return degen

def degen_gen(E,L): 
    ''' Degenracy computation. This is too slow. Best to compute once degen_set and use in later functions'''
    degen_set = np.array(deg_E(L))
    i = np.where(degen_set[:,0] == E)[0][0]
    return degen_set[i][1]


# In[ ]:


# functions below use a global precomputed result for the energies and degeneracies for $L=2,3,4$
# In this cell, we compute once and for all the degeneracies used 

E_set = [ set(E_list(n)) for n in [2,3,4]]
degen_set = [deg_E(n) for n in [2,3,4]]


# In[ ]:


def part_func(L,T,H):
    ''' The partition function for a square LxL lattice, 
    nearest neighbor interactions, at temperature T and magnetic field H '''
    
    return np.sum([np.exp(-1/T*Ham(M,H)) for M in config_gen(L) ])

def prob_E(L,T,H,E):
    ''' A function to the compute probabiliy of given energy E of 
    a square LxL lattice, nearest neighbor interactions, at temperature T and magnetic field H '''
    
    i = [2,3,4].index(L)
    degen_set_here = np.array(degen_set[i])
    if H != 0:
        el = [ Ham(M,H) for M in config_gen(L)]
        degen_set_here = np.array([ (x,el.count(x)) for x in set(el)])
    def degen(Es):
        j = np.where(degen_set_here[:,0] == Es)[0][0]
        return degen_set_here[j][1]
    part_func = np.sum( [g*np.exp(-1/T*e) for e,g in degen_set_here])
    return part_func**(-1)*degen(E)*np.exp(-1/T*E)

def prob_Eset(L,T,H,E_samples): 
    ''' A function to compute the probabiliy of a given sample of energies E_samples
    of a square LxL lattice, nearest neighbor interactions, at temperature T and magnetic field H '''

    n = len(E_samples)
    i = [2,3,4].index(L)
    degen_set_here=np.array(degen_set[i])
    if H != 0:
        el = [ Ham(M,H) for M in config_gen(L)]
        degen_set_here = np.array([ (x,el.count(x)) for x in set(el)])
    def degen(Es):
        j = np.where(degen_set_here[:,0] == Es)[0][0]
        return degen_set_here[j][1]
    part_func = np.sum( [g*np.exp(-1/T*e) for e,g in degen_set_here])
    return part_func**(-n)*np.prod( [(degen(e)*np.exp(-1/T*e))**(E_samples.count(e)) for e in set(E_samples) ] )

def log_prob_Eset(L,T,H,E_samples):
    ''' A function to compute the log-probabiliy of a given sample of energies E_samples
    of a square LxL lattice, nearest neighbor interactions, at temperature T and magnetic field H '''

    n = len(E_samples)
    i = [2,3,4].index(L)
    degen_set_here = np.array(degen_set[i])
    if H != 0:
        el = [ Ham(M,H) for M in config_gen(L)]
        degen_set_here = np.array([ (x,el.count(x)) for x in set(el)])
    def degen(Es):
        j = np.where(degen_set_here[:,0] == Es)[0][0]
        return degen_set_here[j][1]
    part_func = np.sum( [g*np.exp(-1/T*e) for e,g in degen_set_here])
    return -n*np.log(part_func) + np.sum([E_samples.count(E)*(-1/T*E +np.log(degen(E))  ) for E in set(E_samples) ])     


# In[ ]:




