# Ising-Model-Playground

Consider the classical 2D Ising Hamiltonian
```math
\mathcal{H}[\{s_i\}]=-J \, \sum_{NN} s_i \, s_j- H\,\sum s_i
```
and the partition function at inverse temperature $\beta$
```math
\mathcal{Z}=\exp\left(-\beta \sum_{\{s_i\}} \mathcal{H}[\{s_i\}]\right)
```
for a given configuration of spins $\{s_i\}$ on an $L^2$ lattice with periodic boundary conditions and nearst neighbor (NN) interactions. In this notebook, we will examine aspects of Monte Carlo simulations in the context of the spin 1/2 square lattice Ising model, and using simulated data, we will explore various data analysis techniques. The idea of this notebook is to tie together the physics of the Ising model, Monte Carlo simulations, and statistical analysis into one project. Below we give a rough sketch of the outline of this project. Furthermore, this project is a work in progres and will used expanded upon as new topics are visited.


Monte Carlo Ising model simulation             |  Comparison of update methods
:-------------------------:|:-------------------------:
![IM](https://github.com/user-attachments/assets/7f47c924-2faf-4ca6-badc-62595b115630)  |  ![image](https://github.com/user-attachments/assets/34a1db2f-90ad-4dfb-ba6a-1309bb129de5)




__Sections:__
1. Exact Ising Model Computations for Small Lattices
    - Generation of configurations, thermodynamic functions, density of states, energy distributions
    - Computational complexity - basic algorithm scaling
2. Monte Carlo Algorithms
    - Markov Chains and Markov Chain Monte Carlo (MCMC)
    - Implementations: Metropolis-Hastings, Heatbath (Glauber), Wolff, and Wang-Landau
    - Diagnostics of MCMC: equilibriation, trace plots, initial condition dependence, autocorrelation
    - Error analysis: benchmarking against exact results, effective sample size, thinning
3. Extensions (not done)
   - More error analysis: Batching, jackknife, bootstrap
   - Rewighting (multiple histogram techniques), parallel tempering, simulated annealing
4. Statistical Methodologies
	<ol type="a">
	  <li> Estimation of $\beta$ and $H$: maximum likelihood </li>
	  <li> Phase transitions: principle component analysis as machine learning </li>
		<li> Logistic regression for the nearest-neighbor coupling $J$. (Not done) </li>
		<li> ANOVA for Monte Carlo runs over ranges of initial conditions. (Not done) </li>
	</ol>
5. Future Extensions
   - Computations for generalizations of the Ising model:
   		- Higher order spin interactions, NN and NNN interactions, $\mathbb{Z}_n$ models, $O(N)$ spin models, spin $s$ Ising, Potts models, Blume-Capel, Ashkin-Teller, lattice field theories
   - Partition function zeros at complex H and complex T:
   		- Cummulants, Yang-Lee edge, hysterisis/metestable states
   - Monte-Carlo renormalization group 
   - Wang-Landau computation of universal data of Ising critical point
   - Wang-Landau computation of the universal location of the Yang-Lee edge singulairty

__Discussion of Code:__
1. Sections 1 and 2 are contained in a single Jupyter notebook, Ising_model_playground,  so as the be more akin to lectures notes or a textbook.
2. Section 3 is contained in a single Jupyter notebook
3. Section 4 are both split up into multiple jupyter notebooks, one for each subsection, IMP_MLE and IMP_PCA.
4. This project will be a work in progress, updated as new work is done.
