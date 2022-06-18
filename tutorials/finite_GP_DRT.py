
import numpy as np
from numpy import inf, pi
from numpy.random import randn
from numpy.linalg import cholesky
from numpy.matlib import repmat
from math import pi, log, exp
from numpy import linalg as la
from numpy import *

def kernel(log_tau, log_tau_prime, sigma_f, ell):

    return (sigma_f**2)*exp(-0.5/(ell**2)*((log_tau-log_tau_prime)**2))

def compute_K(log_tau_vec, sigma_f, ell):

    N_tau = log_tau_vec.size
    out_K = np.zeros((N_tau, N_tau))

    for m in range(0, N_tau):

        log_tau_m = log_tau_vec[m]

        for n in range(0, N_tau):

            log_tau_n = log_tau_vec[n]
            out_K[m,n] = kernel(log_tau_m, log_tau_n, sigma_f, ell)
    
    out_K = 0.5*(out_K+out_K.T)
    return out_K

def compute_A_re(freq_vec, tau_vec):
    
    omega_vec = 2.*pi*freq_vec
    log_tau_vec = np.log(tau_vec)

    # number of elements in tau and freqs
    N_tau = tau_vec.size
    N_f = freq_vec.size

    # define output function
    out_A_re = np.zeros((N_f, N_tau))

    # integrand
    f_re = lambda omega, log_tau: 1./(1+(omega*exp(log_tau))**2)

    for m in range(0, N_f):
        
        for n in range(0, N_tau):
            
            if n == 0:
                log_tau_center = log_tau_vec[n]
                log_tau_right = 0.5*(log_tau_vec[n]+log_tau_vec[n+1])

                Delta_np1 = log_tau_vec[n+1]-log_tau_vec[n]

                a_vec = 1/4*np.array([Delta_np1, Delta_np1])
                I_vec = np.array([f_re(omega_vec[m], log_tau_center), 
                                    f_re(omega_vec[m], log_tau_right)])

            elif n == N_tau-1:
                log_tau_left = 0.5*(log_tau_vec[n-1]+log_tau_vec[n])
                log_tau_center = log_tau_vec[n]

                Delta_nm1 = log_tau_vec[n]-log_tau_vec[n-1]
                a_vec = 0.25*np.array([Delta_nm1, Delta_nm1])
                I_vec = np.array([f_re(omega_vec[m], log_tau_left), 
                                    f_re(omega_vec[m], log_tau_center)])

            else:
                log_tau_left = 0.5*(log_tau_vec[n-1]+log_tau_vec[n])
                log_tau_center = log_tau_vec[n]
                log_tau_right = 0.5*(log_tau_vec[n]+log_tau_vec[n+1])

                Delta_nm1 = log_tau_vec[n]-log_tau_vec[n-1]
                Delta_np1 = log_tau_vec[n+1]-log_tau_vec[n]

                a_vec = 0.25*np.array([Delta_nm1, Delta_nm1+Delta_np1, Delta_np1])
                I_vec = np.array([f_re(omega_vec[m], log_tau_left), 
                                    f_re(omega_vec[m], log_tau_center), 
                                    f_re(omega_vec[m], log_tau_right)])

            out_A_re[m,n] = np.dot(a_vec, I_vec)
            
    return out_A_re

def compute_A_im(freq_vec, tau_vec):
    
    omega_vec = 2.*pi*freq_vec
    log_tau_vec = np.log(tau_vec)

    # number of elements in tau and freqs
    N_tau = tau_vec.size
    N_f = freq_vec.size

    # define output function
    out_A_im = np.zeros((N_f, N_tau))

    # integrand
    f_im = lambda omega, log_tau: -omega*exp(log_tau)/(1+(omega*exp(log_tau))**2)

    for m in range(0, N_f):
        
        for n in range(0, N_tau):
            
            if n == 0:
                log_tau_center = log_tau_vec[n]
                log_tau_right = 0.5*(log_tau_vec[n]+log_tau_vec[n+1])

                Delta_np1 = log_tau_vec[n+1]-log_tau_vec[n]

                a_vec = 1/4*np.array([Delta_np1, Delta_np1])
                I_vec = np.array([f_im(omega_vec[m], log_tau_center), 
                                    f_im(omega_vec[m], log_tau_right)])

            elif n == N_tau-1:
                log_tau_left = 0.5*(log_tau_vec[n-1]+log_tau_vec[n])
                log_tau_center = log_tau_vec[n]

                Delta_nm1 = log_tau_vec[n]-log_tau_vec[n-1]
                a_vec = 0.25*np.array([Delta_nm1, Delta_nm1])
                I_vec = np.array([f_im(omega_vec[m], log_tau_left), 
                                    f_im(omega_vec[m], log_tau_center)])

            else:
                log_tau_left = 0.5*(log_tau_vec[n-1]+log_tau_vec[n])
                log_tau_center = log_tau_vec[n]
                log_tau_right = 0.5*(log_tau_vec[n]+log_tau_vec[n+1])

                Delta_nm1 = log_tau_vec[n]-log_tau_vec[n-1]
                Delta_np1 = log_tau_vec[n+1]-log_tau_vec[n]

                a_vec = 0.25*np.array([Delta_nm1, Delta_nm1+Delta_np1, Delta_np1])
                I_vec = np.array([f_im(omega_vec[m], log_tau_left), 
                                    f_im(omega_vec[m], log_tau_center), 
                                    f_im(omega_vec[m], log_tau_right)])

            out_A_im[m,n] = np.dot(a_vec, I_vec)
            
    return out_A_im


# Find the nearest positive-definite matrix

#"""
#Returns true when input is positive-definite, via Cholesky
#is a matrix positive definite?
#if input matrix is positive-definite (<=> Cholesky decomposable), then true is returned otherwise return false
#"""

def is_PD(A):
      
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_PD(A):
    
    # based on 
    # N.J. Higham (1988) https://doi.org/10.1016/0024-3795(88)90223-6
    # and 
    # https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    B = (A + A.T)/2
    _, Sigma_mat, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(Sigma_mat), V))

    A_nPD = (B + H) / 2
    A_symm = (A_nPD + A_nPD.T) / 2

    k = 1
    I = np.eye(A_symm.shape[0])

    while not is_PD(A_symm):
        eps = np.spacing(la.norm(A_symm))

        # MATLAB's 'chol' accepts matrices with eigenvalue = 0, numpy does not not. 
        # So where the matlab implementation uses 'eps(mineig)', we use the above definition.

        min_eig = min(0, np.min(np.real(np.linalg.eigvals(A_symm))))
        A_symm += I * (-min_eig * k**2 + eps)
        k += 1

    return A_symm

    
# calculate the negative marginal log-likelihood (NMLL)
def NMLL_fct(theta, A, Z_exp_re_im, N_freqs, log_tau_vec):

    # load the value of the parameters
    sigma_n = theta[0]
    sigma_R = theta[1]
    sigma_f = theta[2]
    ell = theta[3]

    # number of N
    N_taus = log_tau_vec.size   

    # Gamma
    Gamma = np.zeros((N_taus+1, N_taus+1))
    Gamma[0,0] = sigma_R**2
    # compute the K matrix
    K = compute_K(log_tau_vec, sigma_f, ell)
    Gamma[1:, 1:] = K

    # put together the Gamma matrix
    Psi = A@(Gamma@A.T)+(sigma_n**2)*np.eye(2*N_freqs)
    Psi = 0.5*(Psi + Psi.T) # symmetrize
    
    # Cholesky decomposition of Psi
    if(is_PD(Psi)==False):
        Psi = nearestPD(Psi)
    else:
        Psi = Psi
        
    L = np.linalg.cholesky(Psi)
    
    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp_re_im)
    alpha = np.linalg.solve(L.T, alpha)

    return 0.5*np.dot(Z_exp_re_im,alpha) + np.sum(np.log(np.diag(L)))

def NMLL_L_fct(theta, A, Z_exp_re_im, N_freqs, log_tau_vec):

    # load the value of the parameters
    sigma_n = theta[0]
    sigma_L = theta[1]
    sigma_R = theta[2]
    sigma_f = theta[3]
    ell = theta[4]

    # number of N
    N_taus = log_tau_vec.size

    # Gamma
    Gamma = np.zeros((N_taus+2, N_taus+2))
    Gamma[0,0] = sigma_L**2
    Gamma[1,1] = sigma_R**2
    # compute the K matrix
    K = compute_K(log_tau_vec, sigma_f, ell)
    Gamma[2:, 2:] = K

    # put together the Gamma matrix
    Psi = A@(Gamma@A.T)+(sigma_n**2)*np.eye(2*N_freqs)
    Psi = 0.5*(Psi + Psi.T) # symmetrize
    
    if(is_PD(Psi)==False):
        Psi = nearestPD(Psi)
    else:
        Psi = Psi
        
    # Cholesky decomposition of Psi
    L = np.linalg.cholesky(Psi)
    
    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp_re_im)
    alpha = np.linalg.solve(L.T, alpha)

    return 0.5*np.dot(Z_exp_re_im,alpha) + np.sum(np.log(np.diag(L)))

def generate_tmg(F, g, M, mu_r, initial_X, cov=True, L=1):

    """
    Implementation of the algorithm described in http://arxiv.org/abs/1208.4118
    Author: Ari Pakman

    Returns samples from a d-dimensional Gaussian with constraints given by F*X+g >0 
    If cov == true
    then M is the covariance and the mean is mu = mu_r 
    if cov== false 
    then M is the precision matrix and the log-density is -1/2 X'*M*X + r'*X

    Input
    F:          m x d array
    g:          m x 1 array 
    M           d x d array, must be symmmetric and definite positive
    mu_r        d x 1 array 
    cov:        see explanation above 
    L:          number of samples desired
    initial_X   d x 1 array. Must satisfy the constraint.

    Output
    Xs:      d x L array, each column is a sample

    """

    # sanity check
    m = g.shape[0]
    if F.shape[0] != m:
        print("Error: constraint dimensions do not match")
        return

    # using covariance matrix
    if cov:
        mu = mu_r
        g = g + F@mu
        
        ## Nearest Positive Definite 
        if(is_PD(M)==False):
            M = nearestPD(M)
        else:
            M = M
      
        R = cholesky(M)
        R = R.T #change the lower matrix to upper matrix
        F = F@R.T
        initial_X = initial_X -mu
        initial_X = np.linalg.solve(R.T, initial_X)
    # using precision matrix
    else:
        r = mu_r
        # Nearest Positive Definite 
        if(is_PD(M)==False):
            M = nearestPD(M)
        else:
            M = M
        R = cholesky(M)
        R = R.T #change the lower matrix to upper matrix
        mu = np.linalg.solve(R, np.linalg.solve(R.T, r))
        g = g + F@mu
        F = np.linalg.solve(R, F)
        initial_X = initial_X - mu
        initial_X = R@initial_X

    d = initial_X.shape[0]     # dimension of mean vector; each sample must be of this dimension
    bounce_count = 0
    nearzero = 1E-12

    # more for debugging purposes
    if (F@initial_X + g).any() < 0:
        print("Error: inconsistent initial condition")
        return

    # squared Euclidean norm of constraint matrix columns
    F2 = np.sum(np.square(F), axis=1)
    Ft = F.T

    last_X = initial_X
    Xs = np.zeros((d,L))
    Xs[:,0] = initial_X

    i=2

    # generate samples
    while i <=L:
        
        if i%1000 == 0:
            print('Current sample number',i,'/', L)
            
        stop = False
        j = -1
        # generate inital velocity from normal distribution
        V0 = randn(d)

        X = last_X
        T = pi/2
        tt = 0

        while True:
            a = np.real(V0)
            b = X

            fa = F@a
            fb = F@b

            U = np.sqrt(np.square(fa) + np.square(fb))
    #                print(U.shape)

            # has to be arctan2 not arctan
            phi = np.arctan2(-fa, fb)

            # find the locations where the constraints were hit
            pn = np.array(np.abs(np.divide(g, U))<=1)
            
            if pn.any():
                inds = np.where(pn)[0]
                phn = phi[pn]
                t1 = -1.0*phn + np.arccos(np.divide(-1.0*g[pn], U[pn]))
                
                # if there was a previous reflection (j > -1)
                # and there is a potential reflection at the sample plane
                # make sure that a new reflection at j is not found because of numerical error
                if j > -1:
                    if pn[j] == 1:
                        temp = np.cumsum(pn)
                        indj = temp[j]-1 # we changed this line
                        tt1 = t1[indj]
                        
                        if np.abs(tt1) < nearzero or np.abs(tt1 - pi) < nearzero:
    #                                print(t1[indj])
                            t1[indj] = inf
                
                mt = np.min(t1)
                m_ind = np.argmin(t1)
                # update j
                j = inds[m_ind]
                    
            else:
                mt = T
            
            # update travel time
            tt = tt + mt

            if tt >= T:
                mt = mt- (tt - T)
                stop = True

            # print(a)
            # update position and velocity
            X = a*np.sin(mt) + b*np.cos(mt)
            V = a*np.cos(mt) - b*np.sin(mt)

            if stop:
                break
            
            # update new velocity
            qj = F[j,:]@V/F2[j]
            V0 = V - 2*qj*Ft[:,j]
            
            bounce_count += 1

        if (F@X +g).all() > 0:
            Xs[:,i-1] = X
            last_X = X
            i = i+1

        else:
            print('hmc reject')    

        # need to transform back to unwhitened frame
    if cov:
        Xs = R.T@Xs + repmat(mu.reshape(mu.shape[0],1),1,L)
    else:
        Xs =  np.linalg.solve(R, Xs) + repmat(mu.reshape(mu.shape[0],1),1,L)

    # convert back to array
    return Xs