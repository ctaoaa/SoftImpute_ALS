#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import time
import random


# In[ ]:


def generate_X(m,n,r,prop,random_state=1):
    np.random.seed(random_state)
    random.seed(random_state)
    true_A = np.random.normal(size=(m,r))
    true_B = np.random.normal(size=(n,r))
    X = true_A.dot(true_B.T).flatten()
    nan_pos = random.sample(range(m*n),int(m*n*prop)) 
    X[nan_pos] = np.NAN
    return X.reshape(m,n)


# In[ ]:


class SoftImpute_ALS_subspace_dense:
    def __init__(self, X, Lambda, r, sc='variable', random_state=1, AB_inf=None):
        np.random.seed(random_state)
        self.X = X
        self.Lambda = Lambda
        self.r = r
        self.Omega_c = np.isnan(X)
        self.Omega = np.array([not self.Omega_c[i, j] for i in range(
            self.Omega_c.shape[0]) for j in range(self.Omega_c.shape[1])]).reshape(self.Omega_c.shape)
        self.sc = sc
        m, n = self.X.shape
        U = np.random.normal(size=(m, r))
        self.U, _, _ = np.linalg.svd(U, full_matrices=False)
        self.Dsq = np.eye(r)
        self.V = np.zeros((n, r))
        self.A = np.dot(self.U, np.sqrt(self.Dsq))
        self.B = np.dot(self.V, np.sqrt(self.Dsq))
        self.obj_list = [self.objective()]
        self.running_time = [0.0]
        self.rvar_ratio = []
        self.robj_ratio = []
        if AB_inf is not None:
            self.diff_indicator = True
            self.AB_inf = AB_inf
            self.diff_solu = []
            self.diff_solu.append(self.solution_difference())
            print("diff:",self.solution_difference())
        else:
            self.diff_indicator = False

    def solution_difference(self):
        return np.linalg.norm(self.A.dot(self.B.T)-self.AB_inf)**2/np.linalg.norm(self.AB_inf)**2

    def objective(self):
        return 1/2 * np.linalg.norm((self.X-np.dot(self.A, self.B.T))[self.Omega])**2 +             self.Lambda*np.sum(np.diag(self.Dsq))

    def objective_AB(self, A, B):
        return 1/2 * np.linalg.norm((self.X-np.dot(A, B.T))[self.Omega])**2 +             self.Lambda*np.linalg.norm(np.dot(A, B.T), ord='nuc')

    def objective_AB_F(self, A, B):
        return 1/2 * np.linalg.norm((self.X-np.dot(A, B.T))[self.Omega])**2 +             1/2*self.Lambda*(np.linalg.norm(A)**2+np.linalg.norm(B)**2)

    def X_star(self):
        X_star = np.zeros_like(self.X, np.float64)
        X_star[self.Omega] = self.X[self.Omega]
        X_star[self.Omega_c] = np.dot(self.A, self.B.T)[self.Omega_c]
        return X_star

    def Frob(self, U_old, V_old, Dsq_old):
        frob1 = np.sum(np.square(Dsq_old))
        frob2 = np.sum(np.square(self.Dsq))
        utu = self.Dsq * np.dot(U_old.T, self.U)
        vtv = Dsq_old * np.dot(self.V.T, V_old)
        frob3 = utu.dot(vtv).diagonal().sum()
        return (frob1+frob2-2*frob3)/frob1

    def update_B(self):
        X_star = self.X_star()
        d_sq = np.diag(self.Dsq)
        # \tilde{B}D
        B_hat = np.dot(X_star.T, np.dot(
            self.U, np.diag(d_sq/(d_sq+self.Lambda))))
        B_tilde = np.dot(X_star.T, np.dot(
            self.U, np.diag(np.sqrt(d_sq)/(d_sq+self.Lambda))))
        S, d, R = np.linalg.svd(B_hat, full_matrices=False)
        self.Dsq = np.diag(d)
        self.V = S
        self.U = np.dot(self.U, R.T)
        self.A = np.dot(self.U, np.sqrt(self.Dsq))
        self.B = np.dot(self.V, np.sqrt(self.Dsq))

    def update_A(self):

        X_star = self.X_star()
        X_star_T = X_star.T
        d_sq = np.diag(self.Dsq)
        # \tilde{A}D
        A_hat = np.dot(X_star_T.T, np.dot(
            self.V, np.diag(d_sq/(d_sq+self.Lambda))))
        S, d, R = np.linalg.svd(A_hat, full_matrices=False)
        self.Dsq = np.diag(d)
        self.U = S
        self.V = np.dot(self.V, R.T)
        self.A = np.dot(self.U, np.sqrt(self.Dsq))
        self.B = np.dot(self.V, np.sqrt(self.Dsq))

    def matrix_completion(self, rvar_eps=1e-5, robj_eps=1e-5, max_iter=100):
        print("Algorithm start!")
        for iter_ in range(max_iter):
            U_old = self.U
            V_old = self.V
            Dsq_old = self.Dsq
            start_time = time.perf_counter()
            # update B
            self.update_B()
            # update A
            self.update_A()
            end_time = time.perf_counter()
            self.running_time.append(end_time-start_time+self.running_time[-1])
            self.obj_list.append(self.objective())
            rvar_ratio = self.Frob(U_old, V_old, Dsq_old)
            self.rvar_ratio.append(rvar_ratio)

            robj_ratio = (
                self.obj_list[-2]-self.obj_list[-1])/np.abs(self.obj_list[-2])
            self.robj_ratio.append(robj_ratio)
            if self.diff_indicator:
                self.diff_solu.append(self.solution_difference())
                
            if self.sc == 'variable':
                converge_flag = rvar_ratio <= rvar_eps or iter_ == max_iter-1
            if self.sc == 'objective':
                converge_flag = robj_ratio <= robj_eps or iter_ == max_iter-1

            if converge_flag:
                print("iteration:", iter_+1)
                print("relative objective and variable change:",
                      robj_ratio, rvar_ratio)
                X_star = self.X_star()
                M = np.dot(X_star, self.V)
                S, d, R = np.linalg.svd(M, full_matrices=False)
                self.V = self.V.dot(R.T)
                self.U = S
                d -= self.Lambda
                d[d <= 0] = 0
                self.Dsq = np.diag(d)
                break


# In[ ]:


class SoftImpute_ALS_nonsubspace_dense:
    def __init__(self, X, Lambda, r,sc = 'variable',random_state=1, AB_inf=None):
        np.random.seed(random_state)
        self.X = X
        self.Lambda = Lambda
        self.r = r
        self.Omega_c = np.isnan(X)
        self.Omega = np.array([not self.Omega_c[i, j] for i in range(
            self.Omega_c.shape[0]) for j in range(self.Omega_c.shape[1])]).reshape(self.Omega_c.shape)
        m, n = self.X.shape
        U = np.random.normal(size=(m, r))

        self.A, _, _ = np.linalg.svd(U, full_matrices=False)
        self.B = np.zeros((n, r))
        self.obj_list = [self.objective_AB_F()]
        self.running_time = [0.0]
        self.rvar_ratio = []
        self.robj_ratio = []
        self.sc=sc
        if AB_inf is not None:
            self.diff_indicator = True
            self.AB_inf = AB_inf
            self.diff_solu = []
            self.diff_solu.append(self.solution_difference())
            print("diff:",self.solution_difference())
        else:
            self.diff_indicator = False

    def solution_difference(self):
        return np.linalg.norm(self.A.dot(self.B.T)-self.AB_inf)**2/np.linalg.norm(self.AB_inf)**2
    
    def objective_AB(self):
        return 1/2 * np.linalg.norm((self.X-np.dot(self.A, self.B.T))[self.Omega])**2 +             self.Lambda*np.linalg.norm(np.dot(self.A, self.B.T), ord='nuc')

    def objective_AB_F(self):
        return 1/2 * np.linalg.norm((self.X-np.dot(self.A, self.B.T))[self.Omega])**2 +             1/2*self.Lambda*(np.linalg.norm(self.A)**2 +
                             np.linalg.norm(self.B)**2)

    def X_star(self):
        X_star = np.zeros_like(self.X, np.float64)
        X_star[self.Omega] = self.X[self.Omega]
        X_star[self.Omega_c] = np.dot(self.A, self.B.T)[self.Omega_c]
        return X_star

    def update_B(self):
        X_star = self.X_star()
        inv = np.linalg.inv(self.A.T.dot(self.A)+self.Lambda*np.eye(self.r))
        self.B = X_star.T.dot(self.A.dot(inv))

    def update_A(self):
        X_star = self.X_star()
        inv = np.linalg.inv(self.B.T.dot(self.B)+self.Lambda*np.eye(self.r))
        self.A = X_star.dot(self.B.dot(inv))

    def Frob(self, A_old, B_old):
        ABT = np.dot(self.A, self.B.T)
        ABT_old = np.dot(A_old, B_old.T)
        
        if np.linalg.norm(ABT_old)==0:
            return np.NAN
        else:
            return np.linalg.norm(ABT-ABT_old)**2/np.linalg.norm(ABT_old)**2
 
    
    def matrix_completion(self, rvar_eps=1e-5, robj_eps = 1e-5,  max_iter=100):
        print("Algorithm start!")
        for iter_ in range(max_iter):
            A_old = self.A
            B_old = self.B
            start_time = time.perf_counter()
            # update B
            self.update_B()
            
            # update A
            self.update_A()
            end_time = time.perf_counter()
            self.running_time.append(end_time-start_time+self.running_time[-1])
            self.obj_list.append(self.objective_AB_F())
            rvar_ratio = self.Frob(A_old,B_old)
            self.rvar_ratio.append(rvar_ratio)
            
            robj_ratio = (self.obj_list[-2]-self.obj_list[-1])/np.abs(self.obj_list[-2])
            self.robj_ratio.append(robj_ratio)
            if self.diff_indicator:
                self.diff_solu.append(self.solution_difference())
            if self.sc =='variable':
                converge_flag = rvar_ratio <= rvar_eps or iter_ == max_iter-1
            if  self.sc =='objective':
                converge_flag = robj_ratio <= robj_eps or iter_ == max_iter-1
            
            if converge_flag:
                print("iteration:", iter_+1)
                print("relative objective and variable change:",robj_ratio,rvar_ratio)
                break


# In[ ]:


class als_svd_data:
    def __init__(self, U, Dsq, V):
        self.U = U
        self.Dsq = Dsq
        self.V = V


class Subspace_SVD_solver:
    def __init__(self, X, r, Lambda, warm_start=None):
        self.X = X
        m, n = X.shape
        self.r = r
        self.Lambda = Lambda
        self.warm_start = warm_start
        if warm_start:
            self.U = warm_start.U
            self.Dsq = warm_start.Dsq
            self.V = warm_start.V
        else:
            U = np.random.normal(size=(m, r))
            self.U, _, _ = np.linalg.svd(U, full_matrices=False)
            self.Dsq = np.eye(r)
            self.V = np.zeros((n, r))
       
  

    def update_V(self):
        X_star = self.X
        d_sq = np.diag(self.Dsq)
        B_hat = np.dot(X_star.T, np.dot(
            self.U, np.diag(d_sq/(d_sq+self.Lambda))))
        
        S, d, R = np.linalg.svd(B_hat, full_matrices=False)
        self.Dsq = np.diag(d)
        self.V = S
        self.U = np.dot(self.U, R.T)

    def update_U(self):
        X_star = self.X
        d_sq = np.diag(self.Dsq)
        A_hat = np.dot(X_star, np.dot(
            self.V, np.diag(d_sq/(d_sq+self.Lambda))))
        S, d, R = np.linalg.svd(A_hat, full_matrices=False)
        self.Dsq = np.diag(d)
        self.U = S
        self.V = np.dot(self.V, R.T)

    def Frob(self, U_old, V_old, Dsq_old):
        frob1 = np.sum(np.square(Dsq_old))
        frob2 = np.sum(np.square(self.Dsq))
        utu = self.Dsq * np.dot(U_old.T, self.U)
        vtv = Dsq_old * np.dot(self.V.T, V_old)
        frob3 = utu.dot(vtv).diagonal().sum()
        return (frob1 + frob2- 2 * frob3) / frob1

    def solve(self, eps=1e-10, max_iter=100):
        for iter_ in range(max_iter):
            U_old = self.U
            V_old = self.V
            Dsq_old = self.Dsq
            self.update_V()
            self.update_U()
            ratio =self.Frob(U_old, V_old, Dsq_old)
            if ratio <= eps or iter_ == max_iter-1:
                return self.U, self.Dsq, self.V.T


# In[ ]:


class SoftImpute_SVD_dense:
    def __init__(self, X, Lambda, r,sc = 'variable',random_state=1, AB_inf=None,warm_start=None):
        np.random.seed(random_state)
        self.X = X
        self.Lambda = Lambda
        self.r = r
        self.Omega_c = np.isnan(X)
        self.Omega = np.array([not self.Omega_c[i, j] for i in range(
            self.Omega_c.shape[0]) for j in range(self.Omega_c.shape[1])]).reshape(self.Omega_c.shape)
        m, n = self.X.shape
       
        
        self.running_time = [0.0]
        if warm_start  is not None:
            self.warm_start = warm_start
            self.U = warm_start.U
            self.Dsq = warm_start.Dsq
            self.V = warm_start.V
            
        else:
            U = np.random.normal(size=(m, r))
            self.U, _, _ = np.linalg.svd(U, full_matrices=False)
            self.Dsq = np.eye(r)
            self.V = np.zeros((n, r))
            self.warm_start = als_svd_data(self.U, self.Dsq, self.V)
        self.M = np.dot(self.U, self.Dsq.dot(self.V.T))
        self.obj_list = [self.objective()]
        self.rvar_ratio = []
        self.robj_ratio = []
        self.sc = sc
        if AB_inf is not None:
            self.diff_indicator = True
            self.AB_inf = AB_inf
            self.diff_solu = []
            self.diff_solu.append(self.solution_difference())
            print("diff:",self.solution_difference())
        else:
            self.diff_indicator = False

    def solution_difference(self):
        return np.linalg.norm(self.M-self.AB_inf)**2/np.linalg.norm(self.AB_inf)**2            
            
    def objective(self):
        return 1/2 * np.linalg.norm((self.X-self.M)[self.Omega])**2 +             self.Lambda*np.sum(np.diag(self.Dsq))

    def X_star(self):
        X_star = np.zeros_like(self.X, np.float64)
        X_star[self.Omega] = self.X[self.Omega]
        X_star[self.Omega_c] = self.M[self.Omega_c]
        return X_star

    def update(self):
        X_star = self.X_star()
        svd_solver = Subspace_SVD_solver(
            X_star, Lambda=self.Lambda, r=self.r, warm_start=self.warm_start)
        self.U, self.Dsq, est_VT = svd_solver.solve()
        self.V = est_VT.T
        self.M = np.dot(self.U, self.Dsq.dot(self.V.T))
        self.warm_start = als_svd_data(self.U, self.Dsq, self.V)

    def Frob(self, U_old, V_old, Dsq_old):
        frob1 = np.sum(np.square(Dsq_old))
        frob2 = np.sum(np.square(self.Dsq))
        utu = self.Dsq * np.dot(U_old.T, self.U)
        vtv = Dsq_old * np.dot(self.V.T, V_old)
        frob3 = utu.dot(vtv).diagonal().sum()
        return (frob1 + frob2 - 2 * frob3) / frob1

    def matrix_completion(self,  rvar_eps=1e-5, robj_eps=1e-5, max_iter=100):
        for iter_ in range(max_iter):
            U_old = self.U
            V_old = self.V
            Dsq_old = self.Dsq
            start_time = time.perf_counter()
            self.update()
            end_time = time.perf_counter()
            self.running_time.append(end_time-start_time+self.running_time[-1])

            self.obj_list.append(self.objective())
            rvar_ratio = self.Frob(U_old, V_old, Dsq_old)
            self.rvar_ratio.append(rvar_ratio)

            robj_ratio = (
                self.obj_list[-2]-self.obj_list[-1])/np.abs(self.obj_list[-2])
            self.robj_ratio.append(robj_ratio)
            if self.diff_indicator:
                self.diff_solu.append(self.solution_difference())

            if self.sc == 'variable':
                converge_flag = rvar_ratio <= rvar_eps or iter_ == max_iter-1
            if self.sc == 'objective':
                converge_flag = robj_ratio <= robj_eps or iter_ == max_iter-1

            if converge_flag:
                print("iteration:", iter_+1)
                print("relative objective and variable change:",
                      robj_ratio, rvar_ratio)
                break


# In[ ]:


class MMMF_ALS:
    def __init__(self, X, Lambda, r,sc = 'variable',random_state=1, AB_inf=None):
        np.random.seed(random_state)
        self.X = X
        self.Lambda = Lambda
        self.r = r
        self.Omega_c = np.isnan(X)
        self.Omega = np.array([not self.Omega_c[i, j] for i in range(
            self.Omega_c.shape[0]) for j in range(self.Omega_c.shape[1])]).reshape(self.Omega_c.shape)
        m, n = self.X.shape
        U = np.random.normal(size=(m, r))
        self.A, _, _ = np.linalg.svd(U, full_matrices=False)
        self.B = np.zeros((n, r))
        self.obj_list = [self.objective_AB_F()]
        self.running_time = [0.0]
        self.rvar_ratio = []
        self.robj_ratio = []
        self.sc=sc
        if AB_inf is not None:
            self.diff_indicator = True
            self.AB_inf = AB_inf
            self.diff_solu = []
            self.diff_solu.append(self.solution_difference())
            print("diff:",self.solution_difference())
        else:
            self.diff_indicator = False
            
    def solution_difference(self):
        return np.linalg.norm(self.A.dot(self.B.T)-self.AB_inf)**2/np.linalg.norm(self.AB_inf)**2
    
    def objective_AB_F(self):
        return 1/2 * np.linalg.norm((self.X-np.dot(self.A, self.B.T))[self.Omega])**2 +             1/2*self.Lambda*(np.linalg.norm(self.A)**2+np.linalg.norm(self.B)**2)

    def update_B(self):
        B_T = np.zeros_like(self.B.T)
        for i in range(B_T.shape[1]):
            P_Omega_i = self.Omega[:, i]
            if np.where(P_Omega_i == True)[0].shape[0]:
                P_X = self.X[:,i][P_Omega_i].reshape(-1,1)
                P_A = self.A[P_Omega_i]
                inv = np.linalg.inv(P_A.T.dot(P_A)+self.Lambda*np.eye(self.r))
                B_T[:, i] = np.dot(inv,P_A.T.dot(P_X))[:,0]
        self.B = B_T.T

    def update_A(self):
        A_T= np.zeros_like(self.A.T)
        Omega_T = self.Omega.T
        for i in range(A_T.shape[1]):
            P_Omega_i = Omega_T[:, i]
            if np.where(P_Omega_i == True)[0].shape[0]:
                P_X = self.X.T[:,i][P_Omega_i].reshape(-1,1) 
                P_B = self.B[P_Omega_i]
                inv = np.linalg.inv(P_B.T.dot(P_B)+self.Lambda*np.eye(self.r))
                A_T[:, i] = np.dot(inv,P_B.T.dot(P_X))[:,0]
        self.A = A_T.T         
        
        
    def Frob(self, A_old, B_old):
        ABT = np.dot(self.A, self.B.T)
        ABT_old = np.dot(A_old, B_old.T)
        if np.linalg.norm(ABT_old) == 0:
            return np.NAN
        else:
            return np.linalg.norm(ABT-ABT_old)**2/np.linalg.norm(ABT_old)**2

    def matrix_completion(self, rvar_eps=1e-5, robj_eps = 1e-5, max_iter=100):
        print('Algorithm start!')
        for iter_ in range(max_iter):
            A_old = self.A
            B_old = self.B
            start_time = time.perf_counter()
            # update B
            self.update_B()
            # update A
            self.update_A()
            end_time = time.perf_counter()
            self.running_time.append(end_time-start_time+self.running_time[-1])
            self.obj_list.append(self.objective_AB_F())
            
            rvar_ratio = self.Frob(A_old, B_old)
            self.rvar_ratio.append(rvar_ratio)
            
            robj_ratio = (self.obj_list[-2]-self.obj_list[-1])/np.abs(self.obj_list[-2])
            self.robj_ratio.append(robj_ratio)
            if self.diff_indicator:
                self.diff_solu.append(self.solution_difference())
            if self.sc =='variable':
                converge_flag = rvar_ratio <= rvar_eps or iter_ == max_iter-1
            if  self.sc =='objective':
                converge_flag = robj_ratio <= robj_eps or iter_ == max_iter-1
                
            if converge_flag:
                print("iteration:", iter_+1)
                print("relative objective and variable change:",robj_ratio,rvar_ratio)
                break

