# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:47:00 2021

@author: lsper
"""
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

def data_gen(num_samples):
    alpha=3
    beta=2
    covar1 = torch.eye(alpha) / 6.0
    covar2 = torch.eye(beta) / 6.0
    #covar1 = np.identity(alpha) / 6.0
    #covar2 = np.identity(beta) / 6.0
    A = []
    A_distr1 = MultivariateNormal(loc=torch.tensor([1.,2.,3.]), covariance_matrix=covar1)
    A_distr2 = MultivariateNormal(loc=torch.tensor([-1.,-2.,-3.]), covariance_matrix=covar1)
    A_distr3 = MultivariateNormal(loc=torch.tensor([0.,0.,3.]), covariance_matrix=covar1)
    #print(A_distr1.rsample())
    #Adistr4 = MultivariateNormal(loc=torch.tensor([0.,0.,3.]), covariance_matrix=covar1)
    for i in range(num_samples):
        A.append(A_distr1.rsample().unsqueeze(dim=0))
        #A.append(np.random.multivariate_normal(np.array([1.,2.,3.]), covar1))
    for i in range(num_samples):
        #A.append(np.random.multivariate_normal(np.array([-1.,-2.,-3.]), covar1))
        A.append(A_distr2.rsample().unsqueeze(dim=0))
    for i in range(num_samples):
        A.append(A_distr3.rsample().unsqueeze(dim=0))
        #A.append(np.random.multivariate_normal(np.array([0.,0.,3.]), covar1))
    for i in range(num_samples):
        A.append(A_distr3.rsample().unsqueeze(dim=0))
        #A.append(np.random.multivariate_normal(np.array([0.,0.,3.]), covar1))
    
    B = []
    B_distr1 = MultivariateNormal(loc=torch.tensor([-4.,4.]), covariance_matrix=covar2)
    B_distr2 = MultivariateNormal(loc=torch.tensor([3.,-3.]), covariance_matrix=covar2)
    B_distr3 = MultivariateNormal(loc=torch.tensor([0.,3.]), covariance_matrix=covar2)
    #Bdistr4 = 
    for i in range(num_samples):
        B.append(B_distr1.rsample().unsqueeze(dim=0))
        #B.append(np.random.multivariate_normal(np.array([-4.,4.]), covar2))
    for i in range(num_samples):
        B.append(B_distr2.rsample().unsqueeze(dim=0))
        #B.append(np.random.multivariate_normal(np.array([3.,-3.,]), covar2))
    for i in range(num_samples):
        B.append(B_distr3.rsample().unsqueeze(dim=0))
        #B.append(np.random.multivariate_normal(np.array([0.,3.,]), covar2))
    for i in range(num_samples):
        B.append(B_distr2.rsample().unsqueeze(dim=0))
        #B.append(np.random.multivariate_normal(np.array([3.,-3.,]), covar2))
        
    A_final = torch.Tensor(num_samples*4,alpha)
    torch.cat(A, out=A_final,dim=0)
    B_final = torch.Tensor(num_samples*4,beta)
    torch.cat(B, out=B_final,dim=0)
    
    L = []
    for i in range(num_samples):
        L.append(0.)
    for i in range(num_samples):
        L.append(1.)
    for i in range(num_samples):
        L.append(2.)
    for i in range(num_samples):
        L.append(3.)
    
    L = torch.tensor(L).T
    
    return A_final,B_final,L