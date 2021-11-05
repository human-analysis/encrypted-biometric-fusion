#import numpy as np
import random

#import autograd.numpy as np  # Thinly-wrapped numpy
#from autograd import grad
#from matplotlib import pyplot
from sklearn.decomposition import PCA
import math

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from model import Linear_Feature_Fusion

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

def main():
    A,B,L = data_gen(10)
    alpha = A.shape[1]
    beta = B.shape[1]
    gamma = 2
    #P = torch.rand(alpha+beta,gamma)
    
    lamb = 0.2
    margin = 1#math.cos(math.pi/6)
    X = torch.cat((A,B),dim=1)
    M = []
    V = []
    for i in range(X.shape[0]):
        for j in range(i+1,X.shape[0]):
            if L[i] == L[j]:
                M.append((i,j))
                for k in range(X.shape[0]):
                    if L[k] != L[i]:
                        V.append((i,j,k))
    
    model = Linear_Feature_Fusion(X,M,V,gamma,margin,lamb)
    
    iterations = 100
    break_point = 0.005
    rate = 10e-2
    #for i in range(iterations):
    #    index = random.randint(0,X3.shape[0]-1)
    #    sample = X3[index]
    #    label = L[index]
    print("Initial loss:",model.loss())
    P_history = []
    losses = []
    optim = torch.optim.SGD(model.parameters(), lr=rate, momentum=0.9)
    for i in range(iterations):
        loss = model.loss()
        losses.append(loss)
        print(model.P)
        P_history.append(str(model.P.tolist()))
        loss.backward()
        optim.step()
    print("final loss:",model.loss())
    
    outfile = open("P_values.txt",'w')
    for P_value in P_history:
        outfile.write(P_value)
        outfile.write("\n")
    outfile.close()
    
    outfile_loss = open("loss_values.txt",'w')
    for loss_value in losses:
        outfile_loss.write(str(loss_value.tolist()))
        outfile_loss.write("\n")
    outfile_loss.close()
    
    """
    loss_gradient_function = grad(Loss)
    last_loss = Loss(P)
    for i in range(iterations):
        losses.append(last_loss)
        if i%10 == 0:
            print("Iteration:",i)
        gradient = loss_gradient_function(P)
        #if i%10 == 0:
        #    print(np.linalg.norm(gradient))
        if np.linalg.norm(gradient) < break_point:
            print("Converged on iteration",i)
            break
        P -= gradient * rate
        loss = Loss(P)
        #if loss < last_loss:
        #    alpha = 0.97 * alpha
        last_loss = loss
    print("Final loss:",Loss(P))
    """

if __name__ == "__main__":
    main()