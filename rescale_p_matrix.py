import numpy as np
import random
import os

#import autograd.numpy as np  # Thinly-wrapped numpy
#from autograd import grad
#from matplotlib import pyplot
from sklearn.decomposition import PCA
import math

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from model import Linear_Feature_Fusion_Approximate, Linear_Feature_Fusion
from data_generation import data_gen
import ast

from ROC import New_ROC_AUC

def rescale_p(p_file_name, scale):
    p_file = open(p_file_name,'r')
    
    p = []
    for line in p_file:
        result = torch.tensor(ast.literal_eval(line.strip()))
        p.append(result)
    p_final = torch.Tensor(len(p),p[0].shape[0])
    torch.cat(p, out=p_final,dim=0)
    
    p_final = torch.div(p_final,torch.linalg.norm(p_final))
    p_final = torch.mul(p_final,scale)
    
    print(p_final.shape)
    
    outfile_p_t = open(p_file_name,'w')
    P_final_t = p_final
    P_final_t = str(P_final_t.tolist())
    outfile_p_t.write(P_final_t)
    outfile_p_t.close()

def view_norms(p_file_name, a_file_name, b_file_name):
    p_file = open(p_file_name,'r')
    p = []
    #L = []
    for line in p_file:
        #result, l = line.strip().split(";")
        #print(line)
        result = torch.tensor(ast.literal_eval(line.strip()))
        #l = int(l)
        p.append(result)
        #L.append(l)
        p_final = torch.Tensor(len(p),p[0].shape[0])
    torch.cat(p, out=p_final,dim=0)
    

    a_file = open(a_file_name,'r')
    b_file = open(b_file_name,'r')

    
    A = []
    L = []
    for line in a_file:
        line, l = line.strip().split(";")
        L.append(float(l))
        a = torch.tensor(ast.literal_eval(line.strip())).unsqueeze(dim=0)
        A.append(a)
    A_final = torch.Tensor(len(A),A[0].shape[0])
    torch.cat(A, out=A_final,dim=0)
    
    
    B = []
    for line in b_file:
        line, l = line.strip().split(";")
        b = torch.tensor(ast.literal_eval(line.strip())).unsqueeze(dim=0)
        B.append(b)
    B_final = torch.Tensor(len(B),B[0].shape[0])
    torch.cat(B, out=B_final,dim=0)
        
    #Create feature fusion dataset
    X = torch.cat((A_final,B_final),dim=1)
    #p_final = torch.mul(p_final,10)
    X_prime = torch.mm(X, p_final.T)
    #X_prime = torch.mm(p_final, X.T)
    #X_prime = torch.mm(p_final, X.T)
    
    print()
    print("l2(p):",torch.linalg.norm(p_final))
    print(X_prime.shape)
    print()
    
    
    for i in range(X_prime.shape[0]):
        print(X_prime[i,0])
        #print(torch.linalg.norm(X_prime[i,:]))
    print()
    
    for i in range(X_prime.shape[0]):
        X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
    
    
    print(L)
    print(X_prime.shape)
    
    for i in range(X_prime.shape[0]):
        X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))

if __name__ == "__main__":
    rescale_p("data/approximate_best_P_value_transpose_lambda=0.1_margin=0.5_gamma=256_reg=0.txt",2.5)
    view_norms("data/approximate_best_P_value_transpose_lambda=0.1_margin=0.5_gamma=256_reg=0.txt","data/features_A_values_val.txt","data/features_B_values_val.txt")


