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

def rescale_p(p_file_name, scale=None):
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
    l_file = open("data/features_L_values_val.txt",'r')
    
    A = []
    L = []
    for line in a_file:
        #line, l = line.strip().split(";")
        #L.append(float(l))
        line = line[1:-2]
        line = [float(i) for i in line.split()]
        a = torch.tensor(line).unsqueeze(dim=0)
        #a = torch.tensor(ast.literal_eval(line.strip())).unsqueeze(dim=0)
        A.append(a)
    A_final = torch.Tensor(len(A),A[0].shape[0])
    torch.cat(A, out=A_final,dim=0)
    print(A_final.shape)
    
    B = []
    for line in b_file:
        line = line[1:-2]
        line = [float(i) for i in line.split()]
        b = torch.tensor(line).unsqueeze(dim=0)
        #line, l = line.strip().split(";")
        #b = torch.tensor(ast.literal_eval(line.strip())).unsqueeze(dim=0)
        B.append(b)
    B_final = torch.Tensor(len(B),B[0].shape[0])
    torch.cat(B, out=B_final,dim=0)
        
    #L = l_file.readline()
    #L = [int(i) for i in L[1:len(L)-2].split(", ")]
    
    #Create feature fusion dataset
    X = torch.cat((A_final,B_final),dim=1)
    #p_final = torch.mul(p_final,10)
    #print(X.shape)
    #X_prime = torch.mm(X, p_final.T)
    X_prime = torch.mm(p_final,X.T)
    
    #print()
    #print("l2(p):",torch.linalg.norm(p_final))
    #print("P:",torch.linalg.norm(p_final))
    #print(p_final.shape)
    #print(p_final)
    
    #print("AB:",torch.linalg.norm(X[:,0]))
    #print(X.shape)
    #print(X)
    
    #print("X_prime:",torch.linalg.norm(X_prime))
    #print(X_prime.shape)
    #print(X_prime)
    #p_final = torch.div(p_final,torch.linalg.norm(p_final))
    #print(p_final)
    print()
    
    
    above = 0
    below = 0
    for i in range(X_prime.shape[1]):
        #print(X_prime[i,0])
        #print(X_prime[:,i].shape)
        print(torch.linalg.norm(X_prime[:,i]),torch.linalg.norm(X_prime[:,i])**2)
        if torch.linalg.norm(X_prime[:,i])**2 > 0.7:
            above += 1
        if torch.linalg.norm(X_prime[:,i])**2 < 0.1:
            below += 1
        
    print("above:",above)
    print("below:",below)
    print()
    
    #for i in range(X_prime.shape[0]):
        #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
    
    
    #print(L)
    #print(X_prime.shape)
    
    #for i in range(X_prime.shape[0]):
        #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))

if __name__ == "__main__":
    #rescale_p("data/approximate_best_P_value_transpose_lambda=0.1_margin=0.5_gamma=64_reg=0.txt",1.0)
    #view_norms("data/approximate_best_P_value_transpose_lambda=0.1_margin=0.5_gamma=64_reg=0.txt","data/features_A_values_val.txt","data/features_B_values_val.txt")
    #rescale_p("data/exact_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=64_reg=0.txt",5.25)
    #view_norms("data/exact_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=64_reg=0.txt","data/features_A_values_val.txt","data/features_B_values_val.txt")
    
    #rescale_p("data/approximate_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=64_reg=0.txt",4.75)
    #rescale_p("data/approximate_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=64_reg=0.txt",5.26)
    #view_norms("data/approximate_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=64_reg=0.txt","data/features_A_values_train.txt","data/features_B_values_train.txt")
    #view_norms("data/approximate_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=64_reg=0.txt","data/features_A_values_val.txt","data/features_B_values_val.txt")
    
    #rescale_p("data/1approximate_best_P_value_transpose_lambda=0.1_margin=0.75_gamma=64_reg=0.txt",5.0)
    #view_norms("data/1approximate_best_P_value_transpose_lambda=0.1_margin=0.75_gamma=64_reg=0.txt","data/features_A_values_val.txt","data/features_B_values_val.txt")
    #view_norms("data/2approximate_best_P_value_transpose_lambda=0.1_margin=0.75_gamma=64_reg=0.txt","data/features_A_values_val.txt","data/features_B_values_val.txt")
    #rescale_p("data/degree=3b_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt",4.25)
    #view_norms("data/degree=3b_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt","data/features_A_values_train.txt","data/features_B_values_train.txt")
    
    #rescale_p("data/degree=2strict_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt",4.5) #scale =3.5486826908067868
    #view_norms("data/degree=2strict_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt","data/features_A_values_val.txt","data/features_B_values_val.txt")
    #rescale_p("data/degree=3strict_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt",3.448984107119376) #scale
    #rescale_p("data/degree=3strict_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt",4.0) #scale
    #view_norms("data/degree=3strict_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt","data/features_A_values_val.txt","data/features_B_values_val.txt")
    
    #rescale_p("data/degree=1strict_approximate_best_P_value_transpose_lambda=0.5_margin=0.5_gamma=64_reg=0.txt",8.603522662597202) #10?
    #view_norms("data/degree=1strict_approximate_best_P_value_transpose_lambda=0.5_margin=0.5_gamma=64_reg=0.txt","data/features_A_values_val.txt","data/features_B_values_val.txt")

    #rescale_p("data/degree=2strict_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt",4.5)
    rescale_p("data/degree=2strict_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt",3.323253455743364)
    #3.323253455743364
    view_norms("data/degree=2strict_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt","data/features_A_values_test.txt","data/features_B_values_val.txt")
