#import numpy as np
import random
import os

#import autograd.numpy as np  # Thinly-wrapped numpy
#from autograd import grad
#from matplotlib import pyplot
from sklearn.decomposition import PCA
import math

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from model import Linear_Feature_Fusion
from data_generation import data_gen

def main():
    random.seed(0)
    torch.manual_seed(0)
    
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/random_P"):
        os.mkdir("data/random_P")
    
    A,B,L = data_gen(10)
    alpha = A.shape[1]
    beta = B.shape[1]
    gamma = 2
    lambs = [0.1,0.25,0.5,0.75,0.9]
    margin = 0.5
    indims_list = [[128,128,128,128,128,128,128],[2,4,8,16,32,64,128]]
    outdims_list = [[1,2,4,8,16,32,64],[1,1,1,1,1,1,1]]
    for j in range(len(indims_list)):
        indims = indims_list[j]
        outdims = outdims_list[j]
        lamb = 0.5

        for i in range(len(indims)):
            indim = indims[i]
            outdim = outdims[i]
            
            print("Lambda value of", lamb)
            X = torch.cat((A,B),dim=1)
            M = [(1,1)]
            V = [(1,1)]
            model = Linear_Feature_Fusion(X,M,V,outdim,margin,lamb,indim)
            best_P = model.P

            p_best_file_name = "data/random_P/random_P_value_transpose_indim=" + str(indim) + "_outdim=" + str(outdim) + ".txt"
            outfile_p_t = open(p_best_file_name,'w')
            P_final_t = best_P.T
            P_final_t = str(P_final_t.tolist())
            outfile_p_t.write(P_final_t)
            outfile_p_t.close()

            print()

if __name__ == "__main__":
    main()
