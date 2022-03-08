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
from model import Linear_Feature_Fusion
from data_generation import data_gen

def train(gamma,iters):
    random.seed(0)
    torch.manual_seed(0)
    #A,B,L = data_gen(10)
    
    A = np.load("data/features/MMU_gallery_resnet.npy")
    A = np.reshape(A,(A.shape[0],A.shape[2]))
    A = torch.tensor(A)
    #A = torch.ones((A.shape))
    print(A)
    print(A.shape)
    
    B = np.load("data/features/MMU_gallery_vgg.npy")
    B = np.reshape(B,(B.shape[0],B.shape[2]))
    B = torch.tensor(B)
    #B = torch.ones((B.shape))
    print(B.shape)
    
    
    L = np.load("data/features/MMU_label_gallery.npy")
    #L = np.reshape(B,(B.shape[0],B.shape[2]))
    L = torch.tensor(L)
    L = L[:,0]
    print(L)
    print(L.shape)

    
    """
    for j in range(0,len(L),2):
        total= 0
        ls1 = A[j].tolist()
        ls2 = A[j+1].tolist()
        for i in range(len(ls1)):
            total += abs(ls1[i]-ls2[i])
        print("total diff:", total)
"""
    outfile_a = open("data/features_A_values.txt",'w')
    for i in range(A.shape[0]):
        outfile_a.write(str(A[i].tolist()))
        outfile_a.write(";")
        outfile_a.write(str(L.tolist()[i]))
        outfile_a.write("\n")
    outfile_a.close()


    outfile_b = open("data/features_B_values.txt",'w')
    for i in range(B.shape[0]):
        outfile_b.write(str(B[i].tolist()))
        outfile_b.write(";")
        outfile_b.write(str(L.tolist()[i]))
        outfile_b.write("\n")
    outfile_b.close()
    
    #alpha = A.shape[1]
    #beta = B.shape[1]
    X = torch.cat((A,B),dim=1)
    
    
    outfile_X = open("data/features_X_values.txt",'w')
    for i in range(B.shape[0]):
        outfile_X.write(str(X[i].tolist()))
        outfile_X.write(";")
        outfile_X.write(str(L.tolist()[i]))
        outfile_X.write("\n")
    outfile_X.close()
    
    #Hyperparameters
    #gamma = 512
    lambs = [0.1,0.25,0.5,0.75,0.9]
    lambs = [0.5]
    margin = 0.5
    iterations = iters
    #iterations = 100
    #break_point = 0.005
    rates = {2:1000, 256:10000, 512:10000}
    rate = 10e-2
    rate = 10000
    rate = 1000
    rate = rates[gamma]
    
    
    
    #lamb = 0.2
    for lamb in lambs:
        print("Lambda value of", lamb)
        
        M = []
        V = []
        for i in range(X.shape[0]):
            for j in range(i+1,X.shape[0]):
                if L[i] == L[j]:
                    M.append((i,j))
                    
                    """print("matching class")
                    total= 0
                    ls1 = torch.nn.functional.normalize(X[i],dim=0).tolist()
                    ls2 = torch.nn.functional.normalize(X[j],dim=0).tolist()
                    for c in range(len(ls1)):
                        total += ls1[c]*ls2[c]
                    print("avg diff:", (1-total)/len(ls1))
                    print("total diff:", 1-total)
                    print()"""
                    
                    
                    for k in range(X.shape[0]):
                        if L[k] != L[i]:
                            V.append((i,j,k))
                else:
                    """print("different class")
                    total= 0
                    ls1 = torch.nn.functional.normalize(X[i],dim=0).tolist()
                    ls2 = torch.nn.functional.normalize(X[j],dim=0).tolist()
                    for c in range(len(ls1)):
                        total += ls1[c]*ls2[c]
                    print("avg diff:", (1-total)/len(ls1))
                    print("total diff:", 1-total)
                    print()"""
        print("Same class",len(M))
        print("Trios:",len(V))
        
        model = Linear_Feature_Fusion(X,M,V,gamma,margin,lamb)
        

    
        best_loss = model.loss()
        best_P = model.P
        print("Initial loss:",best_loss)
        P_history = []
        P_history_matrices = []
        losses = []
        optim = torch.optim.SGD(model.parameters(), lr=rate)#, momentum=0.9)
        for i in range(iterations):
            loss = model.loss()
            if loss < best_loss:
                best_loss = loss
                best_P = model.P
            losses.append(loss)
            P_history.append(str(model.P.tolist()))
            P_history_matrices.append(model.P)
            loss.backward()
            optim.step()
            print(model.loss())
            if i%10 == 0:
                print("Iteration",str(i) + "/" + str(iterations))
            rate *= 0.95
        print("final loss:",model.loss())
        

        #P values file gets too large for github
        p_file_name = "data/features_P_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + ".txt"
        outfile_p = open(p_file_name,'w')
        #for P_value in P_history:
            #outfile_p.write(P_value)
            #outfile_p.write("\n")
        outfile_p.close()
        
        p_best_file_name = "data/features_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + ".txt"
        outfile_p_t = open(p_best_file_name,'w')
        P_final_t = best_P.T
        P_final_t = str(P_final_t.tolist())
        outfile_p_t.write(P_final_t)
        outfile_p_t.close()
        
        loss_file_name = "data/features_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + ".txt"
        outfile_loss = open(loss_file_name,'w')
        for loss_value in losses:
            outfile_loss.write(str(loss_value.tolist()))
            outfile_loss.write("\n")
        outfile_loss.close()
        print()
        
        X_prime = torch.mm(X,P_history_matrices[-1])
        for i in range(X_prime.shape[0]):
            X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
        print("new features:", X_prime)
        
        X_prime_filename = "data/features_labels_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + ".txt"
        outfile_x = open(X_prime_filename,'w')
        x_list = X_prime.tolist()
        for i in range(len(x_list)):
            x_list[i] = str(x_list[i])+";"+str(L.tolist()[i])
            outfile_x.write(x_list[i])
            outfile_x.write("\n")
        #x_str = str(x_list)
        #outfile_x.write(x_str)
        outfile_x.close()

if __name__ == "__main__":
    train(2,200)


