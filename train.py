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

from ROC import ROC_AUC

def train(gamma,iters,spec_margin=None,spec_lamb=None):
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
    print()
    print()
    print()
    
    t1 = torch.tensor([[1.,2.],[2.,4.]])
    t2 = torch.tensor([[56.,50.],[30.,32.]])
    t3 = torch.mm(t1,t2)
    
    print(torch.div(t3,torch.linalg.norm(t3)))
    
    t1 = torch.tensor([[1.,2.],[2.,4.]])
    t1 = torch.div(t1,torch.linalg.norm(t1))
    t2 = torch.tensor([[56.,50.],[30.,32.]])
    t3 = torch.mm(t1,t2)
    
    print(torch.div(t3,torch.linalg.norm(t3)))
    #print(torch.linalg.norm(t3))
    
    0/0
    """
    
    """
    
    A = np.load("data/features/MMU_probe_resnet.npy")
    A = np.reshape(A,(A.shape[0],A.shape[2]))
    A = torch.tensor(A)
    #A = torch.ones((A.shape))
    print(A)
    print("A:",A.shape)
    
    
    B = np.load("data/features/MMU_probe_vgg.npy")
    B = np.reshape(B,(B.shape[0],B.shape[2]))
    B = torch.tensor(B)
    #B = torch.ones((B.shape))
    print("B:",B.shape)
    
    
    L = np.load("data/features/MMU_label_probe.npy")
    #L = np.reshape(B,(B.shape[0],B.shape[2]))
    L = torch.tensor(L)
    L = L[:,0]
    print(L)
    print("L:",L.shape)
    """
    
    
    X = torch.cat((A,B),dim=1)
    
    num_samples = L.shape[0]
    num_each_class = 8
    num_classes = 45
    num_each_class = 2
    
    split1 = math.floor(num_classes * 0.6)*num_each_class
    split2 = math.floor(num_classes * 0.8)*num_each_class
    
    #print(split1,split2)
    
    
    X_train = X[:split1,:]
    X_val = X[split1:split2,:]
    X_test = X[split2:,:]
    
    L_train = L[:split1]
    L_val = L[split1:split2]
    print("labels, val:",L_val)
    L_test = L[split1:]
    
    
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
    
    
    outfile_a_test = open("data/features_A_values_test.txt",'w')
    for i in range(70,90):
        outfile_a_test.write(str(A[i].tolist()))
        outfile_a_test.write(";")
        outfile_a_test.write(str(L.tolist()[i]))
        outfile_a_test.write("\n")
    outfile_a_test.close()


    outfile_b_test = open("data/features_B_values_test.txt",'w')
    for i in range(70,90):
        outfile_b_test.write(str(B[i].tolist()))
        outfile_b_test.write(";")
        outfile_b_test.write(str(L.tolist()[i]))
        outfile_b_test.write("\n")
    outfile_b_test.close()
    
    outfile_x_test = open("data/features_X_values_test.txt",'w')
    for i in range(70,90):
        outfile_x_test.write(str(X[i].tolist()))
        outfile_x_test.write(";")
        outfile_x_test.write(str(L.tolist()[i]))
        outfile_x_test.write("\n")
    outfile_x_test.close()
    
    #alpha = A.shape[1]
    #beta = B.shape[1]

    
    
    outfile_X = open("data/features_X_values.txt",'w')
    for i in range(B.shape[0]):
        outfile_X.write(str(X[i].tolist()))
        outfile_X.write(";")
        outfile_X.write(str(L.tolist()[i]))
        outfile_X.write("\n")
    outfile_X.close()
    
    #Hyperparameters
    #gamma = 512
    lambs = [0.1,0.25,0.5,0.75,0.99]
    margins = [0.0,0.25,0.5]
    margins = [0.25,0.5]
    margins = [0.25]
    lambs = [0.1]
    #margin = 0.5
    iterations = iters
    #regularizers = [1,0.1,0.001,0.0001,0.00001,0.000001,0.0000001]
    #regularizers = [0.1]
    #regularizers = [1,10,100]
    #regularizers = [1000,10000,100000]
    regularizers = [0]
    #regularizers = [0.0001,0.1,10]
    #iterations = 100
    #break_point = 0.005
    rates = {2:1000, 128: 10000, 256:10000, 512:10000}
    
    if spec_margin:
        margins = [spec_margin]
    if spec_lamb:
        lambs = [spec_lamb]
    
    
    #lamb = 0.2
    aucs = []
    for reg in regularizers:
        for margin in margins:
            for lamb in lambs:
                rate = rates[gamma]
                print("Lambda value of", lamb, "Margin value of", margin, "regularizer of",reg)
                
                M = []
                V = []
                for i in range(X_train.shape[0]):
                    for j in range(i+1,X_train.shape[0]):
                        if L_train[i] == L_train[j]:
                            M.append((i,j))
                            for k in range(X_train.shape[0]):
                                if L_train[k] != L_train[i]:
                                    V.append((i,j,k))
                print("Same class",len(M))
                print("Trios:",len(V))
                
                model = Linear_Feature_Fusion(X_train,M,V,gamma,margin,lamb,regularization=reg)
                
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
                    
                    if i%10 == 0:
                        print("Iteration",str(i) + "/" + str(iterations))
                        print(model.loss())
                    rate *= 0.95
                print("final loss:",model.loss())
                print("old best p norm:",torch.linalg.norm(best_P))
                best_P = torch.div(best_P,torch.linalg.norm(best_P))
                print("new best p norm:",torch.linalg.norm(best_P))
                X_prime = torch.mm(X_val,best_P)
                print("new NOT normalized validation X_prime:", X_prime)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                auc = ROC_AUC(X_prime, L_val)
                aucs.append((margin,lamb,auc))
                print("AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                #P values file gets too large for github
                #p_file_name = "data/features_P_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + ".txt"
                #outfile_p = open(p_file_name,'w')
                #for P_value in P_history:
                    #outfile_p.write(P_value)
                    #outfile_p.write("\n")
                #outfile_p.close()
                
                p_best_file_name = "data/features_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_p_t = open(p_best_file_name,'w')
                P_final_t = best_P.T
                P_final_t = str(P_final_t.tolist())
                outfile_p_t.write(P_final_t)
                outfile_p_t.close()
                
                loss_file_name = "data/features_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_loss = open(loss_file_name,'w')
                for loss_value in losses:
                    outfile_loss.write(str(loss_value.tolist()))
                    outfile_loss.write("\n")
                outfile_loss.close()
                print()
                
                #X_prime = torch.mm(X_val,P_history_matrices[-1])
                #X_prime = torch.mm(X_val,best_P)
                #for i in range(X_prime.shape[0]):
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                print("new normalized validation X_prime:", X_prime)
                
                print("norm of best P:",torch.linalg.norm(best_P))
                
                X_prime_filename = "data/features_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_prime_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_val.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                #x_str = str(x_list)
                #outfile_x.write(x_str)
                outfile_x.close()
                
                #X_prime = torch.mm(X_val,P_history_matrices[-1])
                X_prime = torch.mm(X_test,best_P)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                print("new normalized test x_prime:", X_prime)
                
                X_prime_filename = "data/features_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_prime_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
    print("(margin, lambda, Validation AUC)")
    print(aucs)
if __name__ == "__main__":
    #train(256,200)#for gamma=128, use margin=0.25,lamb=0.1)
    train(128,200)#for gamma=128, use margin=0.25,lamb=0.1)


