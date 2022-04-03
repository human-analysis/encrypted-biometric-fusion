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
from model import Linear_Feature_Fusion_Approximate, Linear_Feature_Fusion, Linear_Feature_Fusion_No_Normal, Linear_Feature_Fusion_Approximate2, Linear_Feature_Fusion_Approximate3
from data_generation import data_gen

from ROC import New_ROC_AUC



def approximate_inv_norm(x_in):
    coeffs = [[3.6604110068015703, -7.308745554603273, 5.359140241417692, -0.03216663533709177], [-1.761181767348659, 5.619133141454438, -7.496635998204148, 5.491355198579896]]
    
    coeffs = [[-14.87368246,23.74576715,-13.66592657,4.17688396]]
    
    #[ 3.54940138 -7.3475699   5.91965872]
    coeffs = [[5.91965872,-7.3475699,3.54940138]]
    
    #coeffs = [[-2.61776258,2.78221164]]
    
    #coeffs = [[11.836520387699572, -18.076619596914263, 9.213047940260486, -0.1390999565263271], [-5.035385227584069, 14.361565311498836, -14.664452287760135, 7.742745833744212]]
    
    x = torch.linalg.norm(x_in)**2
    
    #print("Squared norm:",x**0.5)
    result = 0
    for coeff_list in coeffs:
        result = coeff_list[0]
        for i in range(1,len(coeff_list)):
            result = result * x + coeff_list[i]
        x = result
        result = 0
        
    #print("approx:",x,"truth:",1/torch.linalg.norm(x_in))
    #print()
    return x



def train_exact(gamma,iters,spec_margin=None,spec_lamb=None):
    #random.seed(1)
    torch.manual_seed(0)
    #A,B,L = data_gen(10)
    
    a = []
    A_infile = open("feature-extraction/extractions/VGG16_lfw-deepfunneled.txt",'r')
    for line in A_infile:
        line = line.strip().split()
        a.append(torch.tensor([float(char) for char in line]))
    A = torch.stack(a)
    #print(A)
    print("A:",A.shape)
    
    
    B = np.load("data/features/MMU_gallery_vgg.npy")
    B = np.reshape(B,(B.shape[0],B.shape[2]))
    B = torch.tensor(B)
    #B = torch.ones((B.shape))
    print("B:",B.shape)
    
    for i in range(B.shape[0]):
        #print(torch.linalg.norm(torch.subtract(B[i,:],torch.div(B[i,:],torch.linalg.norm(B[i,:])))))
        #print(torch.linalg.norm(B[i,:]))
        B[i,:] = torch.div(B[i,:],torch.linalg.norm(B[i,:]))
    
    
    L = np.load("data/features/MMU_label_gallery.npy")
    #L = np.reshape(B,(B.shape[0],B.shape[2]))
    L = torch.tensor(L)
    L = L[:,0]
    print(L)
    print(L.shape)
    
    X = torch.cat((A,B),dim=1)
    
    #num_samples = L.shape[0]
    #num_each_class = 8
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
    
    

    


    outfile_a = open("data/features_A_values.txt",'w')
    for row in A.tolist():
        for item in row:
            outfile_a.write(str(f'{item:.9f}'))
            outfile_a.write(" ")
        outfile_a.write("\n")
    outfile_a.close()

    outfile_b = open("data/features_B_values.txt",'w')
    for i in range(B.shape[0]):
        outfile_b.write(str(B[i].tolist()))
        outfile_b.write(";")
        outfile_b.write(str(L.tolist()[i]))
        outfile_b.write("\n")
    outfile_b.close()
    
    
    outfile_a_train = open("data/features_A_values_train.txt",'w')
    #outfile_a_test.write("[")
    for row in A.tolist()[:split1]:
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_train.write(str(f'{item:.9f}'))
            outfile_a_train.write(" ")
        #outfile_a_test.write("]")
        outfile_a_train.write("\n")
    outfile_a_train.close()

    
    outfile_b_train = open("data/features_B_values_train.txt",'w')
    for row in B.tolist()[:split1]:
        for item in row:
            outfile_b_train.write(str(f'{item:.9f}'))
            outfile_b_train.write(" ")
        outfile_b_train.write("\n")
    outfile_b_train.close()

    outfile_a_test = open("data/features_A_values_test.txt",'w')
    #outfile_a_test.write("[")
    for row in A.tolist()[split2:]:
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_test.write(str(f'{item:.9f}'))
            outfile_a_test.write(" ")
        #outfile_a_test.write("]")
        outfile_a_test.write("\n")
    outfile_a_test.close()

    
    outfile_b_test = open("data/features_B_values_test.txt",'w')
    for row in B.tolist()[split2:]:
        for item in row:
            outfile_b_test.write(str(f'{item:.9f}'))
            outfile_b_test.write(" ")
        outfile_b_test.write("\n")
    outfile_b_test.close()
    

    
    outfile_a_val = open("data/features_A_values_val.txt",'w')
    for row in A.tolist()[split1:split2]:
        outfile_a_val.write("[")
        for item in row:
            outfile_a_val.write(str(f'{item:.9f}'))
            outfile_a_val.write(" ")
        outfile_a_val.write("]")
        outfile_a_val.write("\n")
    outfile_a_val.close()
    
    
    outfile_b_val = open("data/features_B_values_val.txt",'w')
    for row in B.tolist()[split1:split2]:
        outfile_b_val.write("[")
        for item in row:
            outfile_b_val.write(str(f'{item:.9f}'))
            outfile_b_val.write(" ")
        outfile_b_val.write("]")
        outfile_b_val.write("\n")
    outfile_b_val.close()
    

    
    outfile_x_test = open("data/features_X_values_test.txt",'w')
    for row in X.tolist()[split2:]:
        outfile_x_test.write("[")
        for item in row:
            outfile_x_test.write(str(f'{item:.9f}'))
            outfile_x_test.write(" ")
        outfile_x_test.write("]")
        outfile_x_test.write("\n")
    outfile_x_test.close()
    
    outfile_L = open("data/features_L_values_val.txt",'w')
    outfile_L.write(str(L_val.tolist()))
    outfile_L.close()
    
    outfile_L = open("data/features_L_values_test.txt",'w')
    outfile_L.write(str(L_test.tolist()))
    outfile_L.close()
    
    #Hyperparameters
    #gamma = 512
    lambs = [0.01,0.1,0.25,0.5,0.75,0.99]
    #lambs = [0.01,0.1,0.25,0.5]
    #lambs = [0.1,0.25,0.5]
    #lambs = [0.5]
    #margins = [0.0,0.25,0.5]
    margins = [0.25,0.5,0.75,1.0]
    #lambs = [0.25]
    #margins = [0.1]
    #margins = [1.0]
    #margins = [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]
    #margins = [0.5,0.75,1.0]
    #lambs = [0.1]
    iterations = iters

    regularizers = [0]

    rates = {2:1000, 32:500, 64:10000, 128: 10000, 256:10000, 512:10000}
    #rates[64] = 500
    rates[64] = 20000
    anneal_rate = 0.995
    anneal_rate = 0.9999
    #anneal_rate = 1
    #anneal_rate = 0.9
    #anneal_rate = 0.90
    if spec_margin:
        margins = [spec_margin]
    if spec_lamb:
        lambs = [spec_lamb]
    
    
    f = 0
    randoms = [i*5 for i in range(21,31)]
    
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
                #randie = randoms[f]
                #f+=1
                #randie = 24
                randie = 30
                randie = 0
                print("seed of:",randie)
                model = Linear_Feature_Fusion(X_train,M,V,gamma,margin,lamb,regularization=reg,seed=randie)
                #model = Linear_Feature_Fusion(X_train,M,V,gamma,margin,lamb,regularization=reg)
                
                best_loss = model.loss()
                best_P = model.P
                best_scale = model.scale
                print("Initial loss:",best_loss)
                #P_history = []
                #P_history_matrices = []
                losses = []
                optimizer = torch.optim.SGD(model.parameters(), lr=rate)#, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=anneal_rate)
                for i in range(iterations):
                    

                    loss = model.loss()
                    #print("Loss:",loss)
                    if loss <= best_loss:#+0.003
                        best_loss = loss
                        best_P = model.P
                        best_scale = model.scale

                    losses.append(loss)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    #model.P = torch.mul(torch.div(model.P,torch.linalg.norm(model.P)),3.0)
                    if i%10 == 0:
                        print("Iteration",str(i) + "/" + str(iterations))
                        print(model.loss())

                best_P = torch.div(best_P,torch.linalg.norm(best_P))

                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)

                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                #auc = New_ROC_AUC(X_prime, L_val)
                #print("EXACTED AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                
                #X_prime = torch.mm(X_val,best_P)
                #print(X_prime.shape)

                
                auc = New_ROC_AUC(X_prime, L_val)
                aucs.append((margin,lamb,auc))
                print("AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)

                p_best_file_name = "data/exact_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #p_best_file_name = "data/exact_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_p_t = open(p_best_file_name,'w')
                P_final_t = best_P.T
                P_final_t = str(P_final_t.tolist())
                outfile_p_t.write(P_final_t)
                outfile_p_t.close()
                
                loss_file_name = "data/exact_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #loss_file_name = "data/exact_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_loss = open(loss_file_name,'w')
                for loss_value in losses:
                    outfile_loss.write(str(loss_value.tolist()))
                    outfile_loss.write("\n")
                outfile_loss.close()

                X_prime_filename = "data/exact_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
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
                
                
                X_filename = "data/exact_labels_X_NOT_NORMAL_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
                
                
                for i in range(X_prime.shape[0]):
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                #print("new normalized test x_prime:", X_prime)
                
                X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_prime_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
    print("(margin, lambda, Validation AUC)")
    print(aucs)

def train(gamma,iters,spec_margin=None,spec_lamb=None):
    #random.seed(1)
    torch.manual_seed(0)
    #A,B,L = data_gen(10)
    
    a = []
    A_infile = open("feature-extraction/extractions/VGG16_lfw-deepfunneled.txt",'r')
    for line in A_infile:
        line = line.strip().split()
        a.append(torch.tensor([float(char) for char in line]))
    A = torch.stack(a)
    #print(A)
    print("A:",A.shape)
    
    
    B = np.load("data/features/MMU_gallery_vgg.npy")
    B = np.reshape(B,(B.shape[0],B.shape[2]))
    B = torch.tensor(B)
    #B = torch.ones((B.shape))
    print("B:",B.shape)
    
    C = np.load("data/features/MMU2_gallery_vgg.npy")
    C = np.reshape(C,(C.shape[0],C.shape[2]))
    C = torch.tensor(C)
    #B = torch.ones((B.shape))
    print("C:",C.shape)
    
    for i in range(B.shape[0]):
        #print(torch.linalg.norm(torch.subtract(B[i,:],torch.div(B[i,:],torch.linalg.norm(B[i,:])))))
        #print(torch.linalg.norm(B[i,:]))
        B[i,:] = torch.div(B[i,:],torch.linalg.norm(B[i,:]))
    
    
    L = np.load("data/features/MMU_label_gallery.npy")
    #L = np.reshape(B,(B.shape[0],B.shape[2]))
    L = torch.tensor(L)
    L = L[:,0]
    print(L)
    print(L.shape)
    
    X = torch.cat((A,B),dim=1)
    
    #num_samples = L.shape[0]
    #num_each_class = 8
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
    
    

    


    outfile_a = open("data/features_A_values.txt",'w')
    for row in A.tolist():
        for item in row:
            outfile_a.write(str(f'{item:.9f}'))
            outfile_a.write(" ")
        outfile_a.write("\n")
    outfile_a.close()

    outfile_b = open("data/features_B_values.txt",'w')
    for i in range(B.shape[0]):
        outfile_b.write(str(B[i].tolist()))
        outfile_b.write(";")
        outfile_b.write(str(L.tolist()[i]))
        outfile_b.write("\n")
    outfile_b.close()
    
    
    outfile_a_train = open("data/features_A_values_train.txt",'w')
    #outfile_a_test.write("[")
    for row in A.tolist()[:split1]:
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_train.write(str(f'{item:.9f}'))
            outfile_a_train.write(" ")
        #outfile_a_test.write("]")
        outfile_a_train.write("\n")
    outfile_a_train.close()

    
    outfile_b_train = open("data/features_B_values_train.txt",'w')
    for row in B.tolist()[:split1]:
        for item in row:
            outfile_b_train.write(str(f'{item:.9f}'))
            outfile_b_train.write(" ")
        outfile_b_train.write("\n")
    outfile_b_train.close()

    outfile_a_test = open("data/features_A_values_test.txt",'w')
    #outfile_a_test.write("[")
    for row in A.tolist()[split2:]:
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_test.write(str(f'{item:.9f}'))
            outfile_a_test.write(" ")
        #outfile_a_test.write("]")
        outfile_a_test.write("\n")
    outfile_a_test.close()

    
    outfile_b_test = open("data/features_B_values_test.txt",'w')
    for row in B.tolist()[split2:]:
        for item in row:
            outfile_b_test.write(str(f'{item:.9f}'))
            outfile_b_test.write(" ")
        outfile_b_test.write("\n")
    outfile_b_test.close()
    

    
    outfile_a_val = open("data/features_A_values_val.txt",'w')
    for row in A.tolist()[split1:split2]:
        outfile_a_val.write("[")
        for item in row:
            outfile_a_val.write(str(f'{item:.9f}'))
            outfile_a_val.write(" ")
        outfile_a_val.write("]")
        outfile_a_val.write("\n")
    outfile_a_val.close()
    
    
    outfile_b_val = open("data/features_B_values_val.txt",'w')
    for row in B.tolist()[split1:split2]:
        outfile_b_val.write("[")
        for item in row:
            outfile_b_val.write(str(f'{item:.9f}'))
            outfile_b_val.write(" ")
        outfile_b_val.write("]")
        outfile_b_val.write("\n")
    outfile_b_val.close()
    

    
    outfile_x_test = open("data/features_X_values_test.txt",'w')
    for row in X.tolist()[split2:]:
        outfile_x_test.write("[")
        for item in row:
            outfile_x_test.write(str(f'{item:.9f}'))
            outfile_x_test.write(" ")
        outfile_x_test.write("]")
        outfile_x_test.write("\n")
    outfile_x_test.close()
    
    outfile_L = open("data/features_L_values_val.txt",'w')
    outfile_L.write(str(L_val.tolist()))
    outfile_L.close()
    
    outfile_L = open("data/features_L_values_test.txt",'w')
    outfile_L.write(str(L_test.tolist()))
    outfile_L.close()
    
    #Hyperparameters
    #gamma = 512
    lambs = [0.01,0.1,0.25,0.5,0.75,0.99]
    lambs = [0.01,0.1,0.25,0.5]
    #lambs = [0.01,0.1,0.25,0.5]
    #lambs = [0.1,0.25,0.5]
    #lambs = [0.1,0.1,0.1,0.1,0.1]
    #lambs = [0.01]
    #margins = [0.0,0.25,0.5]
    margins = [0.1,0.25,0.5,0.75,1.0]
    #lambs = [0.01]
    #margins = [0.1]
    #margins = [0.1]
    #margins = [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]
    #margins = [0.5,0.75,1.0]
    #lambs = [0.1]
    
    lambs = [0.1]
    margins = [0.1]
    
    iterations = iters

    regularizers = [0]

    rates = {2:1000, 32:500, 64:10000, 128: 10000, 256:10000, 512:10000}
    #rates[64] = 500
    rates[64] = 20000
    
    rates[64] = 500
    
    #rates[64] = 15000
    #rates[64] = 40000
    #rates[64] = 10000
    #anneal_rate = 0.995
    anneal_rate = 0.9999
    #anneal_rate = 1
    #anneal_rate = 0.95
    #anneal_rate = 1
    #anneal_rate = 0.9
    #anneal_rate = 0.90
    if spec_margin:
        margins = [spec_margin]
    if spec_lamb:
        lambs = [spec_lamb]
    
    #anneal_rates = [1,0.9999,0.995,0.95,0.9]
    f = 0
    randoms = [i*5 for i in range(21,31)]
    special_counter = 0
    #lamb = 0.2
    aucs = []
    for reg in regularizers:
        for margin in margins:
            for lamb in lambs:
                rate = rates[gamma]
                #anneal_rate = anneal_rates[special_counter]
                #special_counter += 1
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
                #randie = randoms[f]
                #f+=1
                #randie = 24
                randie = 30
                randie = 1
                randie = 0
                #randie = 2
                print("seed of:",randie)
                print("rate, anneal_rate:", rate, anneal_rate)
                model = Linear_Feature_Fusion_Approximate2(X_train,M,V,gamma,margin,lamb,regularization=reg,seed=randie)
                #model = Linear_Feature_Fusion(X_train,M,V,gamma,margin,lamb,regularization=reg)
                
                best_loss = model.loss()
                best_P = model.P
                best_scale = model.scale
                print("Initial loss:",best_loss)
                #P_history = []
                #P_history_matrices = []
                losses = []
                optimizer = torch.optim.SGD(model.parameters(), lr=rate)#, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=anneal_rate)
                for i in range(iterations):
                    
                    #first we select a scale such that our polynomial function will work
                    total = 0.0
                    P_temp = torch.div(model.P,torch.linalg.norm(model.P))
                    min_norm = 10000
                    for c in range(X_train.shape[0]):
                        norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                        total += norm
                        if norm < min_norm:
                            min_norm = norm
                    min_norm = float(min_norm)
                    total = float(total)
                    #print("total",total)
                    avg = total / X_train.shape[0]
                    
                    #print("average:",avg)
                    model.scale = 0.63245553 / avg #0.4 is the median of our valid range, therefore we want squared norm to be 0.4, therefore we use the sqrt of 0.4 as the numerator to make the new norm that
                    
                    #P_temp = torch.mul(P_temp,model.scale)
                    #for c in range(X_train.shape[0]):
                        #norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                        #print(norm**2)
                        
                    
                    #model.scale = 0.59581876 / avg #0.355 is the median of our valid range, therefore we want squared norm to be 0.4, therefore we use the sqrt of 0.4 as the numerator to make the new norm that
                    
                    #model.scale = 0.316227766 / min_norm #0.1 is the lowest of our valid range, therefore we can scale everything up to within our range
                    #model.scale = 0.70710678118 / avg
                    #print("new scale:",model.scale)
                    #P_temp = torch.mul(P_temp,model.scale)
                    #for c in range(X_train.shape[0]):
                        #print(torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))**2)
                    
                    
                    #model.P = torch.div(model.P, torch.linalg.norm(model.P))
                    #model.P = torch.mul(model.P, model.scale)
                    
                    loss = model.loss()
                    #print("Loss:",loss)
                    if loss <= best_loss:#+0.003
                        best_loss = loss
                        best_P = model.P
                        best_scale = model.scale
                    #else:
                        #print("Loss increased, ending training")
                        #break
                    losses.append(loss)
                    #P_history.append(str(model.P.tolist()))
                    #P_history_matrices.append(model.P)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    #model.P = torch.mul(torch.div(model.P,torch.linalg.norm(model.P)),3.0)
                    if i%10 == 0:
                        print("Iteration",str(i) + "/" + str(iterations))
                        print(loss)
                    #rate *= anneal_rate
                #print("final loss:",model.loss())
                #print("old best p norm:",torch.linalg.norm(best_P))
                best_P = torch.div(best_P,torch.linalg.norm(best_P))
                
                print("percentage of escape:",model.escape/model.tote)
                
                #calculate scale for best P
                total = 0.0
                min_norm = 10000
                for c in range(X_train.shape[0]):
                    norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                    total += norm
                    if norm < min_norm:
                        min_norm = norm
                min_norm = float(min_norm)
                total = float(total)
                avg = total / X_train.shape[0]
                best_scale = 0.63245553 / avg
                #best_scale = 0.59581876 / avg
                #best_scale = 0.316227766 / min_norm
                #best_scale = 0.70710678118 / avg
                
                best_P = torch.mul(best_P, best_scale)
                
                
                
                
                
                
                #for i in range(best_P.shape[1]):
                    #best_P[:,i]=torch.div(best_P[:,i], torch.linalg.norm(best_P[:,1]))
                #print("new best p norm:",torch.linalg.norm(best_P))
                
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)
                #print("new NOT normalized validation X_prime:", X_prime)
                #print("one vec:",X_prime[0,:].shape)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                auc = New_ROC_AUC(X_prime, L_val)
                print("EXACTED AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)
                #print("new NOT normalized validation X_prime:", X_prime)
                #print("one vec:",X_prime[0,:].shape)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                    #print("new norm:",torch.linalg.norm(X_prime[i,:]))
                    #print(X_prime[i,:])
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                    #print(X_prime[i,:])
                    #print()
                
                auc = New_ROC_AUC(X_prime, L_val)
                aucs.append((margin,lamb,auc,best_scale))
                print("AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                #P values file gets too large for github
                #p_file_name = "data/features_P_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + ".txt"
                #outfile_p = open(p_file_name,'w')
                #for P_value in P_history:
                    #outfile_p.write(P_value)
                    #outfile_p.write("\n")
                #outfile_p.close()
                
                p_best_file_name = "data/degree=2strict_approximate_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #p_best_file_name = "data/exact_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_p_t = open(p_best_file_name,'w')
                P_final_t = best_P.T
                P_final_t = str(P_final_t.tolist())
                outfile_p_t.write(P_final_t)
                outfile_p_t.close()
                
                loss_file_name = "data/degree=2strict_approximate_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #loss_file_name = "data/exact_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_loss = open(loss_file_name,'w')
                for loss_value in losses:
                    outfile_loss.write(str(loss_value.tolist()))
                    outfile_loss.write("\n")
                outfile_loss.close()
                #print()
                
                #X_prime = torch.mm(X_val,P_history_matrices[-1])
                #X_prime = torch.mm(X_val,best_P)
                #for i in range(X_prime.shape[0]):
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                #print("new normalized validation X_prime:", X_prime)
                
                #print("norm of best P:",torch.linalg.norm(best_P))
                
                X_prime_filename = "data/degree=2strict_approximate_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
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
                
                
                X_filename = "data/degree=2strict_approximate_labels_X_NOT_NORMAL_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
                
                
                for i in range(X_prime.shape[0]):
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                #print("new normalized test x_prime:", X_prime)
                
                X_prime_filename = "data/degree=2strict_approximate_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_prime_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
    print("(margin, lambda, Validation AUC), Scale")
    print(aucs)
    
    
    
    
def train_snap(gamma,iters,spec_margin=None,spec_lamb=None):
    #random.seed(1)
    torch.manual_seed(0)
    #A,B,L = data_gen(10)
    
    a = []
    A_infile = open("feature-extraction/extractions/VGG16_lfw-deepfunneled.txt",'r')
    for line in A_infile:
        line = line.strip().split()
        a.append(torch.tensor([float(char) for char in line]))
    A = torch.stack(a)
    #print(A)
    print("A:",A.shape)
    
    
    B = np.load("data/features/MMU_gallery_vgg.npy")
    B = np.reshape(B,(B.shape[0],B.shape[2]))
    B = torch.tensor(B)
    #B = torch.ones((B.shape))
    print("B:",B.shape)
    
    C = np.load("data/features/MMU2_gallery_vgg.npy")
    C = np.reshape(C,(C.shape[0],C.shape[2]))
    C = torch.tensor(C)
    #B = torch.ones((B.shape))
    print("C:",C.shape)
    
    for i in range(B.shape[0]):
        #print(torch.linalg.norm(torch.subtract(B[i,:],torch.div(B[i,:],torch.linalg.norm(B[i,:])))))
        #print(torch.linalg.norm(B[i,:]))
        B[i,:] = torch.div(B[i,:],torch.linalg.norm(B[i,:]))
    
    
    L = np.load("data/features/MMU_label_gallery.npy")
    #L = np.reshape(B,(B.shape[0],B.shape[2]))
    L = torch.tensor(L)
    L = L[:,0]
    print(L)
    print(L.shape)
    
    X = torch.cat((A,B),dim=1)
    
    #num_samples = L.shape[0]
    #num_each_class = 8
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
    
    

    


    outfile_a = open("data/features_A_values.txt",'w')
    for row in A.tolist():
        for item in row:
            outfile_a.write(str(f'{item:.9f}'))
            outfile_a.write(" ")
        outfile_a.write("\n")
    outfile_a.close()

    outfile_b = open("data/features_B_values.txt",'w')
    for i in range(B.shape[0]):
        outfile_b.write(str(B[i].tolist()))
        outfile_b.write(";")
        outfile_b.write(str(L.tolist()[i]))
        outfile_b.write("\n")
    outfile_b.close()
    
    
    outfile_a_train = open("data/features_A_values_train.txt",'w')
    #outfile_a_test.write("[")
    for row in A.tolist()[:split1]:
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_train.write(str(f'{item:.9f}'))
            outfile_a_train.write(" ")
        #outfile_a_test.write("]")
        outfile_a_train.write("\n")
    outfile_a_train.close()

    
    outfile_b_train = open("data/features_B_values_train.txt",'w')
    for row in B.tolist()[:split1]:
        for item in row:
            outfile_b_train.write(str(f'{item:.9f}'))
            outfile_b_train.write(" ")
        outfile_b_train.write("\n")
    outfile_b_train.close()

    outfile_a_test = open("data/features_A_values_test.txt",'w')
    #outfile_a_test.write("[")
    for row in A.tolist()[split2:]:
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_test.write(str(f'{item:.9f}'))
            outfile_a_test.write(" ")
        #outfile_a_test.write("]")
        outfile_a_test.write("\n")
    outfile_a_test.close()

    
    outfile_b_test = open("data/features_B_values_test.txt",'w')
    for row in B.tolist()[split2:]:
        for item in row:
            outfile_b_test.write(str(f'{item:.9f}'))
            outfile_b_test.write(" ")
        outfile_b_test.write("\n")
    outfile_b_test.close()
    

    
    outfile_a_val = open("data/features_A_values_val.txt",'w')
    for row in A.tolist()[split1:split2]:
        outfile_a_val.write("[")
        for item in row:
            outfile_a_val.write(str(f'{item:.9f}'))
            outfile_a_val.write(" ")
        outfile_a_val.write("]")
        outfile_a_val.write("\n")
    outfile_a_val.close()
    
    
    outfile_b_val = open("data/features_B_values_val.txt",'w')
    for row in B.tolist()[split1:split2]:
        outfile_b_val.write("[")
        for item in row:
            outfile_b_val.write(str(f'{item:.9f}'))
            outfile_b_val.write(" ")
        outfile_b_val.write("]")
        outfile_b_val.write("\n")
    outfile_b_val.close()
    

    
    outfile_x_test = open("data/features_X_values_test.txt",'w')
    for row in X.tolist()[split2:]:
        outfile_x_test.write("[")
        for item in row:
            outfile_x_test.write(str(f'{item:.9f}'))
            outfile_x_test.write(" ")
        outfile_x_test.write("]")
        outfile_x_test.write("\n")
    outfile_x_test.close()
    
    outfile_L = open("data/features_L_values_val.txt",'w')
    outfile_L.write(str(L_val.tolist()))
    outfile_L.close()
    
    outfile_L = open("data/features_L_values_test.txt",'w')
    outfile_L.write(str(L_test.tolist()))
    outfile_L.close()
    
    #Hyperparameters
    #gamma = 512
    lambs = [0.01,0.1,0.25,0.5,0.75,0.99]
    lambs = [0.01,0.1,0.25,0.5]

    margins = [0.1,0.25,0.5,0.75,1.0]

    
    #lambs = [0.1]
    #margins = [0.1]
    
    iterations = iters

    regularizers = [0]

    rates = {2:1000, 32:500, 64:10000, 128: 10000, 256:10000, 512:10000}
    #rates[64] = 500
    rates[64] = 20000
    
    rates[64] = 500
    
    #rates[64] = 15000
    #rates[64] = 40000
    #rates[64] = 10000
    #anneal_rate = 0.995
    anneal_rate = 0.9999
    #anneal_rate = 1
    #anneal_rate = 0.95
    #anneal_rate = 1
    #anneal_rate = 0.9
    #anneal_rate = 0.90
    if spec_margin:
        margins = [spec_margin]
    if spec_lamb:
        lambs = [spec_lamb]
    
    #anneal_rates = [1,0.9999,0.995,0.95,0.9]
    f = 0
    randoms = [i*5 for i in range(21,31)]
    special_counter = 0
    #lamb = 0.2
    aucs = []
    for reg in regularizers:
        for margin in margins:
            for lamb in lambs:
                rate = rates[gamma]
                #anneal_rate = anneal_rates[special_counter]
                #special_counter += 1
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
                #randie = randoms[f]
                #f+=1
                #randie = 24
                randie = 30
                randie = 1
                randie = 0
                #randie = 2
                print("seed of:",randie)
                print("rate, anneal_rate:", rate, anneal_rate)
                model = Linear_Feature_Fusion_Approximate(X_train,M,V,gamma,margin,lamb,regularization=reg,seed=randie)
                #model = Linear_Feature_Fusion(X_train,M,V,gamma,margin,lamb,regularization=reg)
                
                best_loss = model.loss()
                best_P = model.P
                best_scale = model.scale
                print("Initial loss:",best_loss)
                #P_history = []
                #P_history_matrices = []
                losses = []
                optimizer = torch.optim.SGD(model.parameters(), lr=rate)#, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=anneal_rate)
                for i in range(iterations):
                    
                    #first we select a scale such that our polynomial function will work
                    total = 0.0
                    P_temp = torch.div(model.P,torch.linalg.norm(model.P))
                    min_norm = 10000
                    for c in range(X_train.shape[0]):
                        norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                        total += norm
                        if norm < min_norm:
                            min_norm = norm
                    min_norm = float(min_norm)
                    total = float(total)
                    #print("total",total)
                    avg = total / X_train.shape[0]
                    
                    #print("average:",avg)
                    model.scale = 0.63245553 / avg #0.4 is the median of our valid range, therefore we want squared norm to be 0.4, therefore we use the sqrt of 0.4 as the numerator to make the new norm that
                    
                    #P_temp = torch.mul(P_temp,model.scale)
                    #for c in range(X_train.shape[0]):
                        #norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                        #print(norm**2)
                        
                    
                    #model.scale = 0.59581876 / avg #0.355 is the median of our valid range, therefore we want squared norm to be 0.4, therefore we use the sqrt of 0.4 as the numerator to make the new norm that
                    
                    #model.scale = 0.316227766 / min_norm #0.1 is the lowest of our valid range, therefore we can scale everything up to within our range
                    #model.scale = 0.70710678118 / avg
                    #print("new scale:",model.scale)
                    #P_temp = torch.mul(P_temp,model.scale)
                    #for c in range(X_train.shape[0]):
                        #print(torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))**2)
                    
                    
                    #model.P = torch.div(model.P, torch.linalg.norm(model.P))
                    #model.P = torch.mul(model.P, model.scale)
                    
                    loss = model.loss()
                    #print("Loss:",loss)
                    if loss <= best_loss:#+0.003
                        best_loss = loss
                        best_P = model.P
                        best_scale = model.scale
                    #else:
                        #print("Loss increased, ending training")
                        #break
                    losses.append(loss)
                    #P_history.append(str(model.P.tolist()))
                    #P_history_matrices.append(model.P)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    #model.P = torch.mul(torch.div(model.P,torch.linalg.norm(model.P)),3.0)
                    if i%10 == 0:
                        print("Iteration",str(i) + "/" + str(iterations))
                        print(loss)
                    #rate *= anneal_rate
                #print("final loss:",model.loss())
                #print("old best p norm:",torch.linalg.norm(best_P))
                best_P = torch.div(best_P,torch.linalg.norm(best_P))
                
                print("percentage of escape:",model.escape/model.tote)
                
                #calculate scale for best P
                total = 0.0
                min_norm = 10000
                for c in range(X_train.shape[0]):
                    norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                    total += norm
                    if norm < min_norm:
                        min_norm = norm
                min_norm = float(min_norm)
                total = float(total)
                avg = total / X_train.shape[0]
                best_scale = 0.63245553 / avg
                #best_scale = 0.59581876 / avg
                #best_scale = 0.316227766 / min_norm
                #best_scale = 0.70710678118 / avg
                
                best_P = torch.mul(best_P, best_scale)
                
                
                
                
                
                
                #for i in range(best_P.shape[1]):
                    #best_P[:,i]=torch.div(best_P[:,i], torch.linalg.norm(best_P[:,1]))
                #print("new best p norm:",torch.linalg.norm(best_P))
                
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)
                #print("new NOT normalized validation X_prime:", X_prime)
                #print("one vec:",X_prime[0,:].shape)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                auc = New_ROC_AUC(X_prime, L_val)
                print("EXACTED AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)
                #print("new NOT normalized validation X_prime:", X_prime)
                #print("one vec:",X_prime[0,:].shape)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                    #print("new norm:",torch.linalg.norm(X_prime[i,:]))
                    #print(X_prime[i,:])
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                    #print(X_prime[i,:])
                    #print()
                
                auc = New_ROC_AUC(X_prime, L_val)
                aucs.append((margin,lamb,auc,best_scale))
                print("AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                #P values file gets too large for github
                #p_file_name = "data/features_P_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + ".txt"
                #outfile_p = open(p_file_name,'w')
                #for P_value in P_history:
                    #outfile_p.write(P_value)
                    #outfile_p.write("\n")
                #outfile_p.close()
                
                p_best_file_name = "data/snap/degree=2/approximate_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #p_best_file_name = "data/exact_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_p_t = open(p_best_file_name,'w')
                P_final_t = best_P.T
                P_final_t = str(P_final_t.tolist())
                outfile_p_t.write(P_final_t)
                outfile_p_t.close()
                
                loss_file_name = "data/snap/degree=2/approximate_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #loss_file_name = "data/exact_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_loss = open(loss_file_name,'w')
                for loss_value in losses:
                    outfile_loss.write(str(loss_value.tolist()))
                    outfile_loss.write("\n")
                outfile_loss.close()
                #print()
                
                #X_prime = torch.mm(X_val,P_history_matrices[-1])
                #X_prime = torch.mm(X_val,best_P)
                #for i in range(X_prime.shape[0]):
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                #print("new normalized validation X_prime:", X_prime)
                
                #print("norm of best P:",torch.linalg.norm(best_P))
                
                X_prime_filename = "data/snap/degree=2/approximate_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
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
                
                
                X_filename = "data/snap/degree=2/approximate_labels_X_NOT_NORMAL_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
                
                
                for i in range(X_prime.shape[0]):
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                #print("new normalized test x_prime:", X_prime)
                
                X_prime_filename = "data/snap/degree=2/approximate_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_prime_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
    print("(margin, lambda, Validation AUC), Scale")
    print(aucs)
    
    
def train_no_snap_no_strict(gamma,iters,spec_margin=None,spec_lamb=None):
    #random.seed(1)
    torch.manual_seed(0)
    #A,B,L = data_gen(10)
    
    a = []
    A_infile = open("feature-extraction/extractions/VGG16_lfw-deepfunneled.txt",'r')
    for line in A_infile:
        line = line.strip().split()
        a.append(torch.tensor([float(char) for char in line]))
    A = torch.stack(a)
    #print(A)
    print("A:",A.shape)
    
    
    B = np.load("data/features/MMU_gallery_vgg.npy")
    B = np.reshape(B,(B.shape[0],B.shape[2]))
    B = torch.tensor(B)
    #B = torch.ones((B.shape))
    print("B:",B.shape)
    
    C = np.load("data/features/MMU2_gallery_vgg.npy")
    C = np.reshape(C,(C.shape[0],C.shape[2]))
    C = torch.tensor(C)
    #B = torch.ones((B.shape))
    print("C:",C.shape)
    
    for i in range(B.shape[0]):
        #print(torch.linalg.norm(torch.subtract(B[i,:],torch.div(B[i,:],torch.linalg.norm(B[i,:])))))
        #print(torch.linalg.norm(B[i,:]))
        B[i,:] = torch.div(B[i,:],torch.linalg.norm(B[i,:]))
    
    
    L = np.load("data/features/MMU_label_gallery.npy")
    #L = np.reshape(B,(B.shape[0],B.shape[2]))
    L = torch.tensor(L)
    L = L[:,0]
    print(L)
    print(L.shape)
    
    X = torch.cat((A,B),dim=1)
    
    #num_samples = L.shape[0]
    #num_each_class = 8
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
    
    

    


    outfile_a = open("data/features_A_values.txt",'w')
    for row in A.tolist():
        for item in row:
            outfile_a.write(str(f'{item:.9f}'))
            outfile_a.write(" ")
        outfile_a.write("\n")
    outfile_a.close()

    outfile_b = open("data/features_B_values.txt",'w')
    for i in range(B.shape[0]):
        outfile_b.write(str(B[i].tolist()))
        outfile_b.write(";")
        outfile_b.write(str(L.tolist()[i]))
        outfile_b.write("\n")
    outfile_b.close()
    
    
    outfile_a_train = open("data/features_A_values_train.txt",'w')
    #outfile_a_test.write("[")
    for row in A.tolist()[:split1]:
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_train.write(str(f'{item:.9f}'))
            outfile_a_train.write(" ")
        #outfile_a_test.write("]")
        outfile_a_train.write("\n")
    outfile_a_train.close()

    
    outfile_b_train = open("data/features_B_values_train.txt",'w')
    for row in B.tolist()[:split1]:
        for item in row:
            outfile_b_train.write(str(f'{item:.9f}'))
            outfile_b_train.write(" ")
        outfile_b_train.write("\n")
    outfile_b_train.close()

    outfile_a_test = open("data/features_A_values_test.txt",'w')
    #outfile_a_test.write("[")
    for row in A.tolist()[split2:]:
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_test.write(str(f'{item:.9f}'))
            outfile_a_test.write(" ")
        #outfile_a_test.write("]")
        outfile_a_test.write("\n")
    outfile_a_test.close()

    
    outfile_b_test = open("data/features_B_values_test.txt",'w')
    for row in B.tolist()[split2:]:
        for item in row:
            outfile_b_test.write(str(f'{item:.9f}'))
            outfile_b_test.write(" ")
        outfile_b_test.write("\n")
    outfile_b_test.close()
    

    
    outfile_a_val = open("data/features_A_values_val.txt",'w')
    for row in A.tolist()[split1:split2]:
        outfile_a_val.write("[")
        for item in row:
            outfile_a_val.write(str(f'{item:.9f}'))
            outfile_a_val.write(" ")
        outfile_a_val.write("]")
        outfile_a_val.write("\n")
    outfile_a_val.close()
    
    
    outfile_b_val = open("data/features_B_values_val.txt",'w')
    for row in B.tolist()[split1:split2]:
        outfile_b_val.write("[")
        for item in row:
            outfile_b_val.write(str(f'{item:.9f}'))
            outfile_b_val.write(" ")
        outfile_b_val.write("]")
        outfile_b_val.write("\n")
    outfile_b_val.close()
    

    
    outfile_x_test = open("data/features_X_values_test.txt",'w')
    for row in X.tolist()[split2:]:
        outfile_x_test.write("[")
        for item in row:
            outfile_x_test.write(str(f'{item:.9f}'))
            outfile_x_test.write(" ")
        outfile_x_test.write("]")
        outfile_x_test.write("\n")
    outfile_x_test.close()
    
    outfile_L = open("data/features_L_values_val.txt",'w')
    outfile_L.write(str(L_val.tolist()))
    outfile_L.close()
    
    outfile_L = open("data/features_L_values_test.txt",'w')
    outfile_L.write(str(L_test.tolist()))
    outfile_L.close()
    
    #Hyperparameters
    #gamma = 512
    lambs = [0.01,0.1,0.25,0.5,0.75,0.99]
    lambs = [0.01,0.1,0.25,0.5]

    margins = [0.1,0.25,0.5,0.75,1.0]

    
    #lambs = [0.1]
    #margins = [0.1]
    
    iterations = iters

    regularizers = [0]

    rates = {2:1000, 32:500, 64:10000, 128: 10000, 256:10000, 512:10000}
    #rates[64] = 500
    rates[64] = 20000
    
    rates[64] = 500
    
    #rates[64] = 15000
    #rates[64] = 40000
    #rates[64] = 10000
    #anneal_rate = 0.995
    anneal_rate = 0.9999
    #anneal_rate = 1
    #anneal_rate = 0.95
    #anneal_rate = 1
    #anneal_rate = 0.9
    #anneal_rate = 0.90
    if spec_margin:
        margins = [spec_margin]
    if spec_lamb:
        lambs = [spec_lamb]
    
    #anneal_rates = [1,0.9999,0.995,0.95,0.9]
    f = 0
    randoms = [i*5 for i in range(21,31)]
    special_counter = 0
    #lamb = 0.2
    aucs = []
    for reg in regularizers:
        for margin in margins:
            for lamb in lambs:
                rate = rates[gamma]
                #anneal_rate = anneal_rates[special_counter]
                #special_counter += 1
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
                #randie = randoms[f]
                #f+=1
                #randie = 24
                randie = 30
                randie = 1
                randie = 0
                #randie = 2
                print("seed of:",randie)
                print("rate, anneal_rate:", rate, anneal_rate)
                model = Linear_Feature_Fusion_Approximate3(X_train,M,V,gamma,margin,lamb,regularization=reg,seed=randie)
                #model = Linear_Feature_Fusion(X_train,M,V,gamma,margin,lamb,regularization=reg)
                
                best_loss = model.loss()
                best_P = model.P
                best_scale = model.scale
                print("Initial loss:",best_loss)
                #P_history = []
                #P_history_matrices = []
                losses = []
                optimizer = torch.optim.SGD(model.parameters(), lr=rate)#, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=anneal_rate)
                for i in range(iterations):
                    
                    #first we select a scale such that our polynomial function will work
                    total = 0.0
                    P_temp = torch.div(model.P,torch.linalg.norm(model.P))
                    min_norm = 10000
                    for c in range(X_train.shape[0]):
                        norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                        total += norm
                        if norm < min_norm:
                            min_norm = norm
                    min_norm = float(min_norm)
                    total = float(total)
                    #print("total",total)
                    avg = total / X_train.shape[0]
                    
                    #print("average:",avg)
                    model.scale = 0.63245553 / avg #0.4 is the median of our valid range, therefore we want squared norm to be 0.4, therefore we use the sqrt of 0.4 as the numerator to make the new norm that
                    
                    #P_temp = torch.mul(P_temp,model.scale)
                    #for c in range(X_train.shape[0]):
                        #norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                        #print(norm**2)
                        
                    
                    #model.scale = 0.59581876 / avg #0.355 is the median of our valid range, therefore we want squared norm to be 0.4, therefore we use the sqrt of 0.4 as the numerator to make the new norm that
                    
                    #model.scale = 0.316227766 / min_norm #0.1 is the lowest of our valid range, therefore we can scale everything up to within our range
                    #model.scale = 0.70710678118 / avg
                    #print("new scale:",model.scale)
                    #P_temp = torch.mul(P_temp,model.scale)
                    #for c in range(X_train.shape[0]):
                        #print(torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))**2)
                    
                    
                    #model.P = torch.div(model.P, torch.linalg.norm(model.P))
                    #model.P = torch.mul(model.P, model.scale)
                    
                    loss = model.loss()
                    #print("Loss:",loss)
                    if loss <= best_loss:#+0.003
                        best_loss = loss
                        best_P = model.P
                        best_scale = model.scale
                    #else:
                        #print("Loss increased, ending training")
                        #break
                    losses.append(loss)
                    #P_history.append(str(model.P.tolist()))
                    #P_history_matrices.append(model.P)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    #model.P = torch.mul(torch.div(model.P,torch.linalg.norm(model.P)),3.0)
                    if i%10 == 0:
                        print("Iteration",str(i) + "/" + str(iterations))
                        print(loss)
                    #rate *= anneal_rate
                #print("final loss:",model.loss())
                #print("old best p norm:",torch.linalg.norm(best_P))
                best_P = torch.div(best_P,torch.linalg.norm(best_P))
                
                print("percentage of escape:",model.escape/model.tote)
                
                #calculate scale for best P
                total = 0.0
                min_norm = 10000
                for c in range(X_train.shape[0]):
                    norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                    total += norm
                    if norm < min_norm:
                        min_norm = norm
                min_norm = float(min_norm)
                total = float(total)
                avg = total / X_train.shape[0]
                best_scale = 0.63245553 / avg
                #best_scale = 0.59581876 / avg
                #best_scale = 0.316227766 / min_norm
                #best_scale = 0.70710678118 / avg
                
                best_P = torch.mul(best_P, best_scale)
                
                
                
                
                
                
                #for i in range(best_P.shape[1]):
                    #best_P[:,i]=torch.div(best_P[:,i], torch.linalg.norm(best_P[:,1]))
                #print("new best p norm:",torch.linalg.norm(best_P))
                
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)
                #print("new NOT normalized validation X_prime:", X_prime)
                #print("one vec:",X_prime[0,:].shape)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                auc = New_ROC_AUC(X_prime, L_val)
                print("EXACTED AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)
                #print("new NOT normalized validation X_prime:", X_prime)
                #print("one vec:",X_prime[0,:].shape)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                    #print("new norm:",torch.linalg.norm(X_prime[i,:]))
                    #print(X_prime[i,:])
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                    #print(X_prime[i,:])
                    #print()
                
                auc = New_ROC_AUC(X_prime, L_val)
                aucs.append((margin,lamb,auc,best_scale))
                print("AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                #P values file gets too large for github
                #p_file_name = "data/features_P_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + ".txt"
                #outfile_p = open(p_file_name,'w')
                #for P_value in P_history:
                    #outfile_p.write(P_value)
                    #outfile_p.write("\n")
                #outfile_p.close()
                
                p_best_file_name = "data/loose/degree=2/approximate_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #p_best_file_name = "data/exact_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_p_t = open(p_best_file_name,'w')
                P_final_t = best_P.T
                P_final_t = str(P_final_t.tolist())
                outfile_p_t.write(P_final_t)
                outfile_p_t.close()
                
                loss_file_name = "data/loose/degree=2/approximate_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #loss_file_name = "data/exact_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_loss = open(loss_file_name,'w')
                for loss_value in losses:
                    outfile_loss.write(str(loss_value.tolist()))
                    outfile_loss.write("\n")
                outfile_loss.close()
                #print()
                
                #X_prime = torch.mm(X_val,P_history_matrices[-1])
                #X_prime = torch.mm(X_val,best_P)
                #for i in range(X_prime.shape[0]):
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                #print("new normalized validation X_prime:", X_prime)
                
                #print("norm of best P:",torch.linalg.norm(best_P))
                
                X_prime_filename = "data/loose/degree=2/approximate_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
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
                
                
                X_filename = "data/loose/degree=2/approximate_labels_X_NOT_NORMAL_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
                
                
                for i in range(X_prime.shape[0]):
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                #print("new normalized test x_prime:", X_prime)
                
                X_prime_filename = "data/loose/degree=2/approximate_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_prime_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
    print("(margin, lambda, Validation AUC), Scale")
    print(aucs)
    
    
def train_no_normal(gamma,iters,spec_margin=None,spec_lamb=None):
    #random.seed(1)
    torch.manual_seed(0)
    #A,B,L = data_gen(10)
    
    a = []
    A_infile = open("feature-extraction/extractions/VGG16_lfw-deepfunneled.txt",'r')
    for line in A_infile:
        line = line.strip().split()
        a.append(torch.tensor([float(char) for char in line]))
    A = torch.stack(a)
    #print(A)
    print("A:",A.shape)
    
    
    B = np.load("data/features/MMU_gallery_vgg.npy")
    B = np.reshape(B,(B.shape[0],B.shape[2]))
    B = torch.tensor(B)
    #B = torch.ones((B.shape))
    print("B:",B.shape)
    
    for i in range(B.shape[0]):
        #print(torch.linalg.norm(torch.subtract(B[i,:],torch.div(B[i,:],torch.linalg.norm(B[i,:])))))
        #print(torch.linalg.norm(B[i,:]))
        B[i,:] = torch.div(B[i,:],torch.linalg.norm(B[i,:]))
    
    
    L = np.load("data/features/MMU_label_gallery.npy")
    #L = np.reshape(B,(B.shape[0],B.shape[2]))
    L = torch.tensor(L)
    L = L[:,0]
    print(L)
    print(L.shape)
    
    X = torch.cat((A,B),dim=1)
    
    #num_samples = L.shape[0]
    #num_each_class = 8
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
    
    

    


    outfile_a = open("data/features_A_values.txt",'w')
    for row in A.tolist():
        for item in row:
            outfile_a.write(str(f'{item:.9f}'))
            outfile_a.write(" ")
        outfile_a.write("\n")
    outfile_a.close()

    outfile_b = open("data/features_B_values.txt",'w')
    for i in range(B.shape[0]):
        outfile_b.write(str(B[i].tolist()))
        outfile_b.write(";")
        outfile_b.write(str(L.tolist()[i]))
        outfile_b.write("\n")
    outfile_b.close()
    
    
    outfile_a_train = open("data/features_A_values_train.txt",'w')
    #outfile_a_test.write("[")
    for row in A.tolist()[:split1]:
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_train.write(str(f'{item:.9f}'))
            outfile_a_train.write(" ")
        #outfile_a_test.write("]")
        outfile_a_train.write("\n")
    outfile_a_train.close()

    
    outfile_b_train = open("data/features_B_values_train.txt",'w')
    for row in B.tolist()[:split1]:
        for item in row:
            outfile_b_train.write(str(f'{item:.9f}'))
            outfile_b_train.write(" ")
        outfile_b_train.write("\n")
    outfile_b_train.close()

    outfile_a_test = open("data/features_A_values_test.txt",'w')
    #outfile_a_test.write("[")
    for row in A.tolist()[split2:]:
        #outfile_a_test.write("[")
        for item in row:
            outfile_a_test.write(str(f'{item:.9f}'))
            outfile_a_test.write(" ")
        #outfile_a_test.write("]")
        outfile_a_test.write("\n")
    outfile_a_test.close()

    
    outfile_b_test = open("data/features_B_values_test.txt",'w')
    for row in B.tolist()[split2:]:
        for item in row:
            outfile_b_test.write(str(f'{item:.9f}'))
            outfile_b_test.write(" ")
        outfile_b_test.write("\n")
    outfile_b_test.close()
    

    
    outfile_a_val = open("data/features_A_values_val.txt",'w')
    for row in A.tolist()[split1:split2]:
        outfile_a_val.write("[")
        for item in row:
            outfile_a_val.write(str(f'{item:.9f}'))
            outfile_a_val.write(" ")
        outfile_a_val.write("]")
        outfile_a_val.write("\n")
    outfile_a_val.close()
    
    
    outfile_b_val = open("data/features_B_values_val.txt",'w')
    for row in B.tolist()[split1:split2]:
        outfile_b_val.write("[")
        for item in row:
            outfile_b_val.write(str(f'{item:.9f}'))
            outfile_b_val.write(" ")
        outfile_b_val.write("]")
        outfile_b_val.write("\n")
    outfile_b_val.close()
    

    
    outfile_x_test = open("data/features_X_values_test.txt",'w')
    for row in X.tolist()[split2:]:
        outfile_x_test.write("[")
        for item in row:
            outfile_x_test.write(str(f'{item:.9f}'))
            outfile_x_test.write(" ")
        outfile_x_test.write("]")
        outfile_x_test.write("\n")
    outfile_x_test.close()
    
    outfile_L = open("data/features_L_values_val.txt",'w')
    outfile_L.write(str(L_val.tolist()))
    outfile_L.close()
    
    outfile_L = open("data/features_L_values_test.txt",'w')
    outfile_L.write(str(L_test.tolist()))
    outfile_L.close()
    
    #Hyperparameters
    #gamma = 512
    lambs = [0.01,0.1,0.25,0.5,0.75,0.99]
    #lambs = [0.01,0.1,0.25,0.5]
    #lambs = [0.1,0.25,0.5]
    lambs = [0.5]
    #margins = [0.0,0.25,0.5]
    margins = [0.25,0.5,0.75,1.0]
    #lambs = [0.25]
    #margins = [0.1]
    margins = [1.0]
    #margins = [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]
    #margins = [0.5,0.75,1.0]
    #lambs = [0.1]
    iterations = iters

    regularizers = [0]

    rates = {2:1000, 32:500, 64:10000, 128: 10000, 256:10000, 512:10000}
    #rates[64] = 500
    rates[64] = 20000
    anneal_rate = 0.995
    anneal_rate = 0.9999
    #anneal_rate = 1
    #anneal_rate = 0.9
    #anneal_rate = 0.90
    if spec_margin:
        margins = [spec_margin]
    if spec_lamb:
        lambs = [spec_lamb]
    
    
    f = 0
    randoms = [i*5 for i in range(21,31)]
    
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
                #randie = randoms[f]
                #f+=1
                #randie = 24
                randie = 30
                randie = 0
                print("seed of:",randie)
                model = Linear_Feature_Fusion_No_Normal(X_train,M,V,gamma,margin,lamb,regularization=reg,seed=randie)
                #model = Linear_Feature_Fusion(X_train,M,V,gamma,margin,lamb,regularization=reg)
                
                best_loss = model.loss()
                best_P = model.P
                best_scale = model.scale
                print("Initial loss:",best_loss)
                #P_history = []
                #P_history_matrices = []
                losses = []
                optimizer = torch.optim.SGD(model.parameters(), lr=rate)#, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=anneal_rate)
                for i in range(iterations):
                    

                    loss = model.loss()
                    #print("Loss:",loss)
                    if loss <= best_loss:#+0.003
                        best_loss = loss
                        best_P = model.P
                        best_scale = model.scale

                    losses.append(loss)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    #model.P = torch.mul(torch.div(model.P,torch.linalg.norm(model.P)),3.0)
                    if i%10 == 0:
                        print("Iteration",str(i) + "/" + str(iterations))
                        print(model.loss())

                best_P = torch.div(best_P,torch.linalg.norm(best_P))

                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)

                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                auc = New_ROC_AUC(X_prime, L_val)
                print("EXACTED AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)

                
                auc = New_ROC_AUC(X_prime, L_val)
                aucs.append((margin,lamb,auc,best_scale))
                print("AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)

                p_best_file_name = "data/no_norm_approximate_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #p_best_file_name = "data/exact_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_p_t = open(p_best_file_name,'w')
                P_final_t = best_P.T
                P_final_t = str(P_final_t.tolist())
                outfile_p_t.write(P_final_t)
                outfile_p_t.close()
                
                loss_file_name = "data/no_norm_approximate_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                #loss_file_name = "data/exact_loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_loss = open(loss_file_name,'w')
                for loss_value in losses:
                    outfile_loss.write(str(loss_value.tolist()))
                    outfile_loss.write("\n")
                outfile_loss.close()

                X_prime_filename = "data/no_norm_approximate_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_val_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
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
                
                
                X_filename = "data/no_norm_approximate_labels_X_NOT_NORMAL_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
                
                
                for i in range(X_prime.shape[0]):
                    #X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                #print("new normalized test x_prime:", X_prime)
                
                X_prime_filename = "data/no_norm_approximate_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                #X_prime_filename = "data/exact_labels_X_prime_test_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma) + "_reg=" + str(reg) + ".txt"
                outfile_x = open(X_prime_filename,'w')
                x_list = X_prime.tolist()
                for i in range(len(x_list)):
                    x_list[i] = str(x_list[i])+";"+str(L_test.tolist()[i])
                    outfile_x.write(x_list[i])
                    outfile_x.write("\n")
                outfile_x.close()
                print()
    print("(margin, lambda, Validation AUC), Scale")
    print(aucs)
if __name__ == "__main__":
    #train(256,200)#for gamma=128, use margin=0.25,lamb=0.1)
    train_no_snap_no_strict(64,100)#for gamma=128, use margin=0.25,lamb=0.1)
    #train_no_normal(64,400)#for gamma=128, use margin=0.25,lamb=0.1)


