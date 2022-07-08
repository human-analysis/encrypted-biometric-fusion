"""
Author: Luke Sperling
Created: 04-04-22
Modified: 07-08-22
Training procedure for FHE-aware Learning (HEFT). Also includes version where exact normalization is used. This is considered not FHE-aware Learning.
"""

import numpy as np
import random
import os
from operator import itemgetter

from sklearn.decomposition import PCA
import math

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from model import Linear_Feature_Fusion, Linear_Feature_Fusion_FHEaware
from data_generation import data_gen

from ROC import New_ROC_AUC


#polynomial approx of norm
def approximate_inv_norm(x_in):
    coeffs = [[0.42084296,-1.81897596,2.51308415]]
    x = torch.linalg.norm(x_in)**2
    result = 0
    for coeff_list in coeffs:
        result = coeff_list[0]
        for i in range(1,len(coeff_list)):
            result = result * x + coeff_list[i]
        x = result
        result = 0
    return x

def train_exact(gamma,iters,spec_margin=None,spec_lamb=None):
    """
    train using the exact inverse norm instead of approx inverse norm
    gamma - output dimension
    iters - number of epochs to train
    spec_margin, spec_lamb - margin and lambda hyperparameters may be specified
    """
    random.seed(123)
    torch.manual_seed(123)
    
    
    #start by loading feature vectors and labels
    a = []
    A_infile = open("feature-extraction/extractions/VGGFace_vgg_cplfw_new.txt",'r')
    for line in A_infile:
        line = line.strip().split()
        a.append(torch.tensor([float(char) for char in line]))

    L = []
    L_infile = open("feature-extraction/extractions/deep_speaker_librispeech_google_labels.txt",'r')
    
    l_dict = {}
    for line in L_infile:
        line = line.strip()
        if line not in l_dict:
            l_dict[line] = 0
        l_dict[line] += 1
        L.append(line.strip())
    #convert from unique string labels to integers
    #from https://stackoverflow.com/questions/43203215/map-unique-strings-to-integers-in-python
    d = dict([(y,x+1) for x,y in enumerate(sorted(set(L)))])
    L = [d[x] for x in L]
    
    l_dict = {}
    for l in L:
        if l not in l_dict:
            l_dict[l] = 0
        l_dict[l] += 1
    
    b = []
    B_infile = open("feature-extraction/extractions/deep_speaker_librispeech_google.txt",'r')
    for line in B_infile:
        line = line.strip().split()
        b.append(torch.tensor([float(char) for char in line]))
    
    b, L = (list(t) for t in zip(*sorted(zip(b, L),key=itemgetter(1))))
    
    
    #pair all possible combinations of feature vectors
    completed = 0
    a_index = 0
    a_sub_index = 0
    b_index = 0
    samples_per_face = 2
    samples_per_voice = 10
    num_each_class = samples_per_face * samples_per_voice
    true_a = []
    true_b = []
    true_L = []
    
    
    num_classes = 188
    split1 = 0
    split2 = 0
    
    last_l = L[0]
    i = 0
    while completed <= 188:
        #for i, label in enumerate(L):
        label = L[b_index]
        for j in range(2):
            a_sub_index = j
            print(a_index+a_sub_index, b_index, label)
            true_a.append(a[a_index+a_sub_index])
            true_b.append(b[b_index])
            true_L.append(L[b_index])
        b_index += 1
        
        if b_index >= len(L):
            break
        label = L[b_index]
        if label != last_l:
            last_l = label
            a_index += 2
            completed += 1
            if split1 == 0 and a_index/2 >= math.floor(num_classes * 0.2):
                split1 = i
            if split2 == 0 and a_index/2 >= math.floor(num_classes * 0.4):
                split2 = i
        i += 1

    print(len(true_a), len(true_b))
    
    print("Splits:",split1,split2)
    
    A = torch.stack(true_a)
    print("A:",A.shape)
    B = torch.stack(true_b)
    print("B:",B.shape)
    
    
    L2 = L
    L = torch.tensor(true_L)
    print("L:",L.shape)
    print(L)
    
    
    #now we can create our final dataset
    X = torch.cat((A,B),dim=1)
    X_train = X[split2:,:]
    X_test = X[split1:split2,:]
    X_val = X[:split1,:]
    
    L_train = L[split2:]
    L_test = L[split1:split2]
    L_val = L[:split1]
    
    #randomize train set order to aid in batches
    temp = list(zip(X_train.tolist(), L_train))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    X_train_list, L_train_list = list(res1), list(res2)
    
    X_train = torch.tensor(X_train_list)
    L_train = torch.tensor(L_train_list)
    
    print("X_train shuffled:",X_train.shape)
    print("L_train shuffled:",L_train.shape)
    
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/dataset"):
        os.mkdir("data/dataset")
    if not os.path.exists("data/exact_results"):
        os.mkdir("data/exact_results")
    if not os.path.exists("data/exact_results/"):
        os.mkdir("data/dataset")

    outfile_a = open("data/dataset/A_values.txt",'w')
    for row in A.tolist():
        for item in row:
            outfile_a.write(str(f'{item:.9f}'))
            outfile_a.write(" ")
        outfile_a.write("\n")
    outfile_a.close()
    
    outfile_b = open("data/dataset/B_values.txt",'w')
    for row in B.tolist():
        for item in row:
            outfile_b.write(str(f'{item:.9f}'))
            outfile_b.write(" ")
        outfile_b.write("\n")
    outfile_b.close()
    
    outfile_a_train = open("data/dataset/A_values_train.txt",'w')
    for row in A.tolist()[split2:]:
        for item in row:
            outfile_a_train.write(str(f'{item:.9f}'))
            outfile_a_train.write(" ")
        outfile_a_train.write("\n")
    outfile_a_train.close()
    
    outfile_b_train = open("data/dataset/B_values_train.txt",'w')
    for row in B.tolist()[split2:]:
        for item in row:
            outfile_b_train.write(str(f'{item:.9f}'))
            outfile_b_train.write(" ")
        outfile_b_train.write("\n")
    outfile_b_train.close()

    outfile_a_test = open("data/dataset/A_values_test.txt",'w')
    for row in A.tolist()[split1:split2]:
        for item in row:
            outfile_a_test.write(str(f'{item:.9f}'))
            outfile_a_test.write(" ")
        outfile_a_test.write("\n")
    outfile_a_test.close()

    
    outfile_b_test = open("data/dataset/B_values_test.txt",'w')
    for row in B.tolist()[split1:split2]:
        for item in row:
            outfile_b_test.write(str(f'{item:.9f}'))
            outfile_b_test.write(" ")
        outfile_b_test.write("\n")
    outfile_b_test.close()
    
    outfile_a_test = open("data/dataset/A_values_test_transpose.txt",'w')
    for row in A[split1:split2,:].T.tolist():
        for item in row:
            outfile_a_test.write(str(f'{item:.9f}'))
            outfile_a_test.write(" ")
        outfile_a_test.write("\n")
    outfile_a_test.close()
    
    outfile_b_test = open("data/dataset/B_values_test_transpose.txt",'w')
    for row in B[split1:split2,:].T.tolist():
        for item in row:
            outfile_b_test.write(str(f'{item:.9f}'))
            outfile_b_test.write(" ")
        outfile_b_test.write("\n")
    outfile_b_test.close()
    
    outfile_a_val = open("data/dataset/A_values_val.txt",'w')
    for row in A.tolist()[:split1]:
        outfile_a_val.write("[")
        for item in row:
            outfile_a_val.write(str(f'{item:.9f}'))
            outfile_a_val.write(" ")
        outfile_a_val.write("]")
        outfile_a_val.write("\n")
    outfile_a_val.close()
    
    outfile_b_val = open("data/dataset/B_values_val.txt",'w')
    for row in B.tolist()[:split1]:
        outfile_b_val.write("[")
        for item in row:
            outfile_b_val.write(str(f'{item:.9f}'))
            outfile_b_val.write(" ")
        outfile_b_val.write("]")
        outfile_b_val.write("\n")
    outfile_b_val.close()
    
    outfile_x_test = open("data/dataset/X_values_test.txt",'w')
    for row in X.tolist()[split1:split2]:
        outfile_x_test.write("[")
        for item in row:
            outfile_x_test.write(str(f'{item:.9f}'))
            outfile_x_test.write(" ")
        outfile_x_test.write("]")
        outfile_x_test.write("\n")
    outfile_x_test.close()
    
    outfile_L = open("data/dataset/L_values_val.txt",'w')
    outfile_L.write(str(L_val.tolist()))
    outfile_L.close()
    
    outfile_L = open("data/dataset/L_values_test.txt",'w')
    outfile_L.write(str(L_test.tolist()))
    outfile_L.close()
    
    A2 = torch.stack(a)
    B2 = torch.stack(b)
    L2 = torch.tensor(L2)
    
    outfile_b_test = open("data/dataset/A_values_test_unique.txt",'w')
    for row in A2.tolist()[math.floor(0.2*num_classes)*2:math.floor(0.4*num_classes)*2]:
        for item in row:
            outfile_b_test.write(str(f'{item:.9f}'))
            outfile_b_test.write(" ")
        outfile_b_test.write("\n")
    outfile_b_test.close()
    
    outfile_b_test = open("data/dataset/B_values_test_unique.txt",'w')
    for row in B2.tolist()[split1//2:split2//2]:
        for item in row:
            outfile_b_test.write(str(f'{item:.9f}'))
            outfile_b_test.write(" ")
        outfile_b_test.write("\n")
    outfile_b_test.close()
    
    outfile_L = open("data/dataset/L_values_test_unique.txt",'w')
    outfile_L.write(str(L2[split1//2:split2//2].tolist()))
    outfile_L.close()
    
    #Hyperparameters
    lambs = [0.01,0.1,0.25,0.5,0.75,0.99]
    margins = [0.1,0.25,0.5,0.75,1.0]
    iterations = iters

    regularizers = [0]
    
    rate = 0.005
    decay = 0.0001
    
    #function parameters
    if spec_margin:
        margins = [spec_margin]
    if spec_lamb:
        lambs = [spec_lamb]
    
    
    aucs = []
    #repeat training for each hyperparam combination
    for reg in regularizers:
        for margin in margins:
            for lamb in lambs:
                print("Lambda value of", lamb, "Margin value of", margin, "regularizer of",reg)
                M = []
                V = []
                randie = 123
                print("seed of:",randie)
                model = Linear_Feature_Fusion(X_train,M,V,L_train,gamma,margin,lamb,regularization=reg,seed=randie)
                best_loss = model.loss()
                best_P = model.P
                print("Initial loss:",best_loss)
                losses = []
                optimizer = torch.optim.Adam(model.parameters(), lr=rate, weight_decay=decay)
                for i in range(iterations):
                    loss = model.loss()
                    if loss <= best_loss:
                        best_loss = loss
                        best_P = model.P
                    losses.append(loss)
                    loss.backward()
                    optimizer.step()
                    if i%10 == 0:
                        print("Iteration",str(i) + "/" + str(iterations))
                        print(model.loss())
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)

                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)

                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                
                auc = New_ROC_AUC(X_prime, L_val)
                aucs.append((margin,lamb,auc))
                print("AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)

                p_best_file_name = "data/exact_results/exact_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_p_t = open(p_best_file_name,'w')
                P_final_t = best_P.T
                P_final_t = str(P_final_t.tolist())
                outfile_p_t.write(P_final_t)
                outfile_p_t.close()
                
    print("(margin, lambda, Validation AUC)")
    print(aucs)

def train(gamma,iters,spec_margin=None,spec_lamb=None):
    """
    train using the approx inverse norm. same as train_exact otherwise
    gamma - output dimension
    iters - number of epochs to train
    spec_margin, spec_lamb - margin and lambda hyperparameters may be specified
    """
    random.seed(123)
    torch.manual_seed(123)
    
    #start by loading feature vectors and labels
    a = []
    A_infile = open("feature-extraction/extractions/VGGFace_vgg_cplfw.txt",'r')
    for line in A_infile:
        line = line.strip().split()
        a.append(torch.tensor([float(char) for char in line]))
    
    L = []
    L_infile = open("feature-extraction/extractions/deep_speaker_librispeech_google_labels.txt",'r')
    
    l_dict = {}
    for line in L_infile:
        line = line.strip()
        if line not in l_dict:
            l_dict[line] = 0
        l_dict[line] += 1
        
        L.append(line.strip())
    
    #convert from unique string labels to integers
    #from https://stackoverflow.com/questions/43203215/map-unique-strings-to-integers-in-python
    d = dict([(y,x+1) for x,y in enumerate(sorted(set(L)))])
    L = [d[x] for x in L]
    
    l_dict = {}
    for l in L:
        if l not in l_dict:
            l_dict[l] = 0
        l_dict[l] += 1
    #print(L)
    
    b = []
    B_infile = open("feature-extraction/extractions/deep_speaker_librispeech_google.txt",'r')
    for line in B_infile:
        line = line.strip().split()
        b.append(torch.tensor([float(char) for char in line]))
    
    
    b, L = (list(t) for t in zip(*sorted(zip(b, L),key=itemgetter(1))))
    
    #pair all possible combinations of feature vectors
    completed = 0
    a_index = 0
    a_sub_index = 0
    b_index = 0
    samples_per_face = 2
    samples_per_voice = 10
    num_each_class = samples_per_face * samples_per_voice
    true_a = []
    true_b = []
    true_L = []
    
    
    num_classes = 188
    split1 = 0
    split2 = 0
    
    last_l = L[0]
    i = 0
    while completed <= 188:
        label = L[b_index]
        for j in range(2):
            a_sub_index = j
            print(a_index+a_sub_index, b_index, label)
            true_a.append(a[a_index+a_sub_index])
            true_b.append(b[b_index])
            true_L.append(L[b_index])
        b_index += 1
        
        if b_index >= len(L):
            break
        label = L[b_index]
        if label != last_l:
            last_l = label
            a_index += 2
            completed += 1
            if split1 == 0 and a_index/2 >= math.floor(num_classes * 0.2):
                split1 = i
            if split2 == 0 and a_index/2 >= math.floor(num_classes * 0.4):
                split2 = i
        i += 1
        
    print(len(true_a), len(true_b))
    
    
    print("Splits:",split1,split2)
    
    A = torch.stack(true_a)
    print("A:",A.shape)
    B = torch.stack(true_b)
    print("B:",B.shape)
    
    L = torch.tensor(true_L)
    print("L:",L.shape)
    print(L)
    
    
    
    X = torch.cat((A,B),dim=1)
    X_train = X[split2:,:]
    X_test = X[split1:split2,:]
    X_val = X[:split1,:]
    
    L_train = L[split2:]
    L_test = L[split1:split2]
    L_val = L[:split1]
    
    #randomize train set order to aid in batches
    temp = list(zip(X_train.tolist(), L_train))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    X_train_list, L_train_list = list(res1), list(res2)
    
    X_train = torch.tensor(X_train_list)
    L_train = torch.tensor(L_train_list)
    
    print("X_train shuffled:",X_train.shape)
    print("L_train shuffled:",L_train.shape)
    
    if not os.path.exists("data/degree=3strict"):
        os.mkdir("data/degree=3strict")
    if not os.path.exists("data/degree=2strict"):
        os.mkdir("data/degree=2strict")
    
    
    #Hyperparameters
    lambs = [0.01,0.1,0.25,0.5,0.75,0.99]
    margins = [0.1,0.25,0.5,0.75,1.0]
    iterations = iters

    regularizers = [0]

    rate = 0.005
    decay = 0.0001

    if spec_margin:
        margins = [spec_margin]
    if spec_lamb:
        lambs = [spec_lamb]

    #run full training for every hyperparam combination
    aucs = []
    for reg in regularizers:
        for margin in margins:
            for lamb in lambs:
                print("Lambda value of", lamb, "Margin value of", margin, "regularizer of",reg)
                
                M = []
                V = []
                randie = 123
                print("seed of:",randie)
                print("rate, anneal_rate:", rate, anneal_rate)
                model = Linear_Feature_Fusion_FHEaware(X_train,M,V,L_train,gamma,margin,lamb,regularization=reg,seed=randie)
                
                #first we select a scale such that our polynomial function will work properly
                total = 0.0
                P_temp = torch.div(model.P,torch.linalg.norm(model.P))
                min_norm = 10000
                for c in range(X_train.shape[0]):
                    norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                    total += norm
                total = float(total)
                avg = total / X_train.shape[0]
                model.scale = 1.525**0.5 / avg
                
                best_loss = model.loss()
                best_P = model.P
                best_scale = model.scale
                print("Initial loss:",best_loss)
                losses = []
                optimizer = torch.optim.Adam(model.parameters(), lr=rate, weight_decay=decay)
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
                    avg = total / X_train.shape[0]
                    model.scale = 1.525**0.5 / avg
                    
                    loss = model.loss()
                    if loss <= best_loss:
                        best_loss = loss
                        best_P = model.P
                        best_scale = model.scale
                    losses.append(loss)
                    loss.backward()
                    optimizer.step()
                    if i%10 == 0:
                        print("Iteration",str(i) + "/" + str(iterations))
                        print(loss)
                best_P = torch.div(best_P,torch.linalg.norm(best_P))
                
                print("percentage of escape:",model.escape/model.tote)
                
                #calculate scale for best P
                total = 0.0
                min_norm = 10000
                for c in range(X_train.shape[0]):
                    norm = torch.linalg.norm(torch.matmul(P_temp.T,X_train[c,:].T))
                    total += norm
                total = float(total)
                avg = total / X_train.shape[0]
                best_scale = 1.525**0.5 / avg
                
                best_P = torch.mul(best_P, best_scale)
                
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
                auc = New_ROC_AUC(X_prime, L_val)
                print("AUC (using exact normalziation) of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                
                X_prime = torch.mm(X_val,best_P)
                print(X_prime.shape)
                for i in range(X_prime.shape[0]):
                    X_prime[i,:]=torch.mul(X_prime[i,:], approximate_inv_norm(X_prime[i,:]))
                
                auc = New_ROC_AUC(X_prime, L_val)
                aucs.append((margin,lamb,auc,best_scale))
                print("AUC of lambda="+str(lamb)+", margin="+str(margin)+ "_reg=" + str(reg) + ":", auc)
                
                p_best_file_name = "data/degree=2strict/approximate_best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + "_gamma=" + str(gamma)  + "_reg=" + str(reg) + ".txt"
                outfile_p_t = open(p_best_file_name,'w')
                P_final_t = best_P.T
                P_final_t = str(P_final_t.tolist())
                outfile_p_t.write(P_final_t)
                outfile_p_t.close()
                
    print("(margin, lambda, Validation AUC), Scale")
    print(aucs)
    
    
    
if __name__ == "__main__":
    train_exact(32,1000)
    train(32,1000)

