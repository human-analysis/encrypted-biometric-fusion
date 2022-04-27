import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas
import ast
from operator import itemgetter
import numpy as np
import math

from sklearn import metrics

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import top_k_accuracy_score



def Cosine_Similarity(vec1, vec2):
    return torch.dot(torch.div(vec1, torch.linalg.norm(vec1)), torch.div(vec2, torch.linalg.norm(vec2)))
def Cosine_Similarity_no_div(vec1, vec2):
    return torch.dot(vec1, vec2)

def Cosine_Distance(vec1, vec2):
    return 1 - torch.dot(torch.div(vec1, torch.linalg.norm(vec1)), torch.div(vec2, torch.linalg.norm(vec2)))

def Cosine_Distance_no_div(vec1, vec2):
    #assumes vec1 and vec2 are unit vectors
    return 1 - torch.dot(vec1,vec2)
    
    

def approx_inv_norm(x_in, degree):
    if degree == 6:
        coeffs = [[3.6604110068015703, -7.308745554603273, 5.359140241417692, -0.03216663533709177], [-1.761181767348659, 5.619133141454438, -7.496635998204148, 5.491355198579896]] #is this right?
    elif degree == 3:
        coeffs = [[-9.81663423,19.8459398,-13.57853979,4.38423127]]
    elif degree == 2:
        coeffs = [[5.91965872,-7.3475699,3.54940138]]
    elif degree == 1:
        coeffs = [[-2.61776258,2.78221164]]
    
    if degree == 6:
        coeffs = [[0.6063297491297636, -2.531307751235565, 3.750465920583647, 0.3300996447108867], [-0.08817608635502297, 1.0010471760514683, -3.7422390001841093, 5.2338148909066575]]
    elif degree == 3:
        coeffs = [[-0.3672598,2.10050565,-3.90021011,3.08345595]]
    elif degree == 2:
        coeffs = [[0.42084296,-1.81897596,2.51308415]]
    elif degree == 1:
        coeffs = [[-0.53582579,1.84020171]]
    
    
    #coeffs = [[11.836520387699572, -18.076619596914263, 9.213047940260486, -0.1390999565263271], [-5.035385227584069, 14.361565311498836, -14.664452287760135, 7.742745833744212]]
    
    x = torch.linalg.norm(x_in)**2

    result = 0
    for coeff_list in coeffs:
        result = coeff_list[0]
        for i in range(1,len(coeff_list)):
            result = result * x + coeff_list[i]
        x = result
        result = 0
    return x
    
def one_to_N(filename, title, labels=True, debug=False, original=False):
    enc_results_file = open(filename,'r')
    enc_results = []
    if labels:
        L = []
        for line in enc_results_file:
            result, l = line.strip().split(";")
            result = torch.tensor(ast.literal_eval(result)).unsqueeze(dim=0)
            l = int(l)
            enc_results.append(result)
            #print(torch.linalg.norm(result))
            L.append(l)
        enc_results_final = torch.Tensor(len(enc_results),enc_results[0].shape[0])
        torch.cat(enc_results, out=enc_results_final,dim=0)
        print(enc_results_final.shape)
        #for i in range(enc_results_final.shape[0]):
            #print(enc_results_final[i,0])
        #print(L)
    else:
        for line in enc_results_file:
            result = [float(item) for item in line.strip().split()]
            result = torch.tensor(result).unsqueeze(dim=0)
            #print(result.shape)
            enc_results.append(result)
        enc_results_final = torch.Tensor(len(enc_results),enc_results[0].shape[1])
        
        torch.cat(enc_results, out=enc_results_final,dim=0)
        
        print(enc_results_final.shape)
        #for i in range(enc_results_final.shape[0]):
            #print(enc_results_final[i,0])
        a_file = open("data5/dataset/L_values_test.txt",'r')
        L = a_file.readline()
        L = [int(i) for i in L[1:len(L)-2].split(", ")]
        print("samples =",len(L))
    
    y_score = []
    y_true = []
    
    results = []
    
    
    correct = 0
    count = len(L)
    for i in range(count):
        high_score = -100
        prediction = -1
        for j in range(count):
            if i == j:
                continue
            score = Cosine_Similarity_no_div(enc_results_final[i],enc_results_final[j])
            if score > high_score:
                high_score = score
                prediction = L[j]
        if L[i]==prediction:
            label = 1
        else:
            label = 0
        correct += label
    print(correct/count)
    
 
 
def one_to_N_Concat(filename, title, labels=True, debug=False, original=False):
    enc_results_file = open(filename,'r')
    enc_results = []
    if labels:
        pass
    else:
        for line in enc_results_file:
            line = line.replace("[","").replace("]","")
            result = [float(item) for item in line.strip().split()]
            result = torch.tensor(result).unsqueeze(dim=0)
            #print(result.shape)
            enc_results.append(result)
        enc_results_final = torch.Tensor(len(enc_results),enc_results[0].shape[1])
        
        torch.cat(enc_results, out=enc_results_final,dim=0)
        
        print(enc_results_final.shape)
        #for i in range(enc_results_final.shape[0]):
            #print(enc_results_final[i,0])
        a_file = open("data5/dataset/L_values_train.txt",'r')
        L = a_file.readline()
        L = [int(i) for i in L[1:len(L)-2].split(", ")]
        print("samples =",len(L))
    
    y_score = []
    y_true = []
    
    results = []
    
    
    correct = 0
    count = len(L)
    for i in range(count):
        high_score = -100
        prediction = -1
        for j in range(count):
            if i == j:
                continue
            score = Cosine_Similarity_no_div(enc_results_final[i],enc_results_final[j])
            if score > high_score:
                high_score = score
                prediction = L[j]
        if L[i]==prediction:
            label = 1
        else:
            label = 0
        correct += label
    print(correct/count)
    
    
def one_to_N_SK(filename, title, labels=True):
    enc_results_file = open(filename,'r')
    enc_results = []
    if labels:
        L = []
        for line in enc_results_file:
            result, l = line.strip().split(";")
            result = torch.tensor(ast.literal_eval(result)).unsqueeze(dim=0)
            l = int(l)
            enc_results.append(result)
            
            L.append(l)
        enc_results_final = torch.Tensor(len(enc_results),enc_results[0].shape[0])
        torch.cat(enc_results, out=enc_results_final,dim=0)
        print(enc_results_final.shape)
        #for i in range(enc_results_final.shape[0]):
            #print(enc_results_final[i,0])
        #print(L)
    else:
        for line in enc_results_file:
            line = line.replace("[", "")
            line = line.replace("]", "")
            result = [float(item) for item in line.strip().split()]
            result = torch.tensor(result).unsqueeze(dim=0)
            #print(result.shape)
            enc_results.append(result)
        enc_results_final = torch.Tensor(len(enc_results),enc_results[0].shape[1])
        
        torch.cat(enc_results, out=enc_results_final,dim=0)
        
        print(enc_results_final.shape)
        #for i in range(enc_results_final.shape[0]):
            #print(enc_results_final[i,0])
        a_file = open("data5/dataset/L_values_test.txt",'r')
        L = a_file.readline()
        L = [int(i) for i in L[1:len(L)-1].split(", ")]
        """
        L = L[:55]
        final_L = []
        for l in L:
            for _ in range(20):
                final_L.append(l)
        L = final_L"""
        print(len(L))
        #print(L)
    
    #for i in range(18):
        #print(torch.linalg.norm(enc_results_final[i]))
    
    y_score = []
    y_true = []
    count = len(L)
    #print(count)
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            score = Cosine_Similarity_no_div(enc_results_final[i],enc_results_final[j])
            #print(L[i],L[j])
            if L[i]==L[j]:
                label = 1
            else:
                label = 0
            y_score.append(score)
            y_true.append(label)
            #if label == 1:
                #print(label, score)

    acc = top_k_accuracy_score(y_true,y_score,k=1)
    print("acc:",acc)
    
if __name__ == "__main__":


    print("A")
    #one_to_N("data5/dataset/A_values_test_unique.txt","ROC_X",False)
    print()
    print("B")
    #one_to_N("data5/dataset/B_values_test_unique.txt","ROC_X",False)
    print()
    print("X")
    #one_to_N_Concat("data5/dataset/X_values_test.txt","ROC_X",False)
    print()


    one_to_N("results/fingerprint_normalized_encrypted_results_test_lambda=0.01_margin=0.1_gamma=32_poly2_.txt","ROC_Algo=Poly3Strict_Enc=Poly3",False)

    """
    print("A")
    #one_to_N("data4/dataset/A_values_test.txt","ROC_X",False)
    print()
    print("B")
    #one_to_N("data4/dataset/B_values_test.txt","ROC_X",False)
    print()
    print("X")
    one_to_N_Concat("data4/dataset/X_values_test.txt","ROC_X",False)
    print()

    
    print("Gamma = 64 now")
    print("encrypted exact train, poly 2 inference")
    one_to_N("results/normalized_encrypted_results_test_lambda=0.01_margin=0.1_gamma=32_exact2.txt","ROC_Algo=Poly3Strict_Enc=Poly3",False)
    print()
    
    print("encrypted exact train, poly6 inference")
    one_to_N("results/normalized_encrypted_results_test_lambda=0.01_margin=0.1_gamma=32_exact6.txt","ROC_Algo=Poly3Strict_Enc=Poly3",False)
    print()
    
    print("encrypted exact train, gold inference")
    one_to_N("results/normalized_encrypted_results_goldschmidt_lambda=0.01_margin=0.1_gamma=32.txt","ROC_Algo=Poly3Strict_Enc=Poly3",False)
    print()
    
    print("encrypted poly2 train, poly 2 inference")
    one_to_N("results/normalized_encrypted_results_test_lambda=0.01_margin=0.25_gamma=32_poly2.txt","ROC_Algo=Poly3Strict_Enc=Poly3",False)
    print()
    """
