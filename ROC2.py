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
    

    
def New_ROC_Encrypted_large(filename, title, labels=True, debug=False):
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
    
    y_score = []
    y_true = []
    
    results = []
    
    count = len(L)
    for i in range(count):
        
        for j in range(i,count):
            if i == j:
                continue
            score = Cosine_Similarity_no_div(enc_results_final[i],enc_results_final[j])
            #score = Cosine_Similarity(enc_results_final[i],enc_results_final[j])
            if L[i]==L[j]:
                label = 1
            else:
                label = 0
            y_score.append(score)
            y_true.append(label)
            results.append((score, enc_results_final[i], enc_results_final[j]))

    auc = roc_auc_score(y_true,y_score)
    print("AUC:",auc)
    
    if debug:
        print(y_score)
    
    fpr, tpr, thresholds = roc_curve(y_true,y_score)
    
    data_dict = {"False Positive Rate":fpr,"True Positive Rate":tpr}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="False Positive Rate",y="True Positive Rate",
                  title=title)
    fig.update_yaxes(range=[0,1])
    
    fig_file_name = "figures/" + title + ".png"
    fig.write_image(fig_file_name)
    
    return y_score, results
    
def New_ROC_large(filename, title, labels=True):
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
        
        #enc_results = list(set(enc_results))
        
        enc_results_final = torch.Tensor(len(enc_results),enc_results[0].shape[1])
        
        torch.cat(enc_results, out=enc_results_final,dim=0)
        
        print(enc_results_final.shape)
        #for i in range(enc_results_final.shape[0]):
            #print(enc_results_final[i,0])
        a_file = open("data5/dataset/L_values_test_unique.txt",'r')
        L = a_file.readline()
        L = [int(i) for i in L[1:len(L)-2].split(", ")]
        """
        L = L[:55]
        final_L = []
        for l in L:
            for _ in range(20):
                final_L.append(l)
        L = final_L"""
        print(len(L))
     
     
    
    #L = list(set(L))
    #print(len(L))
    #print(L)
    
    #for i in range(18):
        #print(torch.linalg.norm(enc_results_final[i]))
    
    y_score = []
    y_true = []
    count = len(L)
    #print(count)
    for i in range(count):
        for j in range(i,count):
            if i == j:
                continue
            #if torch.linalg.norm(enc_results_final[i] - enc_results_final[j]) < 0.00001:
                #continue
            score = Cosine_Similarity(enc_results_final[i],enc_results_final[j])
            #if score > 0.9999999:
                #continue
            if L[i]==L[j]:
                label = 1
            else:
                label = 0
            y_score.append(score)
            y_true.append(label)

    
    print(len(y_score))
    #fpr, tpr, thresholds = roc_curve(y_true,y_score)
    #print(fpr)
    #print(tpr)
    """
    auc = 0
    
    auc_dict = {}
    for i in range(len(tpr)):
        auc_dict[fpr[i]] = tpr[i]
    auc_list = list(auc_dict.items())
    auc_list.sort(key=itemgetter(0))
    
    for i in range(1,len(auc_list)):
        interval = abs(auc_list[i][0]-auc_list[i-1][0])
        auc += interval*auc_list[i-1][1]
        auc += 0.5 * interval * (auc_list[i][1]-auc_list[i-1][1])
    """
    auc = roc_auc_score(y_true,y_score)
    #print()
    #print(auc_list)
    print("AUC:",auc)
    
    fpr, tpr, thresholds = roc_curve(y_true,y_score)
    
    data_dict = {"False Positive Rate":fpr,"True Positive Rate":tpr}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="False Positive Rate",y="True Positive Rate",
                  title=title)
    fig.update_yaxes(range=[0,1])
    
    fig_file_name = "figures/" + title + ".png"
    fig.write_image(fig_file_name)



def New_ROC_P_Matrix_voice_face_large(filename, gamma, lamb, title, poly_degree=None):
    p_file = open(filename,'r')
    p = []
    #L = []
    p_final = None
    for line in p_file:
        #result, l = line.strip().split(";")
        #line = line[2:-2]
        #line = line.replace("[","")
        #line = line.replace("]","")
        #line = line.replace(",","")
        #line = line.strip()
        #print(line[0],line[-1])
        #line = [float(i) for i in line.split(" ")]
        #result = torch.tensor(line)
        #print(result.shape)
        result = torch.tensor(ast.literal_eval(line.strip()))
        #print(result.shape)
        #l = int(l)
        p.append(result)
        p_final = result
        #L.append(l)
    #p_final = torch.Tensor(len(p),p[0].shape[0])
    #torch.cat(p, out=p_final,dim=0)
    
    #num_class = len(set(L))
    #samples_per_class = L.count(L[0])
    
    test = True
    
    if test:
        a_file = open("data5/dataset/A_values_test.txt",'r')
        b_file = open("data5/dataset/B_values_test.txt",'r')
        L_file = open("data5/dataset/L_values_test.txt",'r')
    else:
        a_file = open("data5/dataset/A_values_val.txt",'r')
        b_file = open("data5/dataset/B_values_val.txt",'r')
        L_file = open("data5/dataset/L_values_val.txt",'r')
    
    A = []
    
    
    L = L_file.readline()
    L = [int(i) for i in L[1:len(L)-2].split(", ")]
    #L = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
    for line in a_file:
        line = line.replace("[","")
        line = line.replace("]","")
        line = line.replace(",","")
        line = line.strip()
        line = [float(i) for i in line.split(" ")]
        result = torch.tensor(line).unsqueeze(dim=0)
        #print(result.shape)
        A.append(result)
    A_final = torch.Tensor(len(A),A[0].shape[0])
    torch.cat(A, out=A_final,dim=0)
    
    
    B = []
    for line in b_file:
        line = line.replace("[","")
        line = line.replace("]","")
        line = line.replace(",","")
        line = line.strip()
        line = [float(i) for i in line.split(" ")]
        result = torch.tensor(line).unsqueeze(dim=0)
        B.append(result)
    B_final = torch.Tensor(len(B),B[0].shape[0])
    torch.cat(B, out=B_final,dim=0)
        
    #print("here")
    #l_file = open("data/features_labels_X_prime_test_lambda=0.99_margin=0.5_gamma=128.txt",'r')
    #L = []
    #for line in l_file:
        #L.append(float(line.strip().split(";")[1]))
    
    
    #Create feature fusion dataset
    X = torch.cat((A_final,B_final),dim=1)
    #p_final = torch.mul(p_final,10)
    #p_final = torch.div(p_final,torch.linalg.norm(p_final))
    X_prime = torch.mm(X, p_final.T)
    #X_prime = torch.mm(p_final, X.T)
    print(X_prime.shape)
    """
    print()
    print("l2(p):",torch.linalg.norm(p_final))
    
    print()
    for i in range(X_prime.shape[0]):
        print(torch.linalg.norm(X_prime[i,:]))
    print()
    
    print(L)
    print(X_prime.shape)
    """
    for i in range(X_prime.shape[0]):
        if poly_degree:
            X_prime[i,:]=torch.mul(X_prime[i,:], approx_inv_norm(X_prime[i,:],poly_degree))
        else:
            X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
    #X_prime = X_prime.T

    count = len(L)
    
    
    
    y_score = []
    y_true = []
    count = len(L)
    for i in range(count):
        for j in range(i,count):
            if i == j:
                continue
            score = Cosine_Similarity_no_div(X_prime[i],X_prime[j])
            if L[i]==L[j]:
                label = 1
            else:
                label = 0
            y_score.append(score.detach().numpy())
            y_true.append(label)
    auc = roc_auc_score(y_true,y_score)
    print("AUC:",auc)


if __name__ == "__main__":
    #print(torch.torch.cuda.is_available())
    #print()
    print("A")
    New_ROC_large("data5/dataset/A_values_test_unique.txt","ROC_X",False)
    print()
    print("B")
    New_ROC_large("data5/dataset/B_values_test_unique.txt","ROC_X",False)
    print()
    print("X")
    New_ROC_large("data5/dataset/X_values_test.txt","ROC_X",False)
    print()

    
    
    print("Gamma = 32 now")
    print("plaintext exact train, exact inference")
    New_ROC_P_Matrix_voice_face_large("data5/exact_results/exact_best_P_value_transpose_lambda=0.5_margin=0.1_gamma=32_reg=0.txt",1,1,"test")
    print("plaintext exact train, poly6 inference")
    New_ROC_P_Matrix_voice_face_large("data5/exact_results/exact_best_P_value_transpose_lambda=0.5_margin=0.1_gamma=32_reg=0.txt",1,1,"test",6)
    print("plaintext exact train, poly 2 inference")
    New_ROC_P_Matrix_voice_face_large("data5/exact_results/exact_best_P_value_transpose_lambda=0.5_margin=0.1_gamma=32_reg=0.txt",1,1,"test",2)
    #print("plaintext exact train, poly 1 inference")
    #New_ROC_P_Matrix_voice_face_large("data4/exact_results/exact_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=32_reg=0.txt",1,1,"test",1)
    print()
    
    
    print("encrypted exact train, gold inference")
    New_ROC_Encrypted_large("results/finger_normalized_encrypted_results_goldschmidt_gamma=32.txt","ROC_Algo=Poly3Strict_Enc=Poly3",False)
    print()
    
    print("encrypted simple average")
    #New_ROC_Encrypted_large("results/averaged_results.txt","ROC_Algo=Poly3Strict_Enc=Poly3",False)
    print()
    
    print("encrypted exact train, poly 6 inference")
    New_ROC_Encrypted_large("results/finger_normalized_encrypted_results_test_lambda=0.5_margin=0.1_gamma=32_exact6_.txt","ROC_Algo=Poly3Strict_Enc=Poly3",False)
    print()
    
    print("encrypted exact train, poly 2 inference")
    New_ROC_Encrypted_large("results/finger_normalized_encrypted_results_test_lambda=0.5_margin=0.1_gamma=32_exact2_.txt","ROC_Algo=Poly3Strict_Enc=Poly3",False)
    print()
    
    print("plaintext poly2, poly 2 inference")
    New_ROC_P_Matrix_voice_face_large("data5/degree=3strict/approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=32_reg=0.txt",1,1,"test",2)
    print()
    print("encrypted poly2, poly2 inference")
    New_ROC_Encrypted_large("results/fingerprint_normalized_encrypted_results_test_lambda=0.01_margin=0.1_gamma=32_poly2_.txt","ROC_Algo=Poly3Strict_Enc=Poly3",False)
    print()
    
    
