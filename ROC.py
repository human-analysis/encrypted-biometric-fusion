import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas
import ast
from operator import itemgetter
import numpy as np
import math

from sklearn import metrics



def Cosine_Distance(vec1, vec2):
    return 1 - torch.dot(torch.div(vec1, torch.linalg.norm(vec1)), torch.div(vec2, torch.linalg.norm(vec2)))

def Cosine_Distance_no_div(vec1, vec2):
    #assumes vec1 and vec2 are unit vectors
    return 1 - torch.dot(vec1,vec2)

def ROC(filename, tag, title):
    enc_results_file = open(filename,'r')
    enc_results = []
    L = []
    for line in enc_results_file:
        result, l = line.strip().split(";")
        result = torch.tensor(ast.literal_eval(result)).unsqueeze(dim=0)
        l = int(l)
        enc_results.append(result)
        L.append(l)
    enc_results_final = torch.Tensor(len(enc_results),enc_results[0].shape[0])
    torch.cat(enc_results, out=enc_results_final,dim=0)
    
    print(len(enc_results))
    #for i in range(enc_results_final.shape[0]):
        #pass
    
    thresholds = [0.05 * i for i in range(41)]
    thresholds = [0.01 * i for i in range(201)]
    
    num_class = len(set(L))
    samples_per_class = L.count(L[0])
    
    count = num_class * samples_per_class
    count = 90
    count = len(L)
    fps = []
    tps = []
    for threshold in thresholds:
        fp = 0
        tp = 0
        pc = 0
        nc = 0
        for i in range(count):
            for j in range(i,count):
                guess = False
                if Cosine_Distance(enc_results_final[i],enc_results_final[j]) <= threshold:
                    guess = True
                if L[i]==L[j]:
                    pc += 1
                    if guess:
                        tp += 1
                else:
                    nc += 1
                    if guess:
                        fp += 1
        fp = fp/nc
        tp = tp/pc
        fps.append(fp)
        tps.append(tp)
        
        
    #print(L)
    #scores = [ for result in enc_results_final]
    #fps, tps, thresholds = metrics.roc_curve(L, scores)
    
    data_dict = {"False Positive Rate":fps,"True Positive Rate":tps}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="False Positive Rate",y="True Positive Rate",
                  title=title)
    fig.update_yaxes(range=[0,1])
    fig_file_name = "figures/ROC_" + tag + ".png"
    fig.write_image(fig_file_name)
    
    auc = 0
    
    auc_dict = {}
    for i in range(len(tps)):
        auc_dict[fps[i]] = tps[i]
    auc_list = list(auc_dict.items())
    auc_list.sort(key=itemgetter(0))
    
    for i in range(1,len(auc_list)):
        interval = abs(auc_list[i][0]-auc_list[i-1][0])
        auc += interval*auc_list[i-1][1]
        auc += 0.5 * interval * (auc_list[i][1]-auc_list[i-1][1])
        
    #print()
    #print(auc_list)
    print("AUC:",auc)

def ROC2(filename, gamma, lamb, title):
    enc_results_file = open(filename,'r')
    enc_results = []
    L = []
    for line in enc_results_file:
        result, l = line.strip().split(";")
        result = torch.tensor(ast.literal_eval(result)).unsqueeze(dim=0)
        l = int(l)
        enc_results.append(result)
        L.append(l)
    enc_results_final = torch.Tensor(len(enc_results),enc_results[0].shape[0])
    torch.cat(enc_results, out=enc_results_final,dim=0)
    
    thresholds = [0.05 * i for i in range(41)]
    thresholds = [0.01 * i for i in range(201)]
    
    num_class = len(set(L))
    samples_per_class = L.count(L[0])
    
    #count = num_class * samples_per_class
    #count = 90
    #count = 10
    count = len(L)
    fps = []
    tps = []
    for threshold in thresholds:
        fp = 0
        tp = 0
        pc = 0
        nc = 0
        for i in range(count):
            for j in range(i,count):
                guess = False
                if Cosine_Distance(enc_results_final[i],enc_results_final[j]) <= threshold:
                    guess = True
                if L[i]==L[j]:
                    pc += 1
                    if guess:
                        tp += 1
                else:
                    nc += 1
                    if guess:
                        fp += 1
        fp = fp/nc
        tp = tp/pc
        fps.append(fp)
        tps.append(tp)
        
    data_dict = {"False Positive Rate":fps,"True Positive Rate":tps}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="False Positive Rate",y="True Positive Rate",
                  title=title)
    fig.update_yaxes(range=[0,1])
    
    fig_file_name = "figures/ROC_projected_gamma=" + str(gamma) + "_lambda=" + str(lamb) + ".png"
    fig.write_image(fig_file_name)
    
    auc = 0
    
    auc_dict = {}
    for i in range(len(tps)):
        auc_dict[fps[i]] = tps[i]
    auc_list = list(auc_dict.items())
    auc_list.sort(key=itemgetter(0))
    
    for i in range(1,len(auc_list)):
        interval = abs(auc_list[i][0]-auc_list[i-1][0])
        auc += interval*auc_list[i-1][1]
        auc += 0.5 * interval * (auc_list[i][1]-auc_list[i-1][1])
        
    #print()
    #print(auc_list)
    print("AUC:",auc)
    
    #print(fps)
    #print(tps)
    #print()
    
def ROC_P_Matrix(filename, gamma, lamb, title):
    p_file = open(filename,'r')
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
    
    thresholds = [0.05 * i for i in range(41)]
    thresholds = [0.01 * i for i in range(201)]
    
    #num_class = len(set(L))
    #samples_per_class = L.count(L[0])
    
    test = True
    
    if test:
        a_file = open("data/features_A_values_test.txt",'r')
        b_file = open("data/features_B_values_test.txt",'r')
    else:
        a_file = open("data/features_A_values_val.txt",'r')
        b_file = open("data/features_B_values_val.txt",'r')
    
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
        
    #print("here")
    #l_file = open("data/features_labels_X_prime_test_lambda=0.99_margin=0.5_gamma=128.txt",'r')
    #L = []
    #for line in l_file:
        #L.append(float(line.strip().split(";")[1]))
    
    
    #Create feature fusion dataset
    X = torch.cat((A_final,B_final),dim=1)
    #p_final = torch.mul(p_final,10)
    X_prime = torch.mm(X, p_final.T)
    #X_prime = torch.mm(p_final, X.T)
    
    print()
    print("l2(p):",torch.linalg.norm(p_final))
    
    print()
    for i in range(X_prime.shape[0]):
        print(torch.linalg.norm(X_prime[i,:]))
    print()
    
    print(L)
    print(X_prime.shape)
    
    for i in range(X_prime.shape[0]):
        X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
    print(X_prime[3])
    #X_prime = X_prime.T
    
    
    #count = num_class * samples_per_class
    #count = 90
    #count = 10
    count = len(L)
    fps = []
    tps = []
    for threshold in thresholds:
        fp = 0
        tp = 0
        pc = 0
        nc = 0
        for i in range(count):
            for j in range(i,count):
                guess = False
                if Cosine_Distance(X_prime[i],X_prime[j]) <= threshold:
                    guess = True
                if L[i]==L[j]:
                    pc += 1
                    if guess:
                        tp += 1
                else:
                    nc += 1
                    if guess:
                        fp += 1
        fp = fp/nc
        tp = tp/pc
        fps.append(fp)
        tps.append(tp)
        
    data_dict = {"False Positive Rate":fps,"True Positive Rate":tps}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="False Positive Rate",y="True Positive Rate",
                  title=title)
    fig.update_yaxes(range=[0,1])
    
    fig_file_name = "figures/ROC_projected_gamma=" + str(gamma) + "_lambda=" + str(lamb) + ".png"
    fig.write_image(fig_file_name)
    
    auc = 0
    
    auc_dict = {}
    for i in range(len(tps)):
        auc_dict[fps[i]] = tps[i]
    auc_list = list(auc_dict.items())
    auc_list.sort(key=itemgetter(0))
    
    for i in range(1,len(auc_list)):
        interval = abs(auc_list[i][0]-auc_list[i-1][0])
        auc += interval*auc_list[i-1][1]
        auc += 0.5 * interval * (auc_list[i][1]-auc_list[i-1][1])
        
    #print()
    #print(auc_list)
    print("AUC:",auc)
    
    #print(fps)
    #print(tps)
    #print()

def ROC_AUC(data, L):
    thresholds = [0.05 * i for i in range(41)]
    thresholds = [0.01 * i for i in range(201)]
    
    count = L.size(0)
    fps = []
    tps = []
    for threshold in thresholds:
        fp = 0
        tp = 0
        pc = 0
        nc = 0
        for i in range(count):
            for j in range(i,count):
                guess = False
                if Cosine_Distance(data[i],data[j]) <= threshold:
                    guess = True
                if L[i]==L[j]:
                    pc += 1
                    if guess:
                        tp += 1
                else:
                    nc += 1
                    if guess:
                        fp += 1
        fp = fp/nc
        tp = tp/pc
        fps.append(fp)
        tps.append(tp)
        
    """
    data_dict = {"False Positive Rate":fps,"True Positive Rate":tps}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="False Positive Rate",y="True Positive Rate",
                  title=title)
    fig.update_yaxes(range=[0,1])
    
    fig_file_name = "figures/ROC_projected_gamma=" + str(gamma) + "_lambda=" + str(lamb) + ".png"
    fig.write_image(fig_file_name)
    """
    auc = 0
    
    auc_dict = {}
    for i in range(len(tps)):
        auc_dict[fps[i]] = tps[i]
    auc_list = list(auc_dict.items())
    auc_list.sort(key=itemgetter(0))
    
    for i in range(1,len(auc_list)):
        interval = abs(auc_list[i][0]-auc_list[i-1][0])
        auc += interval*auc_list[i-1][1]
        auc += 0.5 * interval * (auc_list[i][1]-auc_list[i-1][1])
    return auc
    
    
def ROC_Labels(filename, L=None):
    enc_results_file = open(filename,'r')
    enc_results = []
    for line in enc_results_file:
        result = line.strip()
        #print(result)
        result = torch.tensor([float(i) for i in result.split()])
        #result = torch.tensor(ast.literal_eval(result)).unsqueeze(dim=0)
        #print(result)
        enc_results.append(result)
    enc_results_final = torch.Tensor(len(enc_results),enc_results[0].shape[0])
    torch.stack(enc_results, out=enc_results_final,dim=0)
    #if L is None:
        #enc_results_final = enc_results_final.T
    print(enc_results_final.shape)
    
    print(enc_results_final[3])
    thresholds = [0.01 * i for i in range(201)]
    
    """if L is None:
        L = np.load("data/features/MMU_label_gallery.npy")
        L = torch.tensor(L)
        L = L[:,0]
        num_samples = L.shape[0]
        num_each_class = 8
        num_classes = 45
        num_each_class = 2
        
        split1 = math.floor(num_classes * 0.6)*num_each_class
        split2 = math.floor(num_classes * 0.8)*num_each_class
        L = L[split1:]
        """
    if L is None:
        a_file = open("data/features_A_values_val.txt",'r')
        A = []
        L = []
        for line in a_file:
            line, l = line.strip().split(";")
            L.append(int(l))
    num_class = len(set(L))
    
    print(L)
    
    #necessary?
    #for i in range(enc_results_final.shape[0]):
        #enc_results_final[i,:]=torch.div(enc_results_final[i,:], torch.linalg.norm(enc_results_final[i,:]))
    
    count = len(L)
    fps = []
    tps = []
    for threshold in thresholds:
        fp = 0
        tp = 0
        pc = 0
        nc = 0
        for i in range(count):
            for j in range(i,count):
                guess = False
                if Cosine_Distance_no_div(enc_results_final[i],enc_results_final[j]) <= threshold:
                    guess = True
                if L[i]==L[j]:
                    pc += 1
                    if guess:
                        tp += 1
                else:
                    nc += 1
                    if guess:
                        fp += 1
        fp = fp/nc
        tp = tp/pc
        fps.append(fp)
        tps.append(tp)
    
    
    data_dict = {"False Positive Rate":fps,"True Positive Rate":tps}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="False Positive Rate",y="True Positive Rate",
                  title="test")
    fig.update_yaxes(range=[0,1])
    
    fig_file_name = "figures/test_ROC.png"
    fig.write_image(fig_file_name)
    
    auc = 0
    
    auc_dict = {}
    for i in range(len(tps)):
        auc_dict[fps[i]] = tps[i]
    auc_list = list(auc_dict.items())
    auc_list.sort(key=itemgetter(0))
    
    for i in range(1,len(auc_list)):
        interval = abs(auc_list[i][0]-auc_list[i-1][0])
        auc += interval*auc_list[i-1][1]
        auc += 0.5 * interval * (auc_list[i][1]-auc_list[i-1][1])
        
    #print()
    #print(auc_list)
    print("AUC:",auc)

def ROC_Labels_Test(filename, L=None):
    enc_results_file = open(filename,'r')
    enc_results = []
    for line in enc_results_file:
        result = line.strip()
        #print(result)
        result = torch.tensor([float(i) for i in result.split()])
        #result = torch.tensor(ast.literal_eval(result)).unsqueeze(dim=0)
        #print(result)
        enc_results.append(result)
    enc_results_final = torch.Tensor(len(enc_results),enc_results[0].shape[0])
    torch.stack(enc_results, out=enc_results_final,dim=0)
    #if L is None:
        #enc_results_final = enc_results_final.T
    print(enc_results_final.shape)
    
    print(enc_results_final[3])
    thresholds = [0.01 * i for i in range(201)]
    
    """if L is None:
        L = np.load("data/features/MMU_label_gallery.npy")
        L = torch.tensor(L)
        L = L[:,0]
        num_samples = L.shape[0]
        num_each_class = 8
        num_classes = 45
        num_each_class = 2
        
        split1 = math.floor(num_classes * 0.6)*num_each_class
        split2 = math.floor(num_classes * 0.8)*num_each_class
        L = L[split1:]
        """
    if L is None:
        a_file = open("data/features_A_values_test.txt",'r')
        A = []
        L = []
        for line in a_file:
            line, l = line.strip().split(";")
            L.append(int(l))
    num_class = len(set(L))
    
    print(L)
    
    #necessary?
    #for i in range(enc_results_final.shape[0]):
        #enc_results_final[i,:]=torch.div(enc_results_final[i,:], torch.linalg.norm(enc_results_final[i,:]))
    
    count = len(L)
    fps = []
    tps = []
    for threshold in thresholds:
        fp = 0
        tp = 0
        pc = 0
        nc = 0
        for i in range(count):
            for j in range(i,count):
                guess = False
                if Cosine_Distance_no_div(enc_results_final[i],enc_results_final[j]) <= threshold:
                    guess = True
                if L[i]==L[j]:
                    pc += 1
                    if guess:
                        tp += 1
                else:
                    nc += 1
                    if guess:
                        fp += 1
        fp = fp/nc
        tp = tp/pc
        fps.append(fp)
        tps.append(tp)
    
    
    data_dict = {"False Positive Rate":fps,"True Positive Rate":tps}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="False Positive Rate",y="True Positive Rate",
                  title="test")
    fig.update_yaxes(range=[0,1])
    
    fig_file_name = "figures/test_ROC.png"
    fig.write_image(fig_file_name)
    
    auc = 0
    
    auc_dict = {}
    for i in range(len(tps)):
        auc_dict[fps[i]] = tps[i]
    auc_list = list(auc_dict.items())
    auc_list.sort(key=itemgetter(0))
    
    for i in range(1,len(auc_list)):
        interval = abs(auc_list[i][0]-auc_list[i-1][0])
        auc += interval*auc_list[i-1][1]
        auc += 0.5 * interval * (auc_list[i][1]-auc_list[i-1][1])
        
    #print()
    #print(auc_list)
    print("AUC:",auc)


if __name__ == "__main__":
    
    ROC("data/features_A_values_test.txt", "A", "ROC - MMU Iris Resnet 1024-dimensional Features")
    ROC("data/features_B_values_test.txt", "B", "ROC - MMU Iris VGG 512-dimensional Features")
    ROC("data/features_X_values_test.txt", "X", "ROC - MMU Iris 1536-dimensional Concatenated Features")
    """
    #ROC2("data/features_labels_best_P_value_transpose_lambda=0.5_margin=0.5_gamma=2.txt", 2, "ROC Projected Dataset γ=2 (Normalized)")
    #ROC2("data/features_labels_best_P_value_transpose_lambda=0.5_margin=0.5_gamma=256.txt", 256, 0.5, "ROC Projected Dataset γ=256 (Normalized) λ=0.5")
    ##ROC2("data/features_labels_X_prime_test_lambda=0.25_margin=0.5_gamma=256.txt", 256, 0.25, "ROC Projected Test Dataset γ=256 (Normalized) λ=0.25")
    #ROC2("data/features_labels_best_P_value_transpose_lambda=0.75_margin=0.5_gamma=256.txt", 256, 0.75, "ROC Projected Dataset γ=256 (Normalized) λ=0.75")
    #ROC2("data/features_labels_best_P_value_transpose_lambda=0.99_margin=0.5_gamma=256.txt", 256, 0.99, "ROC Projected Dataset γ=256 (Normalized) λ=0.99")
    #ROC2("data/features_labels_best_P_value_transpose_lambda=0.5_margin=0.5_gamma=512.txt", 512, "ROC Projected Dataset γ=512 (Normalized)")
    print("lambda = 128")
    ROC_P_Matrix("data/features_best_P_value_transpose_lambda=0.1_margin=0.25_gamma=128.txt", 128, 0.1, "ROC Projected Test Dataset γ=128 (Normalized) λ=0.1, margin=0.25")
    print("lambda = 256")
    ROC_P_Matrix("data/features_best_P_value_transpose_lambda=0.1_margin=0.25_gamma=256.txt", 256, 0.1, "ROC Projected Test Dataset γ=256 (Normalized) λ=0.1, margin=0.25")
    #ROC_P_Matrix("data/features_best_P_value_transpose_lambda=0.25_margin=0.5_gamma=256.txt", 256, 0.25, "ROC Projected Test Dataset γ=256 (Normalized) λ=0.25, margin=0.5")
    """
    ROC_P_Matrix("data/features_best_P_value_transpose_lambda=0.1_margin=0.25_gamma=128_reg=0.txt", 128, 0.1, "ROC Projected Test Dataset γ=128 (Normalized) λ=0.1, margin=0.25")
    
    #ROC_Labels("data/features_best_P_value_diagonal_lambda=0.1_margin=0.25_gamma=128_reg=0.1.txt",[13, 13, 40, 40,  3,  3, 24, 24, 29, 29, 34, 34, 36, 36, 15, 15, 33, 33, 4, 4])
    #ROC_Labels("data/normalized_encrypted_results_lambda=0.1_margin=0.25_gamma=128.txt")
    ROC_Labels_Test("data/normalized_encrypted_results_test_lambda=0.1_margin=0.25_gamma=128.txt")
    #ROC_Labels("data/normalized_encrypted_results_goldschmidt_lambda=0.1_margin=0.25_gamma=128.txt")
    #ROC_Labels("data/encrypted_results_lambda=0.1_margin=0.25_gamma=128.txt")
