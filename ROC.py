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
        coeffs = [[-14.87368246,23.74576715,-13.66592657,4.17688396]]
    elif degree == 2:
        coeffs = [[5.91965872,-7.3475699,3.54940138]]
    elif degree == 1:
        coeffs = [[-2.61776258,2.78221164]]
    
    
    
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


def New_ROC(filename, title, labels=True):
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
        a_file = open("data/features_L_values_val.txt",'r')
        L = a_file.readline()
        L = [int(i) for i in L[1:len(L)-2].split(", ")]
    
    #for i in range(18):
        #print(torch.linalg.norm(enc_results_final[i]))
    
    y_score = []
    y_true = []
    count = len(L)
    for i in range(count):
        for j in range(i,count):
            score = Cosine_Similarity(enc_results_final[i],enc_results_final[j])
            if L[i]==L[j]:
                label = 1
            else:
                label = 0
            y_score.append(score)
            y_true.append(label)

    
    
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
    
def New_ROC_Encrypted(filename, title, labels=True):
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
            result = [float(item) for item in line.strip().split()]
            result = torch.tensor(result).unsqueeze(dim=0)
            #print(result.shape)
            enc_results.append(result)
        enc_results_final = torch.Tensor(len(enc_results),enc_results[0].shape[1])
        
        torch.cat(enc_results, out=enc_results_final,dim=0)
        
        print(enc_results_final.shape)
        #for i in range(enc_results_final.shape[0]):
            #print(enc_results_final[i,0])
        a_file = open("data/features_L_values_val.txt",'r')
        L = a_file.readline().strip()
        L = [int(i) for i in L[1:len(L)-2].split(", ")]
    
    y_score = []
    y_true = []
    
    results = []
    
    count = len(L)
    for i in range(count):
        
        for j in range(i,count):
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
    
    
    fpr, tpr, thresholds = roc_curve(y_true,y_score)
    
    data_dict = {"False Positive Rate":fpr,"True Positive Rate":tpr}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="False Positive Rate",y="True Positive Rate",
                  title=title)
    fig.update_yaxes(range=[0,1])
    
    fig_file_name = "figures/" + title + ".png"
    fig.write_image(fig_file_name)
    
    return y_score, results
    

def New_ROC_AUC(data, L):
    y_score = []
    y_true = []
    count = len(L)
    for i in range(count):
        for j in range(i,count):
            score = Cosine_Similarity_no_div(data[i,:],data[j,:])
            if L[i]==L[j]:
                label = 1
            else:
                label = 0
            y_score.append(score.detach().numpy())
            y_true.append(label)
    auc = roc_auc_score(y_true,y_score)
    return auc    



def New_ROC_P_Matrix(filename, gamma, lamb, title, poly_degree=None):
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
        a_file = open("data/features_A_values_test.txt",'r')
        b_file = open("data/features_B_values_test.txt",'r')
    else:
        a_file = open("data/features_A_values_val.txt",'r')
        b_file = open("data/features_B_values_val.txt",'r')
    
    A = []
    L = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
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
            score = Cosine_Similarity_no_div(X_prime[i],X_prime[j])
            if L[i]==L[j]:
                label = 1
            else:
                label = 0
            y_score.append(score.detach().numpy())
            y_true.append(label)
    auc = roc_auc_score(y_true,y_score)
    print("AUC:",auc)
    
    
    


#Cosine_Similarity_no_div
if __name__ == "__main__":
    
    """
    #ROC("data/features_A_values_test.txt", "A", "ROC - MMU Iris Resnet 1024-dimensional Features")
    print()
    New_ROC("data/features_A_values_val.txt")
    New_ROC("data/features_B_values_val.txt")
    New_ROC("data/features_X_values_val.txt")
    New_ROC("data/approx_labels_X_prime_val_lambda=0.1_margin=0.5_gamma=256_reg=0.txt")
    New_ROC_Encrypted("data/approximate_labels_X_prime_val_lambda=0.1_margin=0.5_gamma=256_reg=0.txt")
    #New_ROC_Encrypted("results/normalized_encrypted_results_val_lambda=0.1_margin=0.5_gamma=256.txt",labels=False)
    print()
    """
    
    print()
    print("A")
    New_ROC("data/features_A_values_test.txt","ROC_A",False)
    print()
    print("B")
    New_ROC("data/features_B_values_test.txt","ROC_B",False)
    print()
    print("X")
    New_ROC("data/features_X_values_test.txt","ROC_X",False)
    print()
    #New_ROC("data/approx_labels_X_prime_test_lambda=0.1_margin=0.5_gamma=256_reg=0.txt")
    #New_ROC_Encrypted("data/approximate_labels_X_prime_test_lambda=0.1_margin=0.5_gamma=64_reg=0.txt")
    print("Plaintext exact:")
    ay1, ar1 =New_ROC_Encrypted("data/exact_labels_X_prime_test_lambda=0.01_margin=0.25_gamma=64_reg=0.txt","ROC_Algo=Exact_Enc=False",True)
    print()
    print("Plaintext poly - degree=6:")
    y1, r1 = New_ROC_Encrypted("data/approximate_labels_X_prime_test_lambda=0.01_margin=0.25_gamma=64_reg=0.txt","ROC_Algo=Poly6_Enc=False",True)
    print()
    
    
    #print("Plaintext poly - degree=6 large:")
    #y1, r1 = New_ROC_Encrypted("data/degree=6large_approximate_labels_X_prime_val_lambda=0.5_margin=0.25_gamma=64_reg=0.txt","ROC_Algo=Poly6_Enc=False",True)
    #print() 0.466
    
    #print("Plaintext poly - degree=3:")
    #New_ROC_Encrypted("data/1approximate_labels_X_prime_test_lambda=0.1_margin=0.75_gamma=64_reg=0.txt","ROC_Algo=Poly3_Enc=False",True)
    #approximate_best_P_value_transpose_lambda=0.1_margin=1.0_gamma=64_reg=0.txt
    #print()
    
    #print("Plaintext poly - degree=3, seed=1:")
    #New_ROC_Encrypted("data/degree=3b_approximate_labels_X_prime_test_lambda=0.01_margin=0.1_gamma=64_reg=0.txt","ROC_Algo=Poly3_again_Enc=False",True)
    #approximate_best_P_value_transpose_lambda=0.1_margin=1.0_gamma=64_reg=0.txt
    #print()
    print("Plaintext poly - degree=3, strict:")
    New_ROC_Encrypted("data/degree=3strict_approximate_labels_X_prime_test_lambda=0.01_margin=0.1_gamma=64_reg=0.txt","ROC_Algo=Poly3strict_Enc=False",True)
    print("Plaintext poly - degree=3, strict (adjusted):")
    New_ROC_P_Matrix("data/degree=3strict_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt",1,1,"title",3)
    #approximate_best_P_value_transpose_lambda=0.1_margin=1.0_gamma=64_reg=0.txt
    print()
    #degree=3strict
    
    
    print("Plaintext poly - degree=2:")
    New_ROC_Encrypted("data/2approximate_labels_X_prime_test_lambda=0.1_margin=0.75_gamma=64_reg=0.txt","ROC_Algo=Poly2_Enc=False",True)
    print()
    
    print("Plaintext poly - degree=2, strict:")
    New_ROC_Encrypted("data/degree=2strict_approximate_labels_X_prime_test_lambda=0.01_margin=0.1_gamma=64_reg=0.txt","ROC_Algo=Poly2strict_Enc=False",True)
    New_ROC_P_Matrix("data/degree=2strict_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt",1,1,"title", 2) #this method is wrong because it normalizes
    print()
    
    
    print("Plaintext poly - degree=1strict:")
    New_ROC_Encrypted("data/degree=1strict_approximate_labels_X_prime_test_lambda=0.5_margin=0.5_gamma=64_reg=0.txt","ROC_Algo=Poly1_Enc=False",True)
    #data/degree=1_approximate_best_P_value_transpose_lambda=0.1_margin=0.25_gamma=64_reg=0.txt
    print()
    
    
    
    
    
    
    
    print("Encrypted poly - exact training - degree=6:")
    ay2, ar2 = New_ROC_Encrypted("results/normalized_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_EXACT.txt","ROC_Algo=Exact_Enc=Poly6",False)
    print()
    print("Encrypted poly - poly training - degree=6:")
    y2, r2 = New_ROC_Encrypted("results/normalized_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY.txt","ROC_Algo=Poly_Enc=Poly6",False)
    #y2, r2 = New_ROC_Encrypted("results/encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY.txt","ROC_Algo=Poly_Enc=Poly",False)
    print()
    
    print("Encrypted poly - exact training - degree=3:")
    New_ROC_Encrypted("results/normalized_EXACT_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY3.txt","ROC_Algo=Exact_Enc=Poly3",False)
    print()
    print("Encrypted poly - poly training - degree=3, strict:")
    New_ROC_Encrypted("results/normalized_encrypted_results_test_lambda=0.01_margin=0.1_gamma=64_POLY3strict.txt","ROC_Algo=Poly_Enc=Poly3strict",False)
    print()
    print("Encrypted poly - poly training - degree=3, seed=1:")
    New_ROC_Encrypted("results/normalized_encrypted_results_test_lambda=0.01_margin=0.1_gamma=64_POLY3b.txt","ROC_Algo=Poly_Enc=Poly3",False)
    print()
    
    
    print("Encrypted poly - exact training - degree=2:")
    New_ROC_Encrypted("results/normalized_EXACT_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY2.txt","ROC_Algo=Exact_Enc=Poly2",False)
    print()
    print("Encrypted poly - poly training - degree=2:")
    New_ROC_Encrypted("results/normalized_encrypted_results_test_lambda=0.1_margin=0.75_gamma=64_POLY2.txt","ROC_Algo=Poly_Enc=Poly2",False)
    print()
    print("Encrypted poly - poly training - degree=2 part B:")
    New_ROC_Encrypted("results/normalized_encrypted_results_test_lambda=0.1_margin=0.75_gamma=64_POLY2B.txt","ROC_Algo=Poly_Enc=Poly2B",False)
    print()
    
    print("Encrypted poly - poly training - degree=2 replicated, anneal=1:")
    New_ROC_Encrypted("results/normalized_encrypted_results_test_lambda=0.1_margin=0.75_gamma=64_POLY2replicated.txt","ROC_Algo=Poly_Enc=Poly2_repl",False)
    print()
    
    
    
    print("Encrypted poly - exact training - degree=1:")
    New_ROC_Encrypted("results/normalized_EXACT_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY1.txt","ROC_Algo=Exact_Enc=Poly1",False)
    print()
    print("Encrypted poly - poly training - degree=1:")
    New_ROC_Encrypted("results/normalized_encrypted_results_test_lambda=0.5_margin=0.5_gamma=64_POLY1strict.txt","ROC_Algo=Poly_Enc=Poly1strict",False)
    print()
    
    #normalized_EXACT_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY2.txt
    #normalized_encrypted_results_test_lambda=0.1_margin=0.75_gamma=64_POLY2.txt
    
    print("Encrypted gold - exact training:")
    New_ROC_Encrypted("results/normalized_encrypted_results_goldschmidt_test_lambda=0.01_margin=0.25_gamma=64.txt","ROC_Algo=Exact_Enc=Gold",False)
    print()
    
    #New_ROC_P_Matrix("data/1approximate_best_P_value_transpose_lambda=0.25_margin=1.0_gamma=64_reg=0.txt",1,1,"test")
    
    """
    #for i in range(len(r1)):
        #print(r1[i]-r2[i])
    large = 0
    for i in range(len(r1)):
        #print(float(y1[i]),float(y2[i]))
        #print(abs(abs(float(y1[i]))-abs(float(y2[i]))))
        if abs(abs(float(r1[i][0]))-abs(float(r2[i][0]))) >= 0.01:
            large += 1
            print(torch.linalg.norm(r1[i][1]))
            print(torch.linalg.norm(r2[i][1]))
            print(torch.linalg.norm(r1[i][2]))
            print(torch.linalg.norm(r2[i][2]))
            print(float(r1[i][0]), float(r2[i][0]))
            print()
    print("large errors:",large)
    """
    
    #New_ROC_P_Matrix("data/approx_best_P_value_transpose_lambda=0.99_margin=0.25_gamma=256_reg=0.txt", 256, 0.99, "title")
    print()

