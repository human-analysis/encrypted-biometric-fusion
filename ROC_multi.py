import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas
import ast
from operator import itemgetter
import numpy as np
import math

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from sklearn import metrics

from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt
plt.style.use(['science','no-latex'])


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
    

def ROC_Encrypted_Results(filenames, title, names, labels=True, debug=False):
    fig = go.Figure()
    
    fig1, ax1 = plt.subplots(figsize=(8,6))
    fprs = []
    tprs = []
    
    for iter, filename in enumerate(filenames):
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
            #a_file = open("data/features_L_values_val.txt",'r')
            #a_file = open("data4/dataset/L_values_test.txt",'r')
            a_file = open("data5/dataset/L_values_test.txt",'r')
            L = a_file.readline().strip()
            L = [int(i) for i in L[1:len(L)-2].split(", ")]
            #L = L[:100] #for debugging
    
        y_score = []
        y_true = []
        
        results = []
        
        count = len(L)
        for i in range(count):
            
            for j in range(i+1,count):
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
        
        fig.add_trace(
            go.Scatter(
                mode='lines',
                x=fpr,
                y=tpr,
                name=names[iter]
            )
        )
        
        ##plt.plot(fpr,tpr)
        #ax1.plot(fpr,tpr)
        #axins1.plot(fpr,tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        
        #data_dict = {"False Positive Rate":fpr,"True Positive Rate":tpr}
        #df = pandas.DataFrame(data_dict)
    #fig = px.line(df,x="False Positive Rate",y="True Positive Rate",
                  #title=title)
    fig.update_yaxes(range=[0,1.0])
    fig.update_xaxes(range=[0,1.0])
    
    #fig.update_yaxes(range=[0.4,1.0])
    #fig.update_xaxes(range=[0,0.6])
    
    
    fig.update_layout(legend=dict(
    yanchor="bottom",
    y=-0.4,
    xanchor="left",
    x=0
    ))
    
    
    fig_file_name = "figures/" + title + ".png"
    fig.write_image(fig_file_name)
    
    
    #ax1.plot(fprs,tprs)
    
    for i in range(len(fprs)):
        fpr = fprs[i]
        tpr = tprs[i]
        ax1.plot(fpr,tpr,linewidth=2)
    
    axins1 = zoomed_inset_axes(ax1, zoom = 1.75, loc=5)#loc=5/7?
    for i in range(len(fprs)):
        fpr = fprs[i]
        tpr = tprs[i]
        axins1.plot(fpr,tpr,linewidth=4)
    
    axins1.set_xlim(0.0,0.3)
    axins1.set_xlim(-0.01,0.3)
    axins1.set_ylim(0.7,1.0)
    axins1.set_ylim(0.7,1.01)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    
    
    
    ax1.set_ylabel('True Positive Rate', fontsize=24)
    ax1.set_xlabel('False Positive Rate',fontsize=24)
    ax1.legend(names, loc=0, frameon=True)
    ax1.grid()
    #ax1.set_yticklabels(fontsize=18)
    #ax1.set_xticklabels(fontsize=18)
    
    ax1.tick_params(axis='x', labelsize=18 )
    ax1.tick_params(axis='y', labelsize=18 )
    
    #
    
    mark_inset(ax1,axins1,loc1=1,loc2=3, fc="none",ec="0.5")
    
    plt.show()
    
    return y_score, results


"""
filenames = []
filenames.append("results/normalized_encrypted_results_test_lambda=0.01_margin=0.25_gamma=32_poly2.txt")
filenames.append("results/normalized_encrypted_results_test_lambda=0.01_margin=0.1_gamma=32_exact2.txt")
filenames.append("results/normalized_encrypted_results_test_lambda=0.01_margin=0.1_gamma=32_exact6.txt")
filenames.append("results/normalized_encrypted_results_goldschmidt_lambda=0.01_margin=0.1_gamma=32.txt")
filenames.append("results/averaged_results.txt")
names = ["HEFT Learning, Polynomial (Degree=2) Inference - 0.9508", "Exact Learning, Polynomial (Degree=2) Inference - 0.9180", "Exact Learning, Polynomial (Degree=6) Inference - 0.9577", "Exact Learning, Goldschmidt's Inference - 0.9657", "Averaged - 0.9159"]
ROC_Encrypted_Results(filenames, "ROC_multi", names, labels=False, debug=False)
"""
filenames = []
filenames.append("results/fingerprint_normalized_encrypted_results_test_lambda=0.01_margin=0.1_gamma=32_poly2_.txt")
filenames.append("results/finger_normalized_encrypted_results_test_lambda=0.5_margin=0.1_gamma=32_exact2_.txt")
filenames.append("results/finger_normalized_encrypted_results_test_lambda=0.5_margin=0.1_gamma=32_exact6_.txt")
filenames.append("results/finger_normalized_encrypted_results_goldschmidt_gamma=32.txt")
names = ["HEFT Learning, Polynomial (Degree=2) Inference - 0.9925", 
         "Exact Learning, Polynomial (Degree=2) Inference - 0.9294", 
         "Exact Learning, Polynomial (Degree=6) Inference - 0.9883", 
         "Exact Learning, Goldschmidt's Inference - 0.9980"]
ROC_Encrypted_Results(filenames, "ROC_multi_fingerprint", names, labels=False, debug=False)

