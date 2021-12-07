# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:41:16 2021

@author: Luke
"""

import torch
import ast
import plotly.express as px
import plotly.graph_objects as go
import pandas
import os

from data_generation import data_gen

def distance(x1,x2):
    #assumes x1 and x2 are unit vectors
    return 1-torch.dot(x1, x2)/(torch.linalg.norm(x1)*torch.linalg.norm(x2))

def main():

    if not os.path.exists("figures"):
        os.mkdir("figures")
    if not os.path.exists("figures/ROC"):
        os.mkdir("figures/ROC")

    num_samples = 10
    num_classes = 4
    A,B,L = data_gen(10)
    lamb = 0.1
    margin = 0.5
    
    p_file_name = "data/best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + ".txt"
    p_file = open(p_file_name,'r')
    P = None
    for line in p_file:
        P = torch.tensor(ast.literal_eval(line.strip()))
    
    X = torch.cat((A,B),dim=1)
    
    X_prime = torch.mm(X,torch.transpose(P,0,1)) 
    
    num_items = num_samples*num_classes
    
    sim_matrix = [[0]*num_items]*num_items
    
    for i in range(num_items):
        for j in range(num_items):
            sim_matrix[i][j]=distance(X_prime[i,:],X_prime[j,:])
    print(sim_matrix)
    
    threshs = [0.25, 0.5, 0.75, 1, 1.5, 1.75]
    false_pos_list = []
    true_pos_list = []
    for thresh in threshs:
        false_pos = 0
        true_pos = 0
        for i in range(num_items):
            for j in range(num_items):
                guess = sim_matrix[i][j] > thresh
                truth = int(L[i]) == int(L[j])
                if guess:
                    if truth:
                        true_pos+=1
                    else:
                        false_pos+=1
        false_pos_list.append(false_pos)
        true_pos_list.append(true_pos)
    
    data_dict = {"False Positive Rate":false_pos_list,"True Positive Rate":true_pos_list}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="False Positive Rate",y="True Positive Rate",title="ROC")
    
    fig_file_name = "figures/ROC/ROC_lambda=" + str(lamb) + "_margin=" + str(margin) + ".png"
    fig.write_image(fig_file_name)
                        
if __name__ == "__main__":
    main()
