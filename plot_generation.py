# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:14:59 2021

@author: lsper
"""
import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas
import ast
import os
from plotly.subplots import make_subplots

import model

def plot_loss():
    loss_file = open("loss_values.txt",'r')
    losses = []
    for line in loss_file:
        losses.append(float(line.strip()))
    epochs = [i for i in range(len(losses))]
    data_dict = {"Epoch":epochs,"Loss":losses}
    df = pandas.DataFrame(data_dict)
    #df = pandas.DataFrame((epochs,losses),columns=["Epoch","Loss"])
    #fig = px.line(df,x="Epoch", y="Loss")
    fig = px.line(df,x="Epoch",y="Loss",
                  title="Train Loss")
    fig.show()

def plot_p():
    
    #this function not yet working
    
    if not os.path.exists("figures"):
        os.mkdir("figures")
    
    p_file = open("P_values.txt",'r')
    p_values = []
    for line in p_file:
        P = torch.tensor(ast.literal_eval(line.strip()))
        p_values.append(P)
        
    a_file = open("A_values.txt",'r')
    A = []
    for line in a_file:
        a = torch.tensor(ast.literal_eval(line.strip())).unsqueeze(dim=0)
        A.append(a)
    A_final = torch.Tensor(len(A),A[0].shape[0])
    torch.cat(A, out=A_final,dim=0)
    
    b_file = open("B_values.txt",'r')
    B = []
    for line in b_file:
        b = torch.tensor(ast.literal_eval(line.strip())).unsqueeze(dim=0)
        B.append(b)
    B_final = torch.Tensor(len(B),B[0].shape[0])
    torch.cat(B, out=B_final,dim=0)
        
    l_file = open("L_values.txt",'r')
    L = []
    for line in l_file:
        L.append(float(line.strip()))
    
    #Create feature fusion dataset
    X = torch.cat((A_final,B_final),dim=1)
    
    #for i in range(len(p_values)):
    i = -1
    X_prime = torch.mm(X,p_values[i])
    for i in range(X_prime.shape[0]):
        X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
    #X_prime = torch.div(X_prime, torch.linalg.norm(X_prime))
    
    circlex = [c*0.001 for c in range(-100,100)]
    circley_pos = [(1-c**2)**0.5 for c in circlex]
    circley_neg = [-1*(1-c**2)**0.5 for c in circlex]
    
    data_dict = {"x1":X_prime[:,0],"x2":X_prime[:,1],"Labels":L}
    df = pandas.DataFrame(data_dict)
    
    #fig = px.scatter(df,x="x1",y="x2",color="Labels",
    #              title="Projection After Training")
    fig1 = px.line(x=circlex,y=circley_pos,color_discrete_sequence=["black"])
    fig2 = px.scatter(df,x="x1",y="x2",color="Labels",
                  title="Projection After Training")
    fig2.update_yaxes(scaleanchor="x",scaleratio=1)
    
    fig3 = go.Figure()
    fig3.add_traces((df,))
    
    fig3.show()
    
    i=1###########################
    #fig.write_image("images/fig"+i+".svg")
    


if __name__ == "__main__":
    plot_loss()
    #plot_p() #not yet working