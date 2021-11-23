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
    loss_file = open("data/loss_values.txt",'r')
    losses = []
    for line in loss_file:
        losses.append(float(line.strip()))
    epochs = [i for i in range(len(losses))]
    data_dict = {"Epoch":epochs,"Loss":losses}
    df = pandas.DataFrame(data_dict)

    fig = px.line(df,x="Epoch",y="Loss",
                  title="Train Loss")
    if not os.path.exists("figures"):
        os.mkdir("figures")
    if not os.path.exists("figures/loss"):
        os.mkdir("figures/loss")
    fig.write_image("figures/loss/train_loss.png")


def plot_dataset():
    if not os.path.exists("figures"):
        os.mkdir("figures")
    if not os.path.exists("figures/original_dataset"):
        os.mkdir("figures/original_dataset")
        
    a_file = open("data/A_values.txt",'r')
    A = []
    for line in a_file:
        a = torch.tensor(ast.literal_eval(line.strip())).unsqueeze(dim=0)
        A.append(a)
    A_final = torch.Tensor(len(A),A[0].shape[0])
    torch.cat(A, out=A_final,dim=0)
    
    b_file = open("data/B_values.txt",'r')
    B = []
    for line in b_file:
        b = torch.tensor(ast.literal_eval(line.strip())).unsqueeze(dim=0)
        B.append(b)
    B_final = torch.Tensor(len(B),B[0].shape[0])
    torch.cat(B, out=B_final,dim=0)
        
    l_file = open("data/L_values.txt",'r')
    L = []
    for line in l_file:
        L.append(int(float(line.strip())))
        
        
    
    data_dictA = {"a1":A_final[:,0],"a2":A_final[:,1],"a3":A_final[:,2]," ":["class "+str(l) for l in L]}
    dfA = pandas.DataFrame(data_dictA)

    figA = px.scatter_3d(dfA,x="a1",y="a2",z="a3",color=" ")
    
    
    num_class = len(set(L))
    samples_per_class = L.count(L[0])
    colors = list(set(L))
    colors = [3*color for color in colors]
    titles = ["class "+str(i) for i in range(num_class)]
    
    figB = go.Figure()
    for i in range(num_class):
        figB.add_scatter(name=titles[i],x=B_final[i*samples_per_class:i*samples_per_class+samples_per_class,0],y=B_final[i*samples_per_class:i*samples_per_class+samples_per_class,1],mode="markers",marker={'size': 15,'color': colors[i]})#,legendgrouptitle={'text':titles[i]})
    

    figA.write_image("figures/original_dataset/figA.png")
    figB.write_image("figures/original_dataset/figB.png")

def plot_p():
    
    if not os.path.exists("figures"):
        os.mkdir("figures")
    if not os.path.exists("figures/projected_dataset"):
        os.mkdir("figures/projected_dataset")
    
    p_file = open("data/P_values.txt",'r')
    p_values = []
    for line in p_file:
        P = torch.tensor(ast.literal_eval(line.strip()))
        p_values.append(P)
        
    a_file = open("data/A_values.txt",'r')
    A = []
    for line in a_file:
        a = torch.tensor(ast.literal_eval(line.strip())).unsqueeze(dim=0)
        A.append(a)
    A_final = torch.Tensor(len(A),A[0].shape[0])
    torch.cat(A, out=A_final,dim=0)
    
    b_file = open("data/B_values.txt",'r')
    B = []
    for line in b_file:
        b = torch.tensor(ast.literal_eval(line.strip())).unsqueeze(dim=0)
        B.append(b)
    B_final = torch.Tensor(len(B),B[0].shape[0])
    torch.cat(B, out=B_final,dim=0)
        
    l_file = open("data/L_values.txt",'r')
    L = []
    for line in l_file:
        L.append(float(line.strip()))
    
    
    #Create feature fusion dataset
    X = torch.cat((A_final,B_final),dim=1)

    circle_x = [c*0.005 for c in range(-200,201)]
    circle_y_pos = [(1-c**2)**0.5 for c in circle_x]
    circle_y_neg = [-1*(1-c**2)**0.5 for c in circle_x]
    
    circle_x_final = circle_x+circle_x[::-1]
    circle_y = circle_y_pos+circle_y_neg

    for j in range(len(p_values)):
        X_prime = torch.mm(X,p_values[j])
        for i in range(X_prime.shape[0]):
            X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
        
        
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                mode='lines',
                x=circle_x_final,
                y=circle_y,
                marker={'color':'black'},
                showlegend=False
            )   
        )
        num_class = len(set(L))
        samples_per_class = L.count(L[0])
        colors = list(set(L))
        colors = [3*color for color in colors]
        titles = ["class "+str(i) for i in range(num_class)]
        for i in range(num_class):
            fig.add_scatter(name=titles[i],x=X_prime[i*samples_per_class:i*samples_per_class+samples_per_class,0],y=X_prime[i*samples_per_class:i*samples_per_class+samples_per_class,1],mode="markers",marker={'size': 15,'color': colors[i]})#,legendgrouptitle={'text':titles[i]})
        fig.update_yaxes(scaleanchor="x",scaleratio=1)
        fig.update_xaxes(range=[-1.1,1.1],constrain="domain")
        fig.update_yaxes(scaleanchor = "x",scaleratio = 1)    
        
        fig.write_image("figures/projected_dataset/fig"+str(j)+".png")


if __name__ == "__main__":
    plot_dataset()
    #plot_loss()
    #plot_p()