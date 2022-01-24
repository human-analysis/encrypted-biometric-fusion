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
import imageio
import numpy as np

import model

def plot_loss():
    
    lambs = [0.1,0.25,0.5,0.75,0.9]
    margin = 0.5
    
    if not os.path.exists("figures"):
        os.mkdir("figures")
    if not os.path.exists("figures/loss"):
        os.mkdir("figures/loss")
    
    
    for lamb in lambs:
        loss_file_name = "data/loss_values_lambda=" + str(lamb) + "_margin=" + str(margin) + ".txt"
        loss_file = open(loss_file_name,'r')
        losses = []
        for line in loss_file:
            losses.append(float(line.strip()))
        epochs = [i for i in range(len(losses))]
        data_dict = {"Epoch":epochs,"Loss":losses}
        df = pandas.DataFrame(data_dict)
    
        fig = px.line(df,x="Epoch",y="Loss",
                      title="Train Loss")
        
        fig_file_name = "figures/loss/train_loss_lambda=" + str(lamb) + "_margin=" + str(margin) + ".png"
        fig.write_image(fig_file_name)


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
    
    lambs = [0.1,0.25,0.5,0.75,0.9]
    margin = 0.5
    
    if not os.path.exists("figures"):
        os.mkdir("figures")
    if not os.path.exists("figures/projected_dataset"):
        os.mkdir("figures/projected_dataset")
        
    for lamb in lambs:
        if not os.path.exists("figures/projected_dataset/lambda=" + str(lamb) + "_margin=" + str(margin)):
            os.mkdir("figures/projected_dataset/lambda=" + str(lamb) + "_margin=" + str(margin))
        filenames = [] #this will be for creating the animation at the end
        p_file_name = "data/P_values_lambda=" + str(lamb) + "_margin=" + str(margin) + ".txt"
        p_file = open(p_file_name,'r')
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
            fig = go.Figure(layout = go.Layout(title = go.layout.Title(text="Lambda: "+str(lamb))))
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
            
            fig_file_name = "figures/projected_dataset/lambda=" + str(lamb) + "_margin=" + str(margin) + "/" + str(j) + ".png"
            filenames.append(fig_file_name)
            fig.write_image(fig_file_name)
        
        #this portion from https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        gif_file_name = "animations/lambda=" + str(lamb) + "_margin=" + str(margin) + ".gif"
        imageio.mimsave(gif_file_name, images)
        #end portion
    
def combine_gifs():
    #this function modified from https://stackoverflow.com/questions/51517685/combine-several-gif-horizontally-python
    
    lambs = [0.1,0.25,0.5,0.75,0.9]
    margin = 0.5
    
    gifs = []
    
    for lamb in lambs:
        gif_file_name = "animations/lambda=" + str(lamb) + "_margin=" + str(margin) + ".gif"
        gif = imageio.get_reader(gif_file_name)
        gifs.append(gif)
    
    
    #Create writer object
    new_gif = imageio.get_writer('animations/combined.gif')
    number_of_frame = 100
    for frame_number in range(number_of_frame):
        imgs = []
        for i in range(len(gifs)):
            imgs.append(gifs[i].get_next_data())
        #img1 = gif1.get_next_data()
        #img2 = gif2.get_next_data()
        #here is the magic
        new_image = np.hstack(imgs)
        new_gif.append_data(new_image)
        if frame_number == 0 or frame_number == 99:
            for i in range(20):
                new_gif.append_data(new_image)
    
    for gif in gifs:
        gif.close()
    new_gif.close()
    
def plot_results():
    
    if not os.path.exists("figures"):
        os.mkdir("figures")
    if not os.path.exists("figures/results"):
        os.mkdir("figures/results")
    
    circle_x = [c*0.005 for c in range(-200,201)]
    circle_y_pos = [(1-c**2)**0.5 for c in circle_x]
    circle_y_neg = [-1*(1-c**2)**0.5 for c in circle_x]
    
    circle_x_final = circle_x+circle_x[::-1]
    circle_y = circle_y_pos+circle_y_neg


    enc_results_file = open("results/toy_data_1_3.txt",'r')
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
    
    pln_results_file = open("results/toy_data_1_3_plain.txt",'r')
    pln_results = []
    L = []
    for line in pln_results_file:
        result, l = line.strip().split(";")
        result = torch.tensor(ast.literal_eval(result)).unsqueeze(dim=0)
        l = int(l)
        pln_results.append(result)
        L.append(l)
    pln_results_final = torch.Tensor(len(pln_results),pln_results[0].shape[0])
    torch.cat(pln_results, out=pln_results_final,dim=0)

    #encrypted
    fig = go.Figure(layout = go.Layout(title = go.layout.Title(text="Encrypted-space Toy Data")))
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
        fig.add_scatter(name=titles[i],x=enc_results_final[i*samples_per_class:i*samples_per_class+samples_per_class,0],y=enc_results_final[i*samples_per_class:i*samples_per_class+samples_per_class,1],mode="markers",marker={'size': 15,'color': colors[i]})#,legendgrouptitle={'text':titles[i]})
    fig.update_yaxes(scaleanchor="x",scaleratio=1)
    fig.update_xaxes(range=[-1.1,1.1],constrain="domain")
    fig.update_yaxes(scaleanchor = "x",scaleratio = 1)    
    
    fig_file_name = "figures/results/toy_data_1_3.png"
    fig.write_image(fig_file_name)
    
    
    #plain
    fig = go.Figure(layout = go.Layout(title = go.layout.Title(text="Message-space Toy Data")))
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
        fig.add_scatter(name=titles[i],x=pln_results_final[i*samples_per_class:i*samples_per_class+samples_per_class,0],y=pln_results_final[i*samples_per_class:i*samples_per_class+samples_per_class,1],mode="markers",marker={'size': 15,'color': colors[i]})#,legendgrouptitle={'text':titles[i]})
    fig.update_yaxes(scaleanchor="x",scaleratio=1)
    fig.update_xaxes(range=[-1.1,1.1],constrain="domain")
    #fig.update_xaxes(scaleanchor = "y",scaleratio = 1)    
    
    fig_file_name = "figures/results/toy_data_1_3_plain.png"
    fig.write_image(fig_file_name)

if __name__ == "__main__":
    #plot_dataset()
    #plot_loss()
    #plot_p()
    #combine_gifs()
    plot_results()
