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
    
def plot_poly_results():
    
    if not os.path.exists("figures"):
        os.mkdir("figures")
    if not os.path.exists("figures/results"):
        os.mkdir("figures/results")
    
    circle_x = [c*0.005 for c in range(-200,201)]
    circle_y_pos = [(1-c**2)**0.5 for c in circle_x]
    circle_y_neg = [-1*(1-c**2)**0.5 for c in circle_x]
    
    circle_x_final = circle_x+circle_x[::-1]
    circle_y = circle_y_pos+circle_y_neg


    enc_results_file = open("results/toy_data_polynomial.txt",'r')
    #enc_results_file = open("results/toy_data_gold_1.txt",'r')
    enc_results = []
    L = []
    for line in enc_results_file:
        result, l = line.strip().split(";")
        result = [float(val) for val in result.split()]
        print(result)
        result = torch.tensor(result).unsqueeze(dim=0)
        l = int(l)
        enc_results.append(result)
        L.append(l)
    enc_results_final = torch.Tensor(len(enc_results),enc_results[0].shape[0])
    torch.cat(enc_results, out=enc_results_final,dim=0)
    
    
    #encrypted
    fig = go.Figure(layout = go.Layout(title = go.layout.Title(text="Encrypted-space Toy Data (Polynomial)")))
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
    
    fig_file_name = "figures/results/toy_data_polynomial.png"
    fig.write_image(fig_file_name)
    
def plot_errors():
    """plots errors vs multiplicative depth"""
    lamb = 0.5
    margin = 0.5

    p_file_name = "data/best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + ".txt"
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
        L.append(int(float(line.strip())))


    #Create feature fusion dataset
    X = torch.cat((A_final,B_final),dim=1)

    X_prime = torch.mm(X,torch.transpose(p_values[0],0,1))
    #1 for the mult
    #1 for the finding squared norm
    worst_errors = []

    #(enc_mults,plain_mults) = total_mult_depth
    #polynomial = (2,2) = 4
    #inv norm mult depth = (0,2) = 2
    #inv norm but with estimation = ()
    #linear approx = (0,1) = 1 //not used in current build
    #goldschmidt overhead = (x:(1,0),h:(0,1)) = 1
    #goldschmidt per iteration = (2,) = 2
    #goldschmidt ending = (0,1) = 1
    #goldschmidt 1 = 6
    #g 2 = 8
    #g 3 = 10
    #g 4 = 12
    mult_depth = [4, 6, 8, 10, 12]

    file_names = ["results/toy_data_polynomial.txt","results/toy_data_gold_1.txt",
    "results/toy_data_gold_2.txt", "results/toy_data_gold_3.txt", "results/toy_data_gold_4.txt"]
    for file_name in file_names:
        print()
        print(file_name)
        encrypted_method_values = []
        enc_file = open(file_name)
        for line in enc_file:
            line = line.strip().split(";")[0]
            line = line.split()
            encrypted_method_values.append((float(line[0]),float(line[1])))
            

        largest_error = 0
        total_error = 0
        count = 40

        #print(X_prime)
        for i in range(X_prime.shape[0]):
            result = (X_prime[i][0]**2 + X_prime[i][1]**2)**0.5
            true_x = float(X_prime[i][0]/result)
            true_y = float(X_prime[i][1]/result)
            test_x = encrypted_method_values[i][0]
            test_y = encrypted_method_values[i][1]
            dist = ((true_x-test_x)**2+(true_y-test_y)**2)**0.5
            total_error += dist
            if dist > largest_error:
                largest_error = dist
        print("Worst error:",largest_error)
        print("Average error:",total_error/count)
        worst_errors.append(largest_error)
        #print(float(X_prime[i][0]/result), str(float(X_prime[i][1]/result))+";"+str(L[i]))

    print()
    print(worst_errors)
    print(mult_depth)
    
    color1 = "blue"
    color2 = "red"
    color1 = "Polynomial Approx"
    color2 = "Goldschmidt"
    data_dict = {"Worst L2 Error":worst_errors, "Multiplicative Depth":mult_depth, "colors":[color1,color2,color2,color2,color2]}
    df = pandas.DataFrame(data_dict)
    fig = px.scatter(df,x="Multiplicative Depth",y="Worst L2 Error",color="colors")
    
    fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    
    """
    fig = go.Figure(layout = go.Layout(title = go.layout.Title(text="Multiplicative Depth vs Error")))
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=mult_depth,
            y=worst_errors,
            marker={'color':'black'},
            showlegend=False
        )
    )"""
    
    fig_file_name = "figures/MultDepthVsError.png"
    fig.write_image(fig_file_name)
    
def plot_matmul_performance():
    
    #hybrid
    dims = [2,4,8,16,32,64]
    times_actual = [13.311,16.124,20.485,32.765,61.457,107.449]
    data_dict = {"Output Dimensionality":dims, "Time (ms)":times_actual}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="Output Dimensionality",y="Time (ms)", title="Matrix-Vector Multiplication Scaling (Hybrid)")
    
    fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    
    fig_file_name = "figures/HybridPerformance.png"
    fig.write_image(fig_file_name)
    
    #vector rows
    dims = [2,4,8,16,32,64,128]
    times_actual = [0.679,1.358,2.18,4.158,7.506,14.841,29.257]
    data_dict = {"Input Dimensionality times Output Dimensionality":dims, "Time (ms)":times_actual}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="Input Dimensionality times Output Dimensionality",y="Time (ms)", title="Matrix-Vector Multiplication Scaling (Vector Rows)")
    
    fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    
    fig_file_name = "figures/VectorRowsPerformance.png"
    fig.write_image(fig_file_name)

def plot_matmul_performance_theoretical():
    ns = [i for i in range(600)]
    gamma = [2**i for i in range(9,15)]*100
    time = []
    
    for gam in gamma:
        temp = []
        for n in ns:
            temp.append(2*n*gam + n*gam)
        time.append(temp)
    
    print(len(ns),len(gamma),len(time))
    data_dictTime = {"n":ns,"γ":gamma,"Time (ms)":time}

    dfTime = pandas.DataFrame(data_dictTime)
    
    
    #print(data_dictTime["Time (ms)"])
    figTime = px.scatter_3d(dfTime,x="n",y="γ",z="Time (ms)")#,color=" ")
    
    #figTime = go.Figure(go.Surface(x = ns, y = gamma, z = time))

    #this line from https://plotly.com/python/3d-surface-plots/
    """figTime.update_layout(
        scene = {
            "xaxis": {"nticks": 20, "label":"dog"},
            "zaxis": {"nticks": 4},
            #'camera_eye': {"x": 0.5, "y": -1, "z": -0.5},
            "aspectratio": {"x": 1, "y": 1, "z": 0.5}
        })"""
    
    figTime.write_image("figures/TheoreticalTime_Naive.png")

if __name__ == "__main__":
    #plot_dataset()
    #plot_loss()
    #plot_p()
    #combine_gifs()
    #plot_results()
    #plot_poly_results()
    #plot_errors()
    #plot_matmul_performance();
    plot_matmul_performance_theoretical();
