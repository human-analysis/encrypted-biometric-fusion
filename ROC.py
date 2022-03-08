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
    main()"""



import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas
import ast

from sklearn import metrics



def Cosine_Distance(vec1, vec2):
    #assumes vec1 and vec2 are unit vectors
    return 1 - torch.dot(torch.div(vec1, torch.linalg.norm(vec1)), torch.div(vec2, torch.linalg.norm(vec2)))

def ROC(filename, tag, title):
    
    
    
    
    #enc_results_file = open("results/toy_data_1_4.txt",'r')
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
    
    count = num_class * samples_per_class
    count = 90
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
    
    """
    fig = go.Figure(layout = go.Layout(title = go.layout.Title(text="ROC")))
    fig.add_trace(
        go.Scatter(
            mode='lines',
            x=fps,
            y=tps,
            marker={'color':'black'},
            showlegend=False
        )   
    )
    fig_file_name = "figures/ROC.png"
    fig.write_image(fig_file_name)
    print(fps)
    print(tps)
    """
    print(fps)
    print(tps)

def ROC2(filename, gamma, title):
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
    
    count = num_class * samples_per_class
    count = 90
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
    
    fig_file_name = "figures/ROC_projected_gamma=" + str(gamma) + ".png"
    fig.write_image(fig_file_name)
    
    print(fps)
    print(tps)


if __name__ == "__main__":
    ROC("data/features_A_values.txt", "A", "ROC - MMU Iris Resnet 1024-dimensional Features")
    ROC("data/features_B_values.txt", "B", "ROC - MMU Iris VGG 512-dimensional Features")
    ROC("data/features_X_values.txt", "X", "ROC - MMU Iris 1536-dimensional Concatenated Features")
    ROC2("data/features_labels_best_P_value_transpose_lambda=0.5_margin=0.5_gamma=2.txt", 2, "ROC Projected Dataset γ=2 (Normalized)")
    ROC2("data/features_labels_best_P_value_transpose_lambda=0.5_margin=0.5_gamma=256.txt", 256, "ROC Projected Dataset γ=256 (Normalized)")

