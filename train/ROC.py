"""
Author: Luke Sperling
Created: 04-04-22
Modified: 07-07-22
Calculates AUROC for a given set of data and labels.
"""

import torch
#import plotly.express as px
#import plotly.graph_objects as go
#import pandas
#import ast
#from operator import itemgetter
#import numpy as np
#import math

#from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score

def Cosine_Distance_no_div(vec1, vec2):
    #assumes vec1 and vec2 are unit vectors
    return 1 - torch.dot(vec1,vec2)
    
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

