# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 11:06:56 2022

@author: Luke
"""

import torch
import ast


def diagonal_encoding(lamb, margin):
    p_file_name = "data/P_values_lambda=" + str(lamb) + "_margin=" + str(margin) + ".txt"
    p_file = open(p_file_name,'r')
    p_values = []
    for line in p_file:
        P = torch.tensor(ast.literal_eval(line.strip()))
        p_values.append(P)
    p_matrix = torch.stack(p_values)
    p_matrix = p_matrix[0]
    print(p_matrix.shape)
    max_dim = max(p_values[0].shape[0], p_values[0].shape[1])
    new_max_dim = 0
    for i in range(100):
        if 2**i >= max_dim:
            new_max_dim = 2**i
            break
    p_final = []
    for i in range(new_max_dim):
        temp = []
        for j in range(new_max_dim):
            if i >= p_values[0].shape[0] or j >= p_values[0].shape[1]:
                temp.append(0)
            else:
                temp.append(float(p_matrix[i][j]))
        p_final.append(torch.tensor(temp))
    p_final = torch.stack(p_final)
        
    print(p_final)
        
        
        
if __name__ == "__main__":
    lamb = 0.5
    margin = 0.5
    diagonal_encoding(lamb, margin)