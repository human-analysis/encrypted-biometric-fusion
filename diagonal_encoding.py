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
        
    
    
    #p_final = []
    #new_max_dim = 3
    #p_final.append(torch.tensor([1,2,3]))
    #p_final.append(torch.tensor([4,5,6]))
    #p_final.append(torch.tensor([7,8,9]))
    #p_final = torch.stack(p_final)
    
    
    print(p_final)
    print()
    
    
    p_diag = []
    
    i = 0
    j = 0
    for k in range(new_max_dim):
        temp = []
        for l in range(new_max_dim):
            temp.append(float(p_final[j][i]))
            i+=1
            j+=1
            if i >= new_max_dim:
                i = 0
            if j >= new_max_dim:
                j = 0
        p_diag.append(torch.tensor(temp))
        i+=1
    p_diag = torch.stack(p_diag)
    print(p_diag)
    
    
    p_output_file_name = "data/diagonal_P_value_lambda=" + str(lamb) + "_margin=" + str(margin) + ".txt"
    outfile_p = open(p_output_file_name,'w')
    for row in p_diag.tolist():
        for item in row:
            outfile_p.write(str(item))
            outfile_p.write(" ")
        outfile_p.write("\n")
    #P_final = str(p_diag.tolist())
    #outfile_p.write(P_final)
    outfile_p.close()
    
    
def diagonal_encoding_transposed(lamb, margin):
    p_file_name = "data/best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + ".txt"
    p_file = open(p_file_name,'r')
    p_values = []
    for line in p_file:
        P = torch.tensor(ast.literal_eval(line.strip()))
        p_values.append(P)
    p_matrix = torch.stack(p_values)
    p_matrix = p_matrix[0]
    #print(p_matrix.shape)
    #print(p_matrix)
    #print()
    
    max_dim_h = p_values[0].shape[0]
    max_dim_w = p_values[0].shape[1]
    new_max_dim_h = 0
    new_max_dim_w = 0
    for i in range(100):
        if 2**i >= max_dim_h:
            new_max_dim_h = 2**i
            break
    for i in range(100):
        if 2**i >= max_dim_w:
            new_max_dim_w = 2**i
            break
    
    p_final = []
    for i in range(new_max_dim_h):
        temp = []
        for j in range(new_max_dim_w):
            if i >= p_values[0].shape[0] or j >= p_values[0].shape[1]:
                temp.append(0)
            else:
                temp.append(float(p_matrix[i][j]))
        p_final.append(torch.tensor(temp))
    p_final = torch.stack(p_final)
    
    print(p_final)
    print()
    
    p_diag = []
    #n by m matrix
    n = 8
    m = 2
    
    i = 0
    j = 0
    for k in range(m):
        temp = []
        i = k
        j = 0
        for l in range(n):
            temp.append(float(p_final[j][i]))
            i+=1
            j+=1
            if i >= n:
                i = 0
            if j >= m:
                j = 0
        p_diag.append(torch.tensor(temp))
        #i+=1
    p_diag = torch.stack(p_diag)
    print(p_diag)
    
    
    p_output_file_name = "data/diagonal_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + ".txt"
    outfile_p = open(p_output_file_name,'w')
    for row in p_diag.tolist():
        for item in row:
            outfile_p.write(str(item))
            outfile_p.write(" ")
        outfile_p.write("\n")
    #P_final = str(p_diag.tolist())
    #outfile_p.write(P_final)
    outfile_p.close()
    
    
if __name__ == "__main__":
    lamb = 0.5
    margin = 0.5
    diagonal_encoding_transposed(lamb, margin)
