"""
Author: Luke Sperling
Created: 01-27-22
Modified: 07-07-22
Learned Matrix (after proper scaling) from HEFT_train needs to be diagonally encoded in order for hybrid scheme for matrix-matrix multiplcation to be used.
The output file from this script is to be used with the SEAL script for the hybrid encoding scheme.
"""


import torch
import ast

def diagonal_encoding_arbitrary(in_name, out_name):
    p_file = open(in_name,'r')
    p_values = []
    for line in p_file:
        P = torch.tensor(ast.literal_eval(line.strip()))
        p_values.append(P)
    p_matrix = torch.stack(p_values)
    p_matrix = torch.squeeze(p_matrix)

    max_dim_h = p_matrix.shape[0]
    max_dim_w = p_matrix.shape[1]
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
    
    print(new_max_dim_h,new_max_dim_w)
    
    p_final = []
    for i in range(new_max_dim_h):
        temp = []
        for j in range(new_max_dim_w):
            if i >= p_matrix.shape[0] or j >= p_matrix.shape[1]:
                temp.append(0)
            else:
                temp.append(float(p_matrix[i][j]))
        p_final.append(torch.tensor(temp))
    p_final = torch.stack(p_final)
    
    p_diag = []
    #n by m matrix
    n = new_max_dim_h
    m = new_max_dim_w
    
    i = 0
    j = 0
    for k in range(n):
        temp = []
        i = 0
        j = k
        for l in range(m):
            temp.append(float(p_final[i][j]))
            i+=1
            j+=1
            if i >= n:
                i = 0
            if j >= m:
                j = 0
        p_diag.append(torch.tensor(temp))
    p_diag = torch.stack(p_diag)
    print("final=",p_diag.shape)
    
    outfile_p = open(out_name,'w')
    for row in p_diag.tolist():
        for item in row:
            outfile_p.write(str(f'{item:.9f}'))
            outfile_p.write(" ")
        outfile_p.write("\n")
    outfile_p.close()

if __name__ == "__main__":
    diagonal_encoding_arbitrary("data/exact_results/exact_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=32_reg=0.txt","data/exact_results/diagonal/diagonal_exact_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=32_reg=0.txt")
    diagonal_encoding_arbitrary("data/degree=3strict/large_approximate_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=32_reg=0.txt","data/degree=3strict/diagonal/diagonal_large_approximate_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=32_reg=0.txt")
    
