import torch
import time
import ast

def Cosine_Similarity_no_div(vec1, vec2):
    return torch.dot(vec1, vec2)

def main():
    a_file = open("data4/dataset/A_values_test.txt",'r')
    b_file = open("data4/dataset/B_values_test.txt",'r')
    L_file = open("data4/dataset/L_values_test.txt",'r')
    
    A = []
    
    
    L = L_file.readline()
    L = [int(i) for i in L[1:len(L)-2].split(", ")]
    #L = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
    for line in a_file:
        line = line.replace("[","")
        line = line.replace("]","")
        line = line.replace(",","")
        line = line.strip()
        line = [float(i) for i in line.split(" ")]
        result = torch.tensor(line).unsqueeze(dim=0)
        #print(result.shape)
        A.append(result)
    A_final = torch.Tensor(len(A),A[0].shape[0])
    torch.cat(A, out=A_final,dim=0)
    
    
    B = []
    for line in b_file:
        line = line.replace("[","")
        line = line.replace("]","")
        line = line.replace(",","")
        line = line.strip()
        line = [float(i) for i in line.split(" ")]
        result = torch.tensor(line).unsqueeze(dim=0)
        B.append(result)
    B_final = torch.Tensor(len(B),B[0].shape[0])
    torch.cat(B, out=B_final,dim=0)

    filename = "data4/degree=2strict/large_approximate_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=32_reg=0.txt"
    p_file = open(filename,'r')
    p = []
    p_final = None
    for line in p_file:
        result = torch.tensor(ast.literal_eval(line.strip()))
        p.append(result)
        p_final = result

    start = time.perf_counter()
    #Create feature fusion dataset
    X = torch.cat((A_final,B_final),dim=1)
    
    end = time.perf_counter()
    
    concat_time = end-start #seconds
    
    start = time.perf_counter()
    X_prime = torch.mm(X, p_final.T)
    end = time.perf_counter()
    
    project_time = end-start #seconds
    
    start = time.perf_counter()
    for i in range(X_prime.shape[0]):
        X_prime[i,:]=torch.div(X_prime[i,:], torch.linalg.norm(X_prime[i,:]))
    end = time.perf_counter()
    
    normal_time = end-start #seconds
    
    start = time.perf_counter()
    y_score = []
    y_true = []
    count = len(L)
    for i in range(count):
        for j in range(i,count):
            if i == j:
                continue
            score = Cosine_Similarity_no_div(X_prime[i],X_prime[j])
            if L[i]==L[j]:
                label = 1
            else:
                label = 0
            y_score.append(score.detach().numpy())
            y_true.append(label)
    end = time.perf_counter()
    
    match_time = end-start #seconds
    
    print("concat time:",concat_time)
    print("project time:",project_time)
    print("normalize time:",normal_time)
    print("match time:",match_time)
main()
