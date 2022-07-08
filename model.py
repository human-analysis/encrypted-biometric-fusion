"""
Author: Luke Sperling
Created: 03-28-22
Modified: 07-07-22
Models for FHE-aware learning and not FHE-aware learning.
"""
import torch
import random
import math

class Linear_Feature_Fusion():
    def __init__(self,X,M,V,L,gamma,margin,lamb,indim=None,regularization=0,seed=0):
        random.seed(0)
        torch.manual_seed(0)
        self.V = V
        self.M = M
        self.X = X
        self.L = L
        self.regularization = regularization
        if not indim:
            alpha_beta = X.shape[1]
        else:
            alpha_beta = indim
        self.margin = margin
        self.P = torch.rand(alpha_beta,gamma)
        self.lamb = lamb
        self.P.requires_grad = True
        self.index = 0
        
    def parameters(self):
        return [self.P]
    def distance(self,x1,x2):
        x1_tilde = torch.matmul(self.P.T, x1.T)
        x2_tilde = torch.matmul(self.P.T, x2.T)
        return 1-torch.dot(x1_tilde, x2_tilde)/(torch.linalg.norm(x1_tilde)*torch.linalg.norm(x2_tilde))
    
    def loss(self):
        batch_size = 64
        random_index = self.index
        self.index += batch_size# // 2
        show_flag = False
        if self.index >= self.X.shape[0] - batch_size - 1:
            self.index = 0
            #randomize train set order to aid in batches
            temp = list(zip(self.X.tolist(), self.L.tolist()))
            random.shuffle(temp)
            res1, res2 = zip(*temp)
            # res1 and res2 come out as tuples, and so must be converted to lists.
            X_train_list, L_train_list = list(res1), list(res2)
            self.X = torch.tensor(X_train_list)
            self.L = torch.tensor(L_train_list)
            show_flag = True
            
        X_temp = self.X[random_index:random_index+batch_size,:]
        L_temp = self.L[random_index:random_index+batch_size]
        M = []
        V = []
        for i in range(X_temp.shape[0]):
            for j in range(i+1,X_temp.shape[0]):
                if L_temp[i] == L_temp[j]:
                    M.append((i,j))
                    for k in range(X_temp.shape[0]):
                        if L_temp[k] != L_temp[i]:
                            V.append((i,j,k))
        if show_flag:
            print("Same class",len(M))
            print("Trios:",len(V))
        pull = 0
        for i, j in M:
            pull += self.distance(X_temp[i],X_temp[j])
        pull = pull / len(M)
        
        push = 0
        for i, j, k in V:
            push += max(0,self.margin + self.distance(X_temp[i],X_temp[j]) - self.distance(X_temp[i],X_temp[k]))
        push = push / len(V)
        
        loss = self.lamb * pull + (1-self.lamb) * push
        return loss

        
class Linear_Feature_Fusion_FHEaware():
    def __init__(self,X,M,V,L,gamma,margin,lamb,indim=None,regularization=0,seed=0):
        random.seed(0)
        torch.manual_seed(0)
        self.V = V
        self.M = M
        self.X = X
        self.L = L
        self.scale = 1.0
        self.regularization = regularization
        if not indim:
            alpha_beta = X.shape[1]
        else:
            alpha_beta = indim
        self.margin = margin
        self.P = torch.rand(alpha_beta,gamma)
        self.lamb = lamb
        self.P.requires_grad = True
        #index tells us what portion of the train data is our current batch
        self.index = 0
        self.coeffs = [[0.42084296,-1.81897596,2.51308415]]
        self.tote = 0
        self.escape = 0
        self.P.requires_grad = True
    def parameters(self):
        return [self.P]
    def approximate_inv_norm(self,x_in, x):
        #self.coeffs should be a list of lists whose coefficients order similar to:
        #[[x^3, x^2, x, 1],[x, 1]]
        self.tote += 1
        #only use values within the valid range
        if x < 0.05:
            self.escape += 1
            print('error')
            return 1/(x**0.5)
        if x > 3.0:
            self.escape += 1
            print('error')
            return 1/(x**0.5)
        result = 0
        for coeff_list in self.coeffs:
            result = coeff_list[0]
            for i in range(1,len(coeff_list)):
                result = result * x + coeff_list[i]
            x = result
            result = 0
        return x
        
    def distance(self,x1_tilde,x2_tilde,xa,xb):
        x1_tilde = torch.mul(x1_tilde,self.approximate_inv_norm(x1_tilde,xa))
        x2_tilde = torch.mul(x2_tilde,self.approximate_inv_norm(x2_tilde,xb))
        dotted = torch.dot(x1_tilde,x2_tilde)
        return 1-dotted

    def loss(self):
        batch_size = 64
        random_index = self.index
        self.index += batch_size
        show_flag = False
        if self.index >= self.X.shape[0] - batch_size - 1:
            self.index = 0
            #randomize train set order to aid in batches
            temp = list(zip(self.X.tolist(), self.L.tolist()))
            random.shuffle(temp)
            res1, res2 = zip(*temp)
            # res1 and res2 come out as tuples, and so must be converted to lists.
            X_train_list, L_train_list = list(res1), list(res2)
            self.X = torch.tensor(X_train_list)
            self.L = torch.tensor(L_train_list)
            show_flag = True
            
        X_temp = self.X[random_index:random_index+batch_size,:]
        L_temp = self.L[random_index:random_index+batch_size]
        M = []
        V = []
        for i in range(X_temp.shape[0]):
            for j in range(i+1,X_temp.shape[0]):
                if L_temp[i] == L_temp[j]:
                    M.append((i,j))
                    for k in range(X_temp.shape[0]):
                        if L_temp[k] != L_temp[i]:
                            V.append((i,j,k))
        if show_flag:
            print("Same class",len(M))
            print("Trios:",len(V))

        P_temp = torch.div(self.P,torch.linalg.norm(self.P))
        P_temp = torch.mul(P_temp,self.scale)
        
        grand_total = len(M) + len(V)
        grand_skipped = 0
        skipped = 0
        
        pull = 0
        for i, j in M:
            x1_tilde = torch.matmul(P_temp.T, X_temp[i].T)
            x2_tilde = torch.matmul(P_temp.T, X_temp[j].T)
            xa = torch.linalg.norm(x1_tilde)**2
            xb = torch.linalg.norm(x2_tilde)**2
            #only use values within the valid range
            if xa < 0.05 or xa > 3.0 or xb < 0.05 or xb > 3.0:
                skipped += 1
                continue
            pull += self.distance(x1_tilde,x2_tilde,xa,xb)
        pull = pull / (len(M)-skipped)
        
        grand_skipped = skipped
        skipped = 0
        
        push = 0
        for i, j, k in V:
            x1_tilde = torch.matmul(P_temp.T, X_temp[i].T)
            x2_tilde = torch.matmul(P_temp.T, X_temp[j].T)
            x3_tilde = torch.matmul(P_temp.T, X_temp[k].T)
            xa = torch.linalg.norm(x1_tilde)**2
            xb = torch.linalg.norm(x2_tilde)**2
            xc = torch.linalg.norm(x3_tilde)**2
            #only use values within the valid range
            if xa < 0.05 or xa > 3.0 or xb < 0.05 or xb > 3.0 or xc < 0.05 or xc > 3.0:
                skipped += 1
                continue
            push += max(0,self.margin + self.distance(x1_tilde,x2_tilde,xa,xb) - self.distance(x1_tilde,x3_tilde,xa,xc))
        push = push / (len(V)-skipped)
        grand_skipped += skipped

        loss = self.lamb * pull + (1-self.lamb) * push
        return loss



