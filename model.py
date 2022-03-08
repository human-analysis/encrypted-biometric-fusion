# -*- coding: utf-8 -*-
import torch

class Linear_Feature_Fusion():
    def __init__(self,X,M,V,gamma,margin,lamb,indim=None):
        self.V = V
        self.M = M
        self.X = X
        if not indim:
            alpha_beta = X.shape[1]
        else:
            alpha_beta = indim
        self.margin = margin
        self.P = torch.rand(alpha_beta,gamma)
        #self.P = torch.ones((alpha_beta,gamma))
        self.lamb = lamb
        
        self.P.requires_grad = True
    def parameters(self):
        return [self.P]
    def distance(self,x1,x2):

        x1_tilde = torch.matmul(self.P.T, x1.T)
        x2_tilde = torch.matmul(self.P.T, x2.T)
        
        
        #print("before",x1/torch.linalg.norm(x1), x2/torch.linalg.norm(x2))
        #print("after",x1_tilde/torch.linalg.norm(x1_tilde), x2_tilde/torch.linalg.norm(x2_tilde))
        #print()
        
        #self.P = torch.rand(self.P.shape)
        #x1_tilde = torch.matmul(self.P.T, x1.T)
        #x2_tilde = torch.matmul(self.P.T, x2.T)
        #print(torch.dot(x1_tilde, x2_tilde), (torch.linalg.norm(x1_tilde)*torch.linalg.norm(x2_tilde)))
        #self.P = torch.ones(self.P.shape)
        #x1_tilde = torch.matmul(self.P.T, x1.T)
        #x2_tilde = torch.matmul(self.P.T, x2.T)
        #print(torch.dot(x1_tilde, x2_tilde), (torch.linalg.norm(x1_tilde)*torch.linalg.norm(x2_tilde)))
        #print(torch.dot(x1, x2), (torch.linalg.norm(x1)*torch.linalg.norm(x2)))
        #print()
        return 1-torch.dot(x1_tilde, x2_tilde)/(torch.linalg.norm(x1_tilde)*torch.linalg.norm(x2_tilde))
    def loss(self):
        pull = 0
        for i, j in self.M:
            pull += self.distance(self.X[i],self.X[j])
        pull = pull / len(self.M)
        
        push = 0
        for i, j, k in self.V:
            push += max(0,self.margin + self.distance(self.X[i],self.X[j]) - self.distance(self.X[i],self.X[k]))
            #print(self.distance(self.X[i],self.X[j]))
            #print(self.distance(self.X[i],self.X[k]))
            #print()
        push = push / len(self.V)
        
        loss = self.lamb * pull + (1-self.lamb) * push
        #print("pull loss:",pull,"push loss:",push)
        #loss = loss + 0.001 * torch.linalg.norm(self.P)
        
        return loss
