# -*- coding: utf-8 -*-
import torch
import random
import math

class Linear_Feature_Fusion():
    def __init__(self,X,M,V,gamma,margin,lamb,indim=None,regularization=0):
        random.seed(0)
        self.V = V
        self.M = M
        self.X = X
        self.regularization = regularization
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
        loss = loss + self.regularization * torch.linalg.norm(self.P)
        #print(torch.linalg.norm(self.P))
        
        return loss
        
    """
    def loss(self):
        #random batch chosen
        pull = 0
        interval = 25
        ranges = math.floor(len(self.M)/interval)
        random_index = random.randint(0,ranges-1)
        for c in range(interval):
            i, j = self.M[random_index*interval+c]
            #for i, j in self.M[random_index*interval:(random_index+1)*interval]:
            pull += self.distance(self.X[i],self.X[j])
        pull = pull / len(self.M)
        
        push = 0
        ranges = math.floor(len(self.V)/interval)
        random_index = random.randint(0,ranges-1)
        for c in range(interval):
            i, j, k = self.V[random_index*interval+c]
            #for i, j in self.V[random_index*interval:(random_index+1)*interval]:
            push += max(0,self.margin + self.distance(self.X[i],self.X[j]) - self.distance(self.X[i],self.X[k]))
            #print(self.distance(self.X[i],self.X[j]))
            #print(self.distance(self.X[i],self.X[k]))
            #print()
        push = push / len(self.V)
        
        loss = self.lamb * pull + (1-self.lamb) * push
        #print("pull loss:",pull,"push loss:",push)
        loss = loss + self.regularization * torch.linalg.norm(self.P)
        #print(torch.linalg.norm(self.P))
        
        return loss
    """
    
class Linear_Feature_Fusion_Approximate():
    def __init__(self,X,M,V,gamma,margin,lamb,indim=None,regularization=0):
        random.seed(0)
        self.V = V
        self.M = M
        self.X = X
        self.regularization = regularization
        if not indim:
            alpha_beta = X.shape[1]
        else:
            alpha_beta = indim
        self.margin = margin
        self.P = torch.rand(alpha_beta,gamma)
        #self.P = torch.ones((alpha_beta,gamma))
        self.lamb = lamb
        self.coeffs = [[1.28740283e+06, -3.30949673e+03, 3.49096065e+00, -1.25847377e-03], [1.70373014e-02, 7.58955955e-08]]
        self.coeffs = [[-1.25847377e-03, 3.49096065e+00, -3.30949673e+03, 1.28740283e+06],[7.58955955e-08, 1.70373014e-02]]
        
        self.coeffs =[[-7.81259766e-09, 4.06252214e-03]]
        
        #[ 5.21696229e+07  3.51836911e+06 -1.66419686e+03  2.86054276e-01
        #[6.89278658e-02 -1.70092408e-11]]
        
        self.coeffs = [[2.86054276e-01,-1.66419686e+03,3.51836911e+06,5.21696229e+07],[-1.70092408e-11,6.89278658e-02]]
        
        
        #[ 9.58797780e-01 -4.43025846e+01  5.54409918e+02 -2.52567888e+03
        #5.46180222e+00  5.13593380e+00  3.20072049e-01  8.03850688e+00]
        
        self.coeffs= [[-2.52567888e+03,5.54409918e+02,-4.43025846e+01,9.58797780e-01],[8.03850688e+00,3.20072049e-01,5.13593380e+00,5.46180222e+00]]
        #[ 0.83145916 -3.07507111  5.3092142  -4.42648824
        #1.33883108  1.00840009 2.29174683  2.43985827]
        self.coeffs= [[-4.42648824,5.3092142,-3.07507111,0.83145916],[2.43985827,2.29174683,1.00840009,1.33883108]]
        
        #interval = 0.1,1.0
        #relative error = 0.0527
        self.coeffs= [[4.400171203929593, -9.438491353540634, 6.701697618379752, -0.2080454748840371], [-4.869694790507371, 12.833246397743965, -11.941356320502285, 5.9861811924000055]]
        #self.powers = [3,1]
        
        self.P.requires_grad = True
    def parameters(self):
        return [self.P]
    def approximate_inv_norm(self,x_in):
        #self.coeffs should be a list of lists whose coefficients order similar to:
        #[[x^3, x^2, x, 1],[x, 1]]
        #
        x = torch.linalg.norm(x_in)**2
        #print("x:",x)
        #print(self.coeffs)
        result = 0
        for coeff_list in self.coeffs:
            result = coeff_list[0]
            for i in range(1,len(coeff_list)):
                #print(coeff_list[i])
            
                result = result * x + coeff_list[i]
                #result += coeff_list[i]
            x = result
            #print(x)
            result = 0
        #print("result:",x)
        return x
        
    def distance(self,x1,x2):
        
        #print("P norm:",torch.linalg.norm(P_temp))
        
        #to_mult = torch.linalg.norm(x1)/250.0
        #P_temp = torch.mul(torch.div(self.P,torch.linalg.norm(self.P)),to_mult)
        P_temp = torch.div(self.P,torch.linalg.norm(self.P))
        #P_temp = self.P
        #print("P:",torch.linalg.norm(P_temp), "X1:",torch.linalg.norm(x1), "X2:",torch.linalg.norm(x2))
        #print(P_temp.shape)
        #print(P_temp)
        x1_tilde = torch.matmul(P_temp.T, x1.T)
        x2_tilde = torch.matmul(P_temp.T, x2.T)
        #print("X1_tilde:",torch.linalg.norm(x1_tilde))
        #print("old squared norm:",torch.linalg.norm(x1_tilde)**2)
        #print()
        #print("true inverse norm:", torch.linalg.norm(x1_tilde), "approximate:",self.approximate_norm(x1_tilde))
        r = random.randint(0,100000)
        if r == 2:
            print("approx:", self.approximate_inv_norm(x1_tilde), "true:",1/(torch.linalg.norm(x1_tilde)))
        
        x1_tilde = torch.mul(x1_tilde,self.approximate_inv_norm(x1_tilde))
        
        x2_tilde = torch.mul(x2_tilde,self.approximate_inv_norm(x2_tilde))
        
        #print("new norm:",torch.linalg.norm(x1_tilde))
        #print()
        return 1-torch.dot(x1_tilde,x2_tilde)
        #return 1-torch.dot(x1_tilde, x2_tilde)/(self.approximate_norm(x1)*self.approximate_norm(x2))
    
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
        loss = loss + self.regularization * torch.linalg.norm(self.P)
        #print(torch.linalg.norm(self.P))
        
        return loss
def approximate_norm(x_in):
    x = torch.linalg.norm(x_in)**2
    print(x)
    coeffs = [[3,2,1,4],[2,1,2,2]]
    result = 0
    for coeff_list in coeffs:
        result = coeff_list[0]
        for i in range(1,len(coeff_list)):
            result = result * x
            result += coeff_list[i]
        x = result
        result = 0
    return x

if __name__ == "__main__":
    print(approximate_norm(torch.tensor([2**0.5])))
    print("truth:",111266)
