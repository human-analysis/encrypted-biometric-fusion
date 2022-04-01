# -*- coding: utf-8 -*-
import torch
import random
import math

class Linear_Feature_Fusion():
    def __init__(self,X,M,V,gamma,margin,lamb,indim=None,regularization=0,seed=0):
        random.seed(0)
        torch.manual_seed(0)
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
    def __init__(self,X,M,V,gamma,margin,lamb,indim=None,regularization=0,seed=0):
        random.seed(0)
        torch.manual_seed(seed)

        self.V = V
        self.M = M
        self.X = X
        self.scale = 1.0
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
                
        #interval = 0.1,0.7
        self.coeffs = [[3.6604110068015703, -7.308745554603273, 5.359140241417692, -0.03216663533709177], [-1.761181767348659, 5.619133141454438, -7.496635998204148, 5.491355198579896]]
        
        #interval = 0.1,0.7
        #self.coeffs = [[-14.87368246,23.74576715,-13.66592657,4.17688396]]
        
        
        #interval = 0.01,0.7
        #self.coeffs = [[11.836520387699572, -18.076619596914263, 9.213047940260486, -0.1390999565263271], [-5.035385227584069, 14.361565311498836, -14.664452287760135, 7.742745833744212]]
        
        self.coeffs = [[5.91965872,-7.3475699,3.54940138]]
        
        #self.coeffs = [[-2.61776258,2.78221164]]
        
        self.tote = 0
        self.escape = 0
        #self.powers = [3,1]
        
        self.P.requires_grad = True
    def parameters(self):
        return [self.P]
    def approximate_inv_norm(self,x_in):
        #self.coeffs should be a list of lists whose coefficients order similar to:
        #[[x^3, x^2, x, 1],[x, 1]]
        #
        x = torch.linalg.norm(x_in)**2
        self.tote += 1
        
        
        #if x < 0.01:
        if x < 0.1:
            self.escape += 1
            return 1/(x**0.5)
        if x > 0.7:
            self.escape += 1
            return 1/(x**0.5)
        
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
        P_temp = torch.mul(P_temp,self.scale)
        #P_temp = torch.mul(P_temp,1.0)
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
        """
        if torch.linalg.norm(x1_tilde)**2 < 0.1:
            x1_tilde = torch.mul(x1_tilde,10.0)
        if torch.linalg.norm(x2_tilde)**2 < 0.1:
            x2_tilde = torch.mul(x2_tilde,10.0)
        """
        r = random.randint(0,100000)
        if r <= 0:
            print("approx:", self.approximate_inv_norm(x1_tilde), "true:",1/(torch.linalg.norm(x1_tilde)))
        
        x1_tilde = torch.mul(x1_tilde,self.approximate_inv_norm(x1_tilde))
        
        x2_tilde = torch.mul(x2_tilde,self.approximate_inv_norm(x2_tilde))
        
        #print("new norm:",torch.linalg.norm(x1_tilde))
        #print()
        dotted = torch.dot(x1_tilde,x2_tilde)
        return 1-dotted
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
        
        
class Linear_Feature_Fusion_Approximate2():
    #main difference is that now we don't even include samples in loss that are outside our range
    def __init__(self,X,M,V,gamma,margin,lamb,indim=None,regularization=0,seed=0):
        random.seed(0)
        torch.manual_seed(seed)

        self.V = V
        self.M = M
        self.X = X
        self.scale = 1.0
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
                
        #interval = 0.1,0.7
        self.coeffs = [[3.6604110068015703, -7.308745554603273, 5.359140241417692, -0.03216663533709177], [-1.761181767348659, 5.619133141454438, -7.496635998204148, 5.491355198579896]]
        
        #interval = 0.1,0.7
        #self.coeffs = [[-14.87368246,23.74576715,-13.66592657,4.17688396]]
        
        
        #interval = 0.01,0.7
        #self.coeffs = [[11.836520387699572, -18.076619596914263, 9.213047940260486, -0.1390999565263271], [-5.035385227584069, 14.361565311498836, -14.664452287760135, 7.742745833744212]]
        
        self.coeffs = [[-14.87368246,23.74576715,-13.66592657,4.17688396]]
        
        #self.coeffs = [[5.91965872,-7.3475699,3.54940138]]
        
        self.coeffs = [[-2.61776258,2.78221164]]
        
        self.tote = 0
        self.escape = 0
        #self.powers = [3,1]
        
        self.P.requires_grad = True
    def parameters(self):
        return [self.P]
    def approximate_inv_norm(self,x_in, x):
        #self.coeffs should be a list of lists whose coefficients order similar to:
        #[[x^3, x^2, x, 1],[x, 1]]
        #
        #x = torch.linalg.norm(x_in)**2
        self.tote += 1
        
        #x = torch.linalg.norm(x_in)**2
        #if x < 0.01:
        if x < 0.1:
            self.escape += 1
            print('error')
            return 1/(x**0.5)
        if x > 0.7:
            self.escape += 1
            print('error')
            return 1/(x**0.5)
        
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
        
    def distance(self,x1_tilde,x2_tilde,xa,xb):
        
        #P_temp = torch.div(self.P,torch.linalg.norm(self.P))
        #P_temp = torch.mul(P_temp,self.scale)

        #x1_tilde = torch.matmul(P_temp.T, x1.T)
        #x2_tilde = torch.matmul(P_temp.T, x2.T)
        
        """
        if torch.linalg.norm(x1_tilde)**2 < 0.1:
            x1_tilde = torch.mul(x1_tilde,10.0)
        if torch.linalg.norm(x2_tilde)**2 < 0.1:
            x2_tilde = torch.mul(x2_tilde,10.0)
        """
        #r = random.randint(0,100000)
        #if r <= 0:
            #print("approx:", self.approximate_inv_norm(x1_tilde,xa), "true:",1/(torch.linalg.norm(x1_tilde)))
        
        x1_tilde = torch.mul(x1_tilde,self.approximate_inv_norm(x1_tilde,xa))
        
        x2_tilde = torch.mul(x2_tilde,self.approximate_inv_norm(x2_tilde,xb))
        
        #print("new norm:",torch.linalg.norm(x1_tilde))
        #print()
        dotted = torch.dot(x1_tilde,x2_tilde)
        return 1-dotted
        #return 1-torch.dot(x1_tilde, x2_tilde)/(self.approximate_norm(x1)*self.approximate_norm(x2))
    
    def loss(self):
    
        #x = torch.linalg.norm(x_in)**2
        P_temp = torch.div(self.P,torch.linalg.norm(self.P))
        P_temp = torch.mul(P_temp,self.scale)
        
        grand_total = len(self.M) + len(self.V)
        grand_skipped = 0
        skipped = 0
        
        pull = 0
        for i, j in self.M:
            x1_tilde = torch.matmul(P_temp.T, self.X[i].T)
            x2_tilde = torch.matmul(P_temp.T, self.X[j].T)
            xa = torch.linalg.norm(x1_tilde)**2
            xb = torch.linalg.norm(x2_tilde)**2
            if xa < 0.1 or xa > 0.7 or xb < 0.1 or xb > 0.7:
                skipped += 1
                continue
            pull += self.distance(x1_tilde,x2_tilde,xa,xb)
        pull = pull / (len(self.M)-skipped)
        
        grand_skipped = skipped
        skipped = 0
        
        push = 0
        for i, j, k in self.V:
            x1_tilde = torch.matmul(P_temp.T, self.X[i].T)
            x2_tilde = torch.matmul(P_temp.T, self.X[j].T)
            x3_tilde = torch.matmul(P_temp.T, self.X[k].T)
            xa = torch.linalg.norm(x1_tilde)**2
            xb = torch.linalg.norm(x2_tilde)**2
            xc = torch.linalg.norm(x3_tilde)**2
            if xa < 0.1 or xa > 0.7 or xb < 0.1 or xb > 0.7 or xc < 0.1 or xc > 0.7:
                skipped += 1
                continue
            push += max(0,self.margin + self.distance(x1_tilde,x2_tilde,xa,xb) - self.distance(x1_tilde,x3_tilde,xa,xc))
            #print(self.distance(self.X[i],self.X[j]))
            #print(self.distance(self.X[i],self.X[k]))
            #print()
        push = push / (len(self.V)-skipped)
        grand_skipped += skipped
        
        print("% skipped:",grand_skipped/grand_total)
        
        loss = self.lamb * pull + (1-self.lamb) * push
        #print("pull loss:",pull,"push loss:",push)
        #loss = loss + self.regularization * torch.linalg.norm(self.P)
        #print(torch.linalg.norm(self.P))
        
        return loss
        
class Linear_Feature_Fusion_Goldschmidt():
    def __init__(self,X,M,V,gamma,margin,lamb,indim=None,regularization=0):
        random.seed(0)
        torch.manual_seed(0)

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
        self.coeffs = [[-2.61776258,2.78221164]]
        
        self.P.requires_grad = True
    def parameters(self):
        return [self.P]
    def approximate_inv_norm(self,x_in):
        #self.coeffs should be a list of lists whose coefficients order similar to:
        #[[x^3, x^2, x, 1],[x, 1]]
        #
        x = torch.linalg.norm(x_in)**2
        
        inital_guess = x * self.coeffs[0][0] + self.coeffs[0][1]
        
        #fisr
        x = x * inital_guess**2
        x = initial_guess * -0.5 * x
        temp = 1.5 * inital_guess
        inital_guess = temp + x
        
        #gold
        
        
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
        
        """
        if torch.linalg.norm(x1_tilde)**2 < 0.1:
            x1_tilde = torch.mul(x1_tilde,10.0)
        if torch.linalg.norm(x2_tilde)**2 < 0.1:
            x2_tilde = torch.mul(x2_tilde,10.0)
        """
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


class Linear_Feature_Fusion_No_Normal():
    def __init__(self,X,M,V,gamma,margin,lamb,indim=None,regularization=0,seed=0):
        random.seed(0)
        torch.manual_seed(seed)

        self.V = V
        self.M = M
        self.X = X
        self.scale = 1.0
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
                
        #interval = 0.1,0.7
        self.coeffs = [[3.6604110068015703, -7.308745554603273, 5.359140241417692, -0.03216663533709177], [-1.761181767348659, 5.619133141454438, -7.496635998204148, 5.491355198579896]]
        
        #interval = 0.1,0.7
        self.coeffs = [[-14.87368246,23.74576715,-13.66592657,4.17688396]]
        
        self.coeffs = [[5.91965872,-7.3475699,3.54940138]]
        
        #self.coeffs = [[-2.61776258,2.78221164]]
        
        self.tote = 0
        self.escape = 0
        #self.powers = [3,1]
        
        self.P.requires_grad = True
    def parameters(self):
        return [self.P]
    def approximate_inv_norm(self,x_in):
        #self.coeffs should be a list of lists whose coefficients order similar to:
        #[[x^3, x^2, x, 1],[x, 1]]
        #
        x = torch.linalg.norm(x_in)**2
        self.tote += 1
        """
        if x < 0.1:
            self.escape += 1
            return 1/(x**0.5)
        if x > 0.7:
            self.escape += 1
            return 1/(x**0.5)
        """
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
        #P_temp = torch.mul(P_temp,self.scale)
        #P_temp = torch.mul(P_temp,1.0)
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
        
        #if torch.linalg.norm(x1_tilde)**2 < 0.1:
            #x1_tilde = torch.mul(x1_tilde,10.0)
        #if torch.linalg.norm(x2_tilde)**2 < 0.1:
            #x2_tilde = torch.mul(x2_tilde,10.0)
        
        #r = random.randint(0,100000)
        #if r <= 0:
            #print("approx:", self.approximate_inv_norm(x1_tilde), "true:",1/(torch.linalg.norm(x1_tilde)))
        
        #x1_tilde = torch.mul(x1_tilde,self.approximate_inv_norm(x1_tilde))
        
        #x2_tilde = torch.mul(x2_tilde,self.approximate_inv_norm(x2_tilde))
        
        #print("new norm:",torch.linalg.norm(x1_tilde))
        #print()
        dotted = torch.dot(x1_tilde,x2_tilde)
        return 1-dotted
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
        


if __name__ == "__main__":
    print(approximate_norm(torch.tensor([2**0.5])))
    print("truth:",111266)
