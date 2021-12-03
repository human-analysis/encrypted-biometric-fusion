# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:41:16 2021

@author: Luke
"""

import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
from data_generation import data_gen
import ast


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.soft(x)
        
        return x

def main():
    net = Net()
    
    A,B,L = data_gen(10)
    
    lamb = 0.1
    margin = 0.5
    
    p_file_name = "data/best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + ".txt"
    p_file = open(p_file_name,'r')
    P = None
    for line in p_file:
        P = torch.tensor(ast.literal_eval(line.strip()))
    
    A_test,B_test,L_test = data_gen(10)
    X = torch.cat((A,B),dim=1)
    X_test = torch.cat((A_test,B_test),dim=1)
    
    X_prime = torch.mm(X,torch.transpose(P,0,1)) 
    X_prime_test = torch.mm(X_test,torch.transpose(P,0,1)) 
    
    output = net(X_prime)
    inter_target = []
    for label in L:
        temp = [0]*4#len(set(L))
        temp[int(label)] = 1
        temp = torch.tensor(temp)
        temp = torch.unsqueeze(temp,dim=0)
        inter_target.append(torch.tensor(temp))
    target = torch.Tensor(40,4)
    torch.cat(inter_target, out=target)
    
    inter_target_test = []
    for label in L:
        temp = [0]*4#len(set(L))
        temp[int(label)] = 1
        temp = torch.tensor(temp)
        temp = torch.unsqueeze(temp,dim=0)
        inter_target_test.append(torch.tensor(temp))
    target_test = torch.Tensor(40,4)
    torch.cat(inter_target_test, out=target_test)
    
        
    #target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()
    
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    
    for i in range(1000):
        if i%100 == 0:
            print("Iteration",i)
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(X_prime)
        #print(output.size())
        #print(target.size())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update
    
    #time to test
    output = net(X_prime_test)
    
    print("OUTPUT:")
    print(output)
    print()
    
    thresholds = [0.1 * i for i in range(10)]
    fps = []
    tps = []
    
    for thresh in thresholds:
        fp = 0
        #fn = 0
        tp = 0
        #tn = 0
        for i in range(40):
            if max(output[i]) < thresh:
                continue
            pred_class = list(output[i]).index(max(output[i]))
            true_class = list(target_test[i]).index(1)
            if pred_class == true_class:
                tp += 1
            else:
                fp += 1
        fps.append(fp)
        tps.append(tp)
    print(tps)
    print()
    print(fps)
    print()
    
    loss = criterion(output, target_test)
    print("Test lose:",loss)
    
if __name__ == "__main__":
    main()