#import tenseal as ts
#import time
#import enchant

#from nltk.corpus import words

import torch
import ast


lamb = 0.5
margin = 0.5

p_file_name = "data/best_P_value_transpose_lambda=" + str(lamb) + "_margin=" + str(margin) + ".txt"
p_file = open(p_file_name,'r')
p_values = []
for line in p_file:
    P = torch.tensor(ast.literal_eval(line.strip()))
    p_values.append(P)
    
a_file = open("data/A_values.txt",'r')
A = []
for line in a_file:
    a = torch.tensor(ast.literal_eval(line.strip())).unsqueeze(dim=0)
    A.append(a)
A_final = torch.Tensor(len(A),A[0].shape[0])
torch.cat(A, out=A_final,dim=0)

b_file = open("data/B_values.txt",'r')
B = []
for line in b_file:
    b = torch.tensor(ast.literal_eval(line.strip())).unsqueeze(dim=0)
    B.append(b)
B_final = torch.Tensor(len(B),B[0].shape[0])
torch.cat(B, out=B_final,dim=0)
    
l_file = open("data/L_values.txt",'r')
L = []
for line in l_file:
    L.append(int(float(line.strip())))


#Create feature fusion dataset
X = torch.cat((A_final,B_final),dim=1)

X_prime = torch.mm(X,torch.transpose(p_values[0],0,1))


encrypted_method_values = []
enc_file = open("results/toy_data_polynomial.txt")
for line in enc_file:
    line = line.strip().split(";")[0]
    line = line.split()
    encrypted_method_values.append((float(line[0]),float(line[1])))
    

largest_error = 0
total_error = 0
count = 40

print(X_prime)
for i in range(X_prime.shape[0]):
    result = (X_prime[i][0]**2 + X_prime[i][1]**2)**0.5
    true_x = float(X_prime[i][0]/result)
    true_y = float(X_prime[i][1]/result)
    test_x = encrypted_method_values[i][0]
    test_y = encrypted_method_values[i][1]
    dist = ((true_x-test_x)**2+(true_y-test_y)**2)**0.5
    total_error += dist
    if dist > largest_error:
        largest_error = dist
print("Worst error:",largest_error)
print("Average error:",total_error/count)
    
    #print(float(X_prime[i][0]/result), str(float(X_prime[i][1]/result))+";"+str(L[i]))

"""
results = []
word = ["0"]*5
word[1] = 'o'
guaranteed = ['r','t']
possible = ['q','w','z','x','v','b','k']
possible.append('r')
possible.append('t')
possible.append('o')

free = [0,2,3,4]
for pos_t in free:
    #print("iter")
    freetemp = free[:]
    wordtemp = word[:]
    freetemp.remove(pos_t)
    for pos_r in freetemp:
        if pos_r == 0:
            continue
        #freetemp = freetemp - [pos_t]
        #freetemp = freetemp - [pos_r]
        wordtemp[pos_t] = 'r'
        wordtemp[pos_r] = 't'
        freetemp3 = freetemp[:]
        freetemp3.remove(pos_r)
        for letter1 in possible:
            freetemp2 = possible[:]
            freetemp2.remove(letter1)
            for letter2 in freetemp2:
                #print(freetemp)
                wordtemp[freetemp3[0]] = letter1
                wordtemp[freetemp3[1]] = letter2
                results.append("".join(wordtemp))
                #print("".join(wordtemp))
                wordtemp[freetemp3[0]] = letter2
                wordtemp[freetemp3[1]] = letter1
                #print("".join(wordtemp))
                results.append("".join(wordtemp))
print(results)
d = enchant.Dict("en_US")
print("checking dict")
for result in results:
    if d.check(result):
        print(result)
    #if result in words.words():
        #print(result)
print("done checking")
0/0
# Setup TenSEAL context
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40

v1 = []
for i in range(128):
    v1.append(i)
v2 = [4, 4, 4]

# encrypted vectors

tic = time.perf_counter()
#enc_v1 = ts.ckks_vector(context, v1)
enc_v2 = ts.ckks_vector(context, v2)

#result = enc_v1 + enc_v2
#print(result.decrypt()) # ~ [4, 4, 4, 4, 4]

result = enc_v2 * enc_v2#enc_v1.dot(enc_v2)
toc = time.perf_counter()
print(toc-tic)
print(result.decrypt()) # ~ [10]


tic = time.perf_counter()
enc_v1 = ts.ckks_vector(context, v1)

result = enc_v1 * enc_v1
toc = time.perf_counter()
print(toc-tic)
print(result.decrypt()) # ~ [10]
"""
