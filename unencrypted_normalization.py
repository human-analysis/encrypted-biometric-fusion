import tenseal as ts
import math
import torch

def reencrypt(enc_vector, context):
    plain = enc_vector.decrypt()
    return ts.ckks_vector(context,plain)

def Goldschmidt(s, initial_estimate, context, iters):
    x = s * initial_estimate
    h = initial_estimate * 0.5
    for i in range(iters):
        r = -x*h + 0.5
        x = x + x*r
        h = h + h*r
        
    return h#*2

def NewtonsMethod(linear_approximation, s, context, iters):
    x = linear_approximation
    for i in range(iters):
        left = x * 0.5
        #print("sub negate")
        subformula = -s
        #print("mult")
        subformula = subformula * x
        #print("mult")
        subformula = subformula * x
        #subformula = subformula * x**2
        #subformula = subformula.mul(x)
        subformula = subformula + 3
        #x = x * ( * x * x + [3])
        #print("mult")
        x = left * subformula
    return x

def InvNormApprox(linear_approximation, s, context, iters):
    neg_half = -0.5
    three_half = 1.5
    guess = linear_approximation
    sq = s #?
    for i in range(iters):
        sq = sq * guess**2
        sq = guess * neg_half * sq
        temp = three_half * guess
        guess = temp + sq
    return guess

def normalize(vector, context, dimensionality):
    s = torch.linalg.norm(vector)#enc_vector * enc_vector#copy.deepcopy(enc_vector)
    s = torch.dot(vector,vector)
    #for i in range(math.log(dimensionality,2)):
        #s
    #ts.ckks_vector(context, X_test[i])
    
    #these linear approximation constants from Principal Component Analysis using CKKS Homomorphic Scheme, Samanvaya Panda,
    #International Symposium on Cyber Security Cryptography and Machine Learning, 2021
    #linear_weight = -0.00019703
    #linear_bias = 0.14777278
    
    #these values obtained by my own training of a linear regressor #[-0.00029703]), 0.1586186138991184
    linear_weight = -0.00029703#-4.40369448e-05
    linear_bias = 0.1586186138991184#0.08852906761144604
    
    #print("mult")
    linear_approximation = linear_weight * s
    linear_approximation = linear_approximation + linear_bias
    
    
    
    linear_approximation = 0.01
    
    print("linear approx:",linear_approximation)
    
    #initial_estimate = NewtonsMethod(linear_approximation, s, context, iters=10)
    initial_estimate = InvNormApprox(linear_approximation, s, context, iters=1)
    
    
    
    inverse_norm = Goldschmidt(s, initial_estimate, context, iters=4)
    
    inverse_norm *= 2
    #inverse_norm = 1/(s**0.5)
    #print("squared norm:",s)
    #print("inverse norm:",inverse_norm)
    #print(torch.dot(vector * inverse_norm,vector * inverse_norm))
    #print(torch.linalg.norm(vector * inverse_norm))
    return vector * inverse_norm