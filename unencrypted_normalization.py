import tenseal as ts
#import math
import torch

import plotly.express as px
import plotly.graph_objects as go
import pandas

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
    
    linear_weight = -0.00032523
    linear_bias = 0.16795745
    
    #print("mult")
    linear_approximation = linear_weight * s
    linear_approximation = linear_approximation + linear_bias
    
    
    
    #linear_approximation = 0.01
    
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
    #return inverse_norm
    return vector * inverse_norm


def gold_test(s, context, iters):
    linear_weight = -0.00032523
    linear_bias = 0.16795745
    
    #print("mult")
    linear_approximation = linear_weight * s
    linear_approximation = linear_approximation + linear_bias
    
    
    
    #linear_approximation = 0.01
    
    #print("linear approx:",linear_approximation)
    
    #initial_estimate = NewtonsMethod(linear_approximation, s, context, iters=10)
    initial_estimate = InvNormApprox(linear_approximation, s, context, iters=1)
    
    
    
    inverse_norm = Goldschmidt(s, initial_estimate, context, iters=iters)
    
    inverse_norm *= 2

    return inverse_norm

def efficient_normalize(vector, context, dimensionality):
    #0.8178302654835186
    #coeffs = [ 0.00000000e+00, -1.00922126e-01,  7.19802379e-03, -2.59873687e-04,
      #4.52208769e-06, -3.01334013e-08]
    #coeffs = [0.8178302654835186, -1.00922126e-01,  7.19802379e-03, -2.59873687e-04,
      #4.52208769e-06, -3.01334013e-08]
    coeffs = [[-1.74906337e+06,  5.73576978e+04, -2.63561506e+01,  3.31516365e-03],
      [2.14909489e-01, -2.42205486e-08,  9.67622195e-16, -1.23207013e-23]]
    s = torch.dot(vector,vector) #squared norm
    print("squared norm",s)
    inverse_norm = 0.0
    x = s
    for coeff_set in coeffs:
        print("iter")
        for i, coeff in enumerate(coeff_set):
            #if abs(coeff) < 1e-8:
                #continue
            poly = coeff
            for j in range(i):
                poly = poly * x
            poly = float(poly)
            print("adding", poly)
            inverse_norm = inverse_norm + poly
        x = inverse_norm
        inverse_norm = 0.0
    inverse_norm = x
    print("inv norm",inverse_norm)
    return vector * inverse_norm

worsts = []
stds = []
for j in range(1,5):
    worst = 0
    errors = []
    for i in range(1,500):
        error = ((1/(i**0.5))-gold_test(i,None,j))/(1/(i**0.5))
        if error > worst:
            worst = error
        errors.append(error)
    mean = sum(errors)/len(errors)
    total = 0
    for error in errors:
        total += (error-mean)**2
    std = (total/len(errors))**0.5
    stds.append(std)
    print(worst)
    worsts.append(mean)
mult_depths = [9,12,15,18]
data_dict = {"Multiplicative Depth":mult_depths, r'$\frac{|G(x)-y|}{|y|}$':worsts}
df = pandas.DataFrame(data_dict)

fig = px.scatter(df,x="Multiplicative Depth",y=r'$\frac{|G(x)-y|}{|y|}$',title="Goldschmidt's Mult. Depth vs Relative Error",error_y=stds)

fig.update_yaxes(range=[0,0.25])


fig_file_name = "figures/GoldschmidtsErrors.png"
fig.write_image(fig_file_name)

fig2 = px.scatter(df,x="Multiplicative Depth",y=r'$\frac{|G(x)-y|}{|y|}$',title="Goldschmidt's Mult. Depth vs Relative Error",error_y=stds)
fig2.update_yaxes(type="log")

fig_file_name = "figures/GoldschmidtsErrorsLog.png"
fig2.write_image(fig_file_name)