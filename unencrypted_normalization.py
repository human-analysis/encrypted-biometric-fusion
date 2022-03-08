import tenseal as ts
#import math
import torch

import plotly.express as px
import plotly.graph_objects as go
import pandas

import os

import numpy as np

from scipy.optimize import curve_fit

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

def poly5(x, a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2):
    x = a1 + b1 * x + c1 * x**2 + d1 * x**3 + e1 * x**4 + f1 * x**5
    x = a2 + b2 * x + c2 * x**2 + d2 * x**3 + e2 * x**4 + f2 * x**5
    return x

def poly4(x, a1, b1, c1, d1, e1, a2, b2, c2, d2, e2):
    x = a1 + b1 * x + c1 * x**2 + d1 * x**3 + e1 * x**4
    x = a2 + b2 * x + c2 * x**2 + d2 * x**3 + e2 * x**4
    return x


def poly3(x, a1, b1, c1, d1, a2, b2, c2, d2):
    x = a1 + b1 * x + c1 * x**2 + d1 * x**3
    x = a2 + b2 * x + c2 * x**2 + d2 * x**3
    return x


def poly2(x, a1, b1, c1, d1, a2, b2, c2):
    x = a1 + b1 * x + c1 * x**2 + d1 * x**3
    x = a2 + b2 * x + c2 * x**2
    return x


def poly1(x, a1, b1, c1, d1, a2, b2):
    #if abs(b2) < 1e-12:
        #b2 = 0
    x = a1 + b1 * x + c1 * x**2 + d1 * x**3
    x = a2 + b2 * x
    return x

def poly0(x, a1, b1):
    x = a1 + b1 * x
    return x
    

def poly_approximation():
    #np.random.seed(1000)
    
    if not os.path.exists("figures"):
        os.mkdir("figures")
    if not os.path.exists("figures/polynomial_approximations"):
        os.mkdir("figures/polynomial_approximations")
    #x = [1.0 * i for i in range(500,4000)]
    x = [1.0 * i for i in range(1,500)]
    #x = [1 * i for i in range(1000,5000)]
    y = [1/(i**0.5) for i in x]
    x = np.array(x)#.reshape(-1,1)
    y = np.array(y)
    
    funs = [poly1, poly2, poly3, poly4]
    degrees = [4,5,6,8]
    worsts = []
    stds = []
    for i, myfun in enumerate(funs):
        print(degrees[i])
        popt, pcov = curve_fit(myfun, x, y)
        errors = []
        worst = 0
        for j in range(1,499):
            error = abs(myfun(x[j], *popt)-y[j])/y[j]
            if error > worst:
                worst = error
            errors.append(error)
        mean = sum(errors)/len(errors)
        total = 0
        #errors_list.append(errors)
        for error in errors:
            total += (error-mean)**2
        std = (total/len(errors))**0.5
        stds.append(std)
        worsts.append(mean)
        
    mult_depths_gold = [9,12,15,18]
    mult_depth_poly = [4,5,6,8]
        
    data_dict = {"Multiplicative Depth":mult_depth_poly, r'$\frac{|G(x)-y|}{|y|}$':worsts}
    df = pandas.DataFrame(data_dict)
    fig = go.Figure(layout = go.Layout(title = go.layout.Title(text="Mult. Depth vs Relative Error")))
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=data_dict["Multiplicative Depth"],
            y=data_dict[r'$\frac{|G(x)-y|}{|y|}$'],
            error_y=dict(type='data', array=stds),
            name="Polynomial"
            #marker={'color':'red'},
            #showlegend=False
        )
    )
    
        

    worsts = []
    stds = []
    errors_list = []
    y_lists = []
    for j in range(1,5):
        worst = 0
        errors = []
        y = []
        for i in range(1,500):
            y.append(gold_test(i,None,j))
            error = ((1/(i**0.5))-gold_test(i,None,j))/(1/(i**0.5))
            if error > worst:
                worst = error
            errors.append(error)
        mean = sum(errors)/len(errors)
        total = 0
        errors_list.append(errors)
        for error in errors:
            total += (error-mean)**2
        std = (total/len(errors))**0.5
        stds.append(std)
        print(worst)
        worsts.append(mean)
    
    
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=mult_depths_gold,
            y=worsts,
            error_y=dict(type='data', array=stds),
            name="Goldschmidt's Algorithm"
            #marker={'color':'red'},
            #showlegend=False
        )
    )
    
    #fig = px.scatter(df,x="Multiplicative Depth",y=r'$\frac{|G(x)-y|}{|y|}$',title="Mult. Depth vs Relative Error",error_y=stds)
    
    #fig.update_yaxes(range=[0,0.25])
    
    fig.update_yaxes(range=[0,0.35])
    
    fig.update_xaxes(title="Multiplicative Depth")
    fig.update_yaxes(title=r'$\frac{|f(x)-y|}{|y|}$')
    
    fig_file_name = "figures/ErrorComparison.png"
    fig.write_image(fig_file_name)
    print("done")
        

#poly_approximation()       

worsts = []
stds = []
errors_list = []
y_lists = []
for j in range(1,5):
    worst = 0
    errors = []
    y = []
    for i in range(1,500):
        y.append(gold_test(i,None,j))
        error = ((1/(i**0.5))-gold_test(i,None,j))/(1/(i**0.5))
        if error > worst:
            worst = error
        errors.append(error)
    mean = sum(errors)/len(errors)
    total = 0
    errors_list.append(errors)
    for error in errors:
        total += (error-mean)**2
    std = (total/len(errors))**0.5
    stds.append(std)
    y_lists.append(y)
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


true_y = [1/i**0.5 for i in range(1,500)]
x = [i for i in range(1,500)]
for i, errors in enumerate(errors_list):
    data_dict = {r'$\frac{|G(x)-y|}{|y|}$':errors, "X":x}
    df = pandas.DataFrame(data_dict)
    fig = px.line(df,x="X",y=r'$\frac{|G(x)-y|}{|y|}$',title="Goldschmidt's Relative Error Mult Depth="+str(9+3*i))
    
    #fig.update_xaxes(title_font=dict(size=18))
    fig.update_yaxes(title_font=dict(size=40))
    
    fig_file_name = "figures/goldschmidts_errors_multdepth=" + str(9+3*i) + ".png"
    fig.write_image(fig_file_name)
    
    fig = go.Figure(layout = go.Layout(title = go.layout.Title(text="Goldschmidt's, Mult Depth="+str(9+3*i))))
    fig.add_trace(
        go.Scatter(
            mode='lines',
            x=x,
            y=y_lists[i],
            marker={'color':'red'},
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            mode='lines',
            x=x,
            y=true_y,
            marker={'color':'black'},
            showlegend=False
        )
    )
    
    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y"
    )
    
    fig_file_name = "figures/goldschmidts_multdepth=" + str(9+3*i) + ".png"
    fig.write_image(fig_file_name)