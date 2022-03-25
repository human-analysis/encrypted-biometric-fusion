
import numpy as np
#from sklearn.linear_model import LinearRegression

#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import make_pipeline, Pipeline

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas
import ast
import os
from plotly.subplots import make_subplots
import imageio


#def poly(x, a1, b1, c1, d1, e1, a2, b2, c2, d2, e2):

    
def poly6(x, a1, b1, c1, d1, e1, f1, g1, a2, b2, c2, d2, e2, f2):
    x = a1 + b1 * x + c1 * x**2 + d1 * x**3 + e1 * x**4 + f1 * x**5 + g1 * x**6
    x = a2 + b2 * x + c2 * x**2 + d2 * x**3 + e2 * x**4 + f2 * x**5
    return x

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
    """
    if abs(a1) < 1e12:
        a1 = 0
    if abs(b1) < 1e12:
        b1 = 0
    if abs(c1) < 1e12:
        c1 = 0
    if abs(d1) < 1e12:
        d1 = 0
    if abs(a2) < 1e12:
        a2 = 0"""
    """
    if abs(b2) < 1e12:
        b2 += 10
    if abs(d1) < 1e12:
        d1 += 10"""
    x = a1 + b1 * x + c1 * x**2 + d1 * x**3
    x = a2 + b2 * x
    return x
    
def polyhalf2(x, a1, b1, c1, d1):
    x = a1 + b1 * x + c1 * x**2 + d1 * x**3
    return x

def polyhalf(x, a1, b1, c1):
    x = a1 + b1 * x + c1 * x**2
    return x


def poly0(x, a1, b1):
    """if abs(a1) < 1e12:
        a1 = 0
    if abs(b1) < 1e12:
        b1 = 0
    """
    x = a1 + b1 * x
    return x

def constant(x, a1):
    return a1
    

def poly_approximation():
    #np.random.seed(1000)
    
    if not os.path.exists("figures"):
        os.mkdir("figures")
    if not os.path.exists("figures/polynomial_approximations"):
        os.mkdir("figures/polynomial_approximations")
    #x = [1.0 * i for i in range(500,4000)]
    x = [1.0 * i for i in range(1,500)]
    x = [0.1 * i for i in range(10,1500)]
    x = [0.1 * i for i in range(810,13500)]
    #x = [1.0 * i for i in range(int(12e3),int(19e3),100)]
    #x = [1.0 * i for i in range(30000000, 50000000,10000)]
    #x = [1.0 * i for i in range(int(6.4e9), int(1.69e10),10000000)]
    
    #x = [1.0 * i for i in range(600, 1600,1)]
    #x = [1.0 * i for i in range(int(6e5), int(1.8e6),int(1e4))]
    print(len(x))
    #x = [1 * i for i in range(1000,5000)]
    y = [1/(i**0.5) for i in x]
    x = np.array(x)#.reshape(-1,1)
    y = np.array(y)
    
    funs = [poly0, polyhalf, polyhalf2, poly1, poly2, poly3, poly4, poly6]
    degrees = [1,2,3,4,5,6,8,11]
    for i, myfun in enumerate(funs):
        popt, pcov = curve_fit(myfun, x, y, maxfev=100000)
        
        #popt, pcov = curve_fit(poly, x, y,bounds=((1e-30,1e-30,1e-30,1e-30,1e-30,1e-30), (np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)))
        
        print(degrees[i],popt, sum(abs(myfun(x, *popt)-y)/y)/len(abs(myfun(x, *popt)-y)/y))
        
        #new_y = [poly(i, *popt) for i in x]
        #x = [0.1 * i for i in range(5000,40000)]
        #new_y = []
        #for i in x:
            #result = poly(i, *popt)
            #print(i, result)
            #new_y.append(result)
        #print(x)
        #plt.plot(x, myfun(x, *popt), 'r-', label="label")
        #plt.plot(x, new_y, 'r-', label="label")
        #plt.plot(x, y, 'r-', label="label")
        
        #plt.show()
        
        
        
        fig = go.Figure(layout = go.Layout(title = go.layout.Title(text="Polynomial Approximation, Degree="+str(degrees[i]))))
        fig.add_trace(
            go.Scatter(
                mode='lines',
                x=x,
                y=myfun(x, *popt),
                marker={'color':'red'},
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                mode='lines',
                x=x,
                y=y,
                marker={'color':'black'},
                showlegend=False
            )
        )
        
        fig.update_layout(
            xaxis_title="X",
            yaxis_title="Y"
        )
        
        fig_file_name = "figures/polynomial_approximations/approx_degree=" + str(degrees[i]) + ".png"
        fig.write_image(fig_file_name)
        
        data_dict = {r'$\frac{|f(x)-y|}{|y|}$':abs(myfun(x, *popt)-y)/y, "X":x}
        df = pandas.DataFrame(data_dict)
        fig = px.line(df,x="X",y=r'$\frac{|f(x)-y|}{|y|}$')
        
        #fig.update_xaxes(title_font=dict(size=18))
        fig.update_yaxes(title_font=dict(size=40))
        
        
        fig_file_name = "figures/polynomial_approximations/approx_error_degree=" + str(degrees[i]) + ".png"
        fig.write_image(fig_file_name)
        
        #data_dict = {"Y":myfun(x, *popt), "X":x}
        #df = pandas.DataFrame(data_dict)
        #fig = px.line(df,x="X",y="Y")
        
        
    #print(poly(4000, *popt), y[3499])
    
    """
    reg = LinearRegression().fit(x, y)
    
    degree=6
    #polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg = Pipeline(memory=None, steps = [("poly", PolynomialFeatures(degree)), ("linear",LinearRegression())])
    polyreg.fit(x,y)
    
    #polyreg = PolynomialFeatures(degree)
    #polyreg.fit(x,y)
    
    print(polyreg.named_steps["linear"].coef_)
    print(polyreg.named_steps["linear"].intercept_)
    print(polyreg.named_steps["poly"].get_feature_names())
    print(polyreg.steps[1][1].coef_)
    
    plt.figure()
    plt.scatter(x,y)
    plt.plot(x,polyreg.predict(x),color="black")
    #plt.plot(x,polyreg.transform(x),color="black")
    plt.title("Polynomial regression with degree "+str(degree))
    plt.show()
    
    
    
    return reg.coef_, reg.intercept_
    """
    
if __name__ == "__main__":
    poly_approximation()
