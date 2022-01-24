
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


#def poly(x, a1, b1, c1, d1, e1, a2, b2, c2, d2, e2):
"""
def poly(x, a1, b1, c1, d1, a2, b2, c2, d2):
    x = a1 + b1 * x + c1 * x**2 + d1 * x**3# + e1 * x**4
    x = a2 + b2 * x + c2 * x**2 + d2 * x**3# + e2 * x**4
    return x
"""
def poly(x, a1, b1, c1, a2, b2, c2):
    x = a1 + b1 * x + c1 * x**2
    x = a2 + b2 * x + c2 * x**2
    return x

def linear_approximation():
    x = [1 * i for i in range(100,5000)]
    y = [1/(i**0.5) for i in x]
    x = np.array(x)#.reshape(-1,1)
    y = np.array(y)
    
    popt, pcov = curve_fit(poly, x, y)
    
    print(popt)
    
    plt.plot(x, poly(x, *popt), 'r-', label="label")
    plt.plot(x, y, 'r-', label="label")
    
    plt.show()
    
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
    print(linear_approximation())