
import numpy as np
from sklearn.linear_model import LinearRegression



def linear_approximation():
    x = [0.1 * i for i in range(21,5000)]
    y = [1/(i**0.5) for i in x]
    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    return reg.coef_, reg.intercept_
    
if __name__ == "__main__":
    print(linear_approximation())