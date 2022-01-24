import tenseal as ts
import math

def reencrypt(enc_vector, context):
    plain = enc_vector.decrypt()
    #print("RE ENCRYPTING:", plain)
    return ts.ckks_vector(context,plain)

def Goldschmidt(s, initial_estimate, context, iters):
    #multiplicative depth = logk + 3(l1+l2) + 2
    #mults = 13
    x = s * initial_estimate
    h = initial_estimate * [0.5]
    for i in range(iters):
        
        #print("gold iter",i)
        #print(h.decrypt())
        r = -x*h + [0.5]
        x = x + x*r
        h = h + h*r
        #if i==1:
            #x = reencrypt(x, context)
            #h = reencrypt(h, context)
            #r = reencrypt(r, context)
        
        
    return h*[2]#times two added by me but why?

def NewtonsMethod(linear_approximation, s, context, iters):
    #multiplicative depth = 3*iters + 1
    x = linear_approximation
    for i in range(iters):
        #print("iter", i)
        #print(x.decrypt())
        x = x * [0.5]
        #print("sub negate")
        subformula = -s
        #print("mult")
        subformula = subformula * x
        #print("mult")
        subformula = subformula * x
        subformula = reencrypt(subformula, context)
        #subformula = subformula * x**2
        #subformula = subformula.mul(x)
        subformula = subformula + [3]
        #x = x * ( * x * x + [3])
        #print("mult")
        x = x * subformula
        x = reencrypt(x, context)
    return x

def InvNormApprox(linear_approximation, s, context, iters):
    #five mults? four maybe
    neg_half = [-0.5]
    three_half = [1.5]
    guess = linear_approximation
    sq = s #?
    for i in range(iters):
        sq = sq * guess * guess
        sq = guess * neg_half * sq
        temp = three_half * guess
        guess = temp + sq
    return guess

def normalize(enc_vector, context, dimensionality):
    
    #ts.ckks_vector(context, )
    #small = ts.ckks_vector(context, [5])
    #big = ts.ckks_vector(context, [1,2,3])
    #combined = small * big
    #print("test:",combined.decrypt())
    
    s = enc_vector.dot(enc_vector)#enc_vector * enc_vector#copy.deepcopy(enc_vector)
    #for i in range(math.log(dimensionality,2)):
        #s
    #ts.ckks_vector(context, X_test[i])
    
    #these linear approximation constants from Principal Component Analysis using CKKS Homomorphic Scheme, Samanvaya Panda,
    #International Symposium on Cyber Security Cryptography and Machine Learning, 2021
    #linear_weight = [-0.00019703]
    #linear_bias = [0.14777278]
    
    #these values obtained by my own training of a linear regressor
    #linear_weight = -4.40369448e-05
    #linear_bias = 0.08852906761144604
    linear_weight = -4.06884198e-05
    linear_bias = 0.08406359237110503
    
    #print("mult")
    linear_approximation = linear_weight * s
    linear_approximation = linear_approximation + linear_bias
    
    linear_approximation = ts.ckks_vector(context,[0.01]) #constant first guess
    
    initial_estimate = InvNormApprox(linear_approximation, s, context, iters=1)
    #initial_estimate = reencrypt(initial_estimate, context)
    
    
    #inverse_norm = initial_estimate
    inverse_norm = Goldschmidt(s, initial_estimate, context, iters=2)
    return enc_vector * inverse_norm

"""
def efficient_normalize(enc_vector, context, dimensionality):
    #0.8178302654835186
    coeffs = [ 0.00000000e+00, -1.00922126e-01,  7.19802379e-03, -2.59873687e-04,
      4.52208769e-06, -3.01334013e-08]
    s = enc_vector.dot(enc_vector) #squared norm
    inverse_norm = ts.ckks_vector(context,[0.0])
    for i, coeff in enumerate(coeffs):
        if abs(coeff) < 1e-8:
            continue
        poly = ts.ckks_vector(context,[coeff])
        for j in range(i):
            poly = poly * s
        inverse_norm = inverse_norm + poly
    return enc_vector * inverse_norm
"""

def efficient_normalize(enc_vector, context, dimensionality):
    coeffs = [[-1.74906337e+06,  5.73576978e+04, -2.63561506e+01,  3.31516365e-03],
      [2.14909489e-01, -2.42205486e-08,  9.67622195e-16, -1.23207013e-23]]
    s = enc_vector.dot(enc_vector) #squared norm
    print("squared norm", s.decrypt())
    inverse_norm = ts.ckks_vector(context,[0.0])
    x = s
    for coeff_set in coeffs:
        print("iter")
        for i, coeff in enumerate(coeff_set):
            #print("coeff:", coeff)
            #if abs(coeff) < 1e-8:
                #continue
            poly = ts.ckks_vector(context, [coeff])
            for j in range(i):
                poly = poly * x
            print("adding", poly.decrypt()[0])
            inverse_norm = inverse_norm + poly
        x = inverse_norm
        inverse_norm = ts.ckks_vector(context,[0.0])
    inverse_norm = x
    print("inv norm",inverse_norm.decrypt())
    return enc_vector * inverse_norm

