

import tenseal as ts
import numpy as np
import time

# Setup TenSEAL context
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40


P_file_name =  "data/best_P_value_transpose_lambda=0.25_margin=0.5.txt"
P_file = open(P_file_name,'r')
P = []
for line in P_file:
    line_list = line.strip("[").strip("]").split("], [")
    for tens in line_list:
        P_vals = [float(val) for val in tens.split(", ")]
        P.append(P_vals)
P_file.close()

P_T = [list(x) for x in list(zip(*P))]

query = [1.296,1.817,3.004,-3.667,4.614]

v1 = [0, 1, 2, 3, 4]
v2 = [4, 3, 2, 1, 0]


tic = time.perf_counter()


# encrypted vectors
#enc_v1 = ts.ckks_vector(context, v1)
#enc_v2 = ts.ckks_vector(context, v2)

enc_query = ts.ckks_vector(context, query)

#result = enc_v1 + enc_v2
#result.decrypt() # ~ [4, 4, 4, 4, 4]

#result = enc_v1.dot(enc_v2)
#result.decrypt() # ~ [10]


np_query = np.matrix(query)
np_P_T = np.matrix(P_T)

np_result = np_query * np_P_T

print(np_result)

#matrix = [
#  [73, 0.5, 8],
#  [81, -5, 66],
#  [-100, -78, -2],
#  [0, 9, 17],
#  [69, 11 , 10],
#]

#result = enc_v1.matmul(matrix)

result = enc_query.matmul(P_T)
result = result.decrypt() # ~ [157, -90, 153]

toc = time.perf_counter()

print("time elapsed:",toc - tic)

print(result)