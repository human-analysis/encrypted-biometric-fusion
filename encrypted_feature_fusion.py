

import tenseal as ts
import time

from data_generation import data_gen
import torch

from encrypted_normalization import normalize, efficient_normalize

import os




def Inference_Time_Encrypted_Fusion():
    
    
    """
    ##unencrypted test
    context = None
    vector = torch.tensor([-51.6411,  24.2202])
    print("normalized:",efficient_normalize(vector, context, dimensionality=5))
    #vector = torch.tensor([37.4037, -21.0916])
    s = torch.dot(vector,vector)
    #print("norm:",s**0.5)
    inverse_norm = 1/(s**0.5)
    print("true inverse norm:",inverse_norm)
    print("truth:",vector * inverse_norm)
    0/0
    """
    
    if not os.path.exists("results"):
        os.mkdir("results")
    
    
    m_poly_mod = 32768
    #m_coeff_mod = [60]
    #m_poly_mod = 32768//2
    m_coeff_mod = [60]
    #for i in range(19-2):#-12):
    
    #for i in range(15):
    #for i in range(15):
    for i in range(13):
        m_coeff_mod.append(40)
    m_coeff_mod.append(60)
    
    tic = time.perf_counter()
    
    # Setup TenSEAL context
    context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=m_poly_mod,
                coeff_mod_bit_sizes=m_coeff_mod,
              )
    context.generate_galois_keys()
    context.global_scale = 2**80
    
    
    toc = time.perf_counter()
    
    
    print("time elapsed to create context:",toc - tic)
    
    
    
    ##encrypted test
    #
    #[37.4037, -21.0916]
    print("encrypted:",efficient_normalize(ts.ckks_vector(context,[-51.6411,  24.2202]), context, dimensionality=5).decrypt())
    vector = torch.tensor([-51.6411,  24.2202])
    s = torch.dot(vector,vector)
    print("true norm:",s**0.5)
    inverse_norm = 1/(s**0.5)
    print("truth:",vector * inverse_norm)
    0/0
    
    
    
    
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
    
    num_classes = 4
    samples_per_class = 10
    count = num_classes*samples_per_class
    
    print("Number of queries:",count)
    
    A_test,B_test,L_test = data_gen(samples_per_class)
    X_test = torch.cat((A_test,B_test),dim=1)
    
    enc_queries = []
    for i in range(count):
        enc_queries.append(ts.ckks_vector(context, X_test[i]))
    
    
    
    
    tic = time.perf_counter()
    
    results = []
    for i in range(count):
        enc_query = enc_queries[i]
        result = enc_query.matmul(P_T)
        result = normalize(result, context, dimensionality=5)
        #result = result.decrypt()
        #result = torch.tensor(result)
        results.append(result)
    
    toc = time.perf_counter()
    
    
    print("time elapsed to perform encrypted fusion:",toc - tic)
    
    tic = time.perf_counter()
    
    plain_results = []
    torch_P_T = torch.tensor(P_T)
    for i in range(count):
        query = X_test[i]
        plain_result = torch.matmul(query, torch_P_T)
        norm = torch.dot(plain_result,plain_result)
        plain_result = plain_result / norm**0.5
        plain_results.append(plain_result)
    
    toc = time.perf_counter()
    
    print("time elapsed to perform unencrypted fusion:",toc - tic)
    
    
    dec_results = [torch.tensor(result.decrypt()) for result in results]
    
    total_error = 0
    for i in range(len(results)):
        total_error += abs(dec_results[i] - plain_results[i])
    avg_error = total_error/count
    
    outfile_name = "results/toy_data_1_2.txt"
    outfile = open(outfile_name, 'w')
    for i  in range(count):
        result = dec_results[i]
        outfile.write(str(result.tolist()))
        outfile.write(";")
        outfile.write(str(int(L_test[i].item())))
        outfile.write("\n")
    outfile.close()
    
    outfile_name_plain = "results/toy_data_1_2_plain.txt"
    outfile_plain = open(outfile_name_plain, 'w')
    for i  in range(count):
        result = plain_results[i]
        outfile_plain.write(str(result.tolist()))
        outfile_plain.write(";")
        outfile_plain.write(str(int(L_test[i].item())))
        outfile_plain.write("\n")
    outfile_plain.close()
    
    
    print("Average error:",avg_error)
    print("Total error:",total_error)
    
    print("sample encrypted result:",dec_results[1])
    print("sample plain result:",plain_results[1])
    

if __name__ == "__main__":
    Inference_Time_Encrypted_Fusion()