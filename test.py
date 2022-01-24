#import tenseal as ts
#import time
import enchant

from nltk.corpus import words


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
