# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:10:10 2022

@author: Luke
"""

def taylor_series_plain(x):
    return 1 - 0.5*(x-1) + (3/8)*(x-1)**2 - (5/16)*(x-1)**2 + (35/128)*(x-1)**4


print(taylor_series_plain(7))
print(1/7**0.5)