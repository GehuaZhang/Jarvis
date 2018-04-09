import pandas as pd
import pickle as pkl



pkl_file = open('dailyReturnMatrix_21_10_10.pkl', 'rb')

a = pkl.load(pkl_file)

print(a)