import pandas as pd
import pickle as pkl

pkl_file = open('marketValueMatrix_21.pkl', 'rb')

a = pkl.load(pkl_file)

print(a)