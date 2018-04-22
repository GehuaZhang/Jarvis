import pandas as pd
import pickle as pkl

pkl_file = open('outcome_21_10_5_5.pkl', 'rb')

a = pkl.load(pkl_file)

print(a)