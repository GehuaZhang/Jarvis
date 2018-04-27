import pandas as pd
import pickle as pkl
import numpy as np


class factor_factory:

    def __init__(self):

        pkl_file_return = open('..\dailyReturnMatrix_21_10_10.pkl', 'rb')
        pkl_file_outcome = open('..\outcome_21_10_5_5.pkl', 'rb')
        pkl_file_market_value = open('..\marketValueMatrix_21.pkl', 'rb')
        self.df_return = pkl.load(pkl_file_return)
        self.df_outcome = pkl.load(pkl_file_outcome)
        self.df_market_value = pkl.load(pkl_file_market_value)

        # pkl_file_expos = open('..\exposureMatrix_21_10_10.pkl', 'rb')
        # self.df_expos = pkl.load(pkl_file_expos)

    def factor1_maker(self):
        return

    def factor2_maker(self):
        return

    ...

    def generate_pkl(self, factor_name):
        return


a = factor_factory()
a.generate_pkl(factor_name = "factor1")
