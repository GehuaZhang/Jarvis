import pandas as pd
import pickle as pkl
from sklearn.linear_model import LinearRegression
import numpy as np


class Regression:

    def __init__(self):
        pkl_file_expos = open('..\exposureMatrix_21_10_10.pkl', 'rb')
        pkl_file_return = open('..\dailyReturnMatrix_21_10_10.pkl', 'rb')
        pkl_file_outcome = open('..\outcome_21_10_5_5.pkl', 'rb')
        pkl_file_market_value = open('..\marketValueMatrix_21.pkl', 'rb')
        self.df_return = pkl.load(pkl_file_return)
        self.df_expos = pkl.load(pkl_file_expos)
        self.df_outcome = pkl.load(pkl_file_outcome)
        self.df_market_value = pkl.load(pkl_file_market_value)

    def get_dic_key(self, dict_temp):
        return

    def reset_index(self, df_temp, index_name):  # Reset index and remove it from previous dataframe
        df_temp.index = df_temp[index_name].apply(lambda x: str(x)).tolist()
        df_temp = df_temp.drop(labels=index_name, axis=1)
        return df_temp

    def expos_initialize(self):

        df_expos = self.df_expos.dropna(axis=1, how='any')  # Drop NaN Values
        df_expos = self.reset_index(df_expos, index_name="trade_date")
        expos_size = df_expos.shape

        expos_dev_sup, expos_dev_inf = df_expos.median() + 5*(abs(df_expos - df_expos.median())).median(), \
                                       df_expos.median() - 5*(abs(df_expos - df_expos.median())).median()  # dev = median(|x-median(x)|)

        # 如果dataframe与series直接比较，赋值时维度出现问题。此处先expand series到dataframe的维度，再两个dataframe直接比较。
        # 此处创建dataframe再赋值的原因：避免循环 速度快的多
        df_expos[df_expos > expos_dev_sup] = pd.DataFrame([expos_dev_sup.tolist()]*expos_size[0], index=df_expos.index, columns=df_expos.columns)
        df_expos[df_expos < expos_dev_inf] = pd.DataFrame([expos_dev_inf.tolist()]*expos_size[0], index=df_expos.index, columns=df_expos.columns)

        df_expos = ((df_expos - df_expos.mean())/df_expos.std(ddof=1)).dropna(axis=1, how='any')
        return df_expos

    def market_filter(self, trade_date, symbol_list):
        df_outcome = self.reset_index(self.df_outcome, index_name="trade_date")

        if not df_outcome.index.contains(trade_date):
            print("No Such Trade Date In outcome.pkl.")
            return

        market_label = list(set(df_outcome['MarketValuePortfolio']))
        factor_label = list(set(df_outcome['FactorPortfolio']))

        df_market_label = pd.DataFrame(np.zeros((len(symbol_list), len(market_label))), index=symbol_list,
                                 columns=market_label)

        # 生成关于symbol_list的字典
        dic_symbol = {}
        for market_index in market_label:
            for factor_index in factor_label:
                dic_symbol[(market_index, factor_index)] = df_outcome['symbols'][(df_outcome["MarketValuePortfolio"] == market_index) & \
                                                              (df_outcome["FactorPortfolio"] == factor_index)].iloc[0]

        # 把symbol_list中有的元素更新到df_market
        # 没有在outcome.pkl中的元素，但在expos中的元素，默认所有MV系数为0
        for symbols in symbol_list:
            for keys, values in dic_symbol.items():
                if symbols in values:
                    market_index, _ = keys
                    df_market_label[market_index].loc[symbols] = 1.0
                    break
                else:
                    continue
        """
        for symbols in symbol_list:
            a = df_outcome['MarketValuePortfolio'][df_outcome['symbols'].apply(lambda x: symbols in x)]
            if a.index.isin([trade_date]).any() :
                market_index = df_outcome['MarketValuePortfolio'][df_outcome['symbols'].apply(lambda x: symbols in x)].loc[trade_date]  # 查找该股票对应的市值值
                df_market[market_index].loc[symbols] = 1.0
                continue
        """
        return df_market_label

    def pnl_filter(self, trade_date, symbol_list):
        df_return = self.reset_index(self.df_return, index_name="trade_date")
        if not df_return.index.contains(trade_date):
            print("No Such Trade Date In dailyReturn.pkl.")
            return

        df_pnl = df_return[symbol_list].loc[trade_date]
        return df_pnl

    def weight_fither(self, trade_date, symbol_list):
        df_market_value = self.reset_index(self.df_market_value, index_name="trade_date")
        if not df_market_value.index.contains(trade_date):
            print("No Such Trade Date In marketValueMatrix.pkl.")
            return

        df_market_value = df_market_value[symbol_list].loc[trade_date]
        return df_market_value

    def linear_model(self):
        df_expos = self.expos_initialize()
        avail_date = df_expos.index.tolist()
        avail_symbol = df_expos.columns.tolist()
        df_expos_T = df_expos.T  #  转置

        df_model = pd.DataFrame(index=avail_date, columns=["model_coef","intercept"])

        for date_temp in avail_date:
            print("trade_date: "+date_temp)
            sr_pnl = self.pnl_filter(date_temp, symbol_list=avail_symbol)                 # StockNumber x 1
            sr_expos = df_expos_T[date_temp]                                              # StockNumber x 1
            df_market_label = self.market_filter(date_temp, symbol_list=avail_symbol)     # StockNumber x MarketIndexNumber
            df_market_value = self.weight_fither(date_temp, symbol_list=avail_symbol)     # StockNumber x 1

            df_x = pd.concat([sr_expos, df_market_label], axis=1)
            df_y = sr_pnl
            df_weight = df_market_value**(-0.5)
            df_weight = df_weight.fillna(0) #NaN值被替代为0

            # Regression
            regre_model = LinearRegression()
            regre_model.fit(df_x, df_y, sample_weight=df_weight)

            df_model["model_coef"].loc[date_temp] = [regre_model.coef_.tolist()]
            df_model["intercept"].loc[date_temp] = [regre_model.intercept_.tolist()]

        return df_model

pd.options.mode.chained_assignment = None
a = Regression()
print(a.linear_model())