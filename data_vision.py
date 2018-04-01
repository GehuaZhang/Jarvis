import numpy as np
import pandas as pd
import pickle as pkl
import pylab as plt

class DataToGraph:
    def __init__(self):
        pkl_file = open('outcome_21_10_5_5.pkl', 'rb')
        self.outcome_dataframe = pkl.load(pkl_file)

    def outcome_visulization(self, xtickNumber = 5, visionType = ""):
        outcome_dataframe = self.outcome_dataframe
        dataNumber = len(self.outcome_dataframe)

        factorFilteredList = list(set(outcome_dataframe["FactorPortfolio"]))  #将factor指标压缩成只包含不同指标的数组，如[1,2,3,4,5]
        marketFilteredList = list(set(outcome_dataframe["MarketValuePortfolio"]))
        #dateFilteredList = list(set(outcome_dataframe["trade_date"]))
        if visionType == "factor":
            for factorIndex in range(len(factorFilteredList)):
                for marketIndex in range(len(marketFilteredList)):
                    profit_series = outcome_dataframe["ret"][
                        (outcome_dataframe["FactorPortfolio"] == factorFilteredList[factorIndex]) & (
                        outcome_dataframe["MarketValuePortfolio"] == marketFilteredList[marketIndex])]
                    date_series = outcome_dataframe["trade_date"][
                        (outcome_dataframe["FactorPortfolio"] == factorFilteredList[factorIndex]) & (
                        outcome_dataframe["MarketValuePortfolio"] == marketFilteredList[marketIndex])]

                    profit_cumulate_series = np.exp(np.log(1 + profit_series).expanding(1).sum()) - 1  # 抽取累积收益Series

                    dataframe_temp = pd.DataFrame({'date': date_series.apply(lambda x: str(x)).tolist(),
                                                   'profit': profit_cumulate_series.tolist()})  # 生成一个新的dataframe来储存/此处只是为了update index

                    xticks_increment = int(
                        dataNumber / xtickNumber / len(factorFilteredList) / len(marketFilteredList))  # 设置画图步长

                    plt.plot(dataframe_temp['date'], dataframe_temp['profit'],
                             label="Market=" + str(marketFilteredList[marketIndex]))
                    plt.xticks([xticks_increment * (x) for x in range(xtickNumber)], dataframe_temp['date'].loc[
                        [xticks_increment * (x) for x in range(xtickNumber)]])  # 设置x轴刻度
                    plt.subplot(len(factorFilteredList), 1, factorIndex + 1)
                    plt.title("Factor = " + str(factorFilteredList[factorIndex]))
                    plt.xlabel('Trade Date')
                    plt.ylabel('Returns')


        if visionType == "market":
            for marketIndex in range(len(marketFilteredList)):
                for factorIndex in range(len(factorFilteredList)):

                    profit_series = outcome_dataframe["ret"][
                        (outcome_dataframe["FactorPortfolio"] == factorFilteredList[factorIndex]) & (
                        outcome_dataframe["MarketValuePortfolio"] == marketFilteredList[marketIndex])]
                    date_series = outcome_dataframe["trade_date"][
                        (outcome_dataframe["FactorPortfolio"] == factorFilteredList[factorIndex]) & (
                        outcome_dataframe["MarketValuePortfolio"] == marketFilteredList[marketIndex])]

                    profit_cumulate_series = np.exp(np.log(1 + profit_series).expanding(1).sum()) - 1  # 抽取累积收益Series

                    dataframe_temp = pd.DataFrame({'date': date_series.apply(lambda x: str(x)).tolist(),
                                                   'profit': profit_cumulate_series.tolist()})  # 生成一个新的dataframe来储存/此处只是为了update index

                    xticks_increment = int(
                        dataNumber / xtickNumber / len(factorFilteredList) / len(marketFilteredList))  # 设置画图步长

                    plt.plot(dataframe_temp['date'], dataframe_temp['profit'],
                             label="Factor=" + str(factorFilteredList[factorIndex]))
                    plt.xticks([xticks_increment * (x) for x in range(xtickNumber)], dataframe_temp['date'].loc[
                        [xticks_increment * (x) for x in range(xtickNumber)]])  # 设置x轴刻度
                    plt.subplot(len(marketFilteredList), 1, marketIndex + 1)
                    plt.title("Market = " + str(marketFilteredList[marketIndex]))
                    plt.xlabel('Trade Date')
                    plt.ylabel('Returns')

        plt.tight_layout()
        plt.show()



a = DataToGraph()
a.outcome_visulization(visionType = 'factor')
