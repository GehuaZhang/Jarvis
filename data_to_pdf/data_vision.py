import numpy as np
import pandas as pd
import pickle as pkl
import pylab as plt
import itertools
from matplotlib.backends.backend_pdf import PdfPages


class DataToGraph:

    def __init__(self):
        pkl_file = open('outcome_21_10_5_5.pkl', 'rb')
        self.outcome_dataframe = pkl.load(pkl_file)

    def daily_return_visulization(self, xtickNumber = 5, visionType = ""):
        global pic_temp
        outcome_dataframe = self.outcome_dataframe
        dataNumber = len(self.outcome_dataframe)

        factorFilteredList = list(set(outcome_dataframe["FactorPortfolio"]))  #将factor指标压缩成只包含不同指标的数组，如[1,2,3,4,5]
        marketFilteredList = list(set(outcome_dataframe["MarketValuePortfolio"]))

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

                    #Draw Pic
                    plt.figure(1, figsize=(20, 30))
                    pic_temp = plt.subplot(len(factorFilteredList), 1, factorIndex + 1)
                    plt.plot(dataframe_temp['date'], dataframe_temp['profit'],
                             label="Market=" + str(marketFilteredList[marketIndex]))

                pic_temp.legend()
                xticks_increment = int(
                    dataNumber / xtickNumber / len(factorFilteredList) / len(marketFilteredList))  # 设置画图步长
                plt.xticks([xticks_increment * (x) for x in range(xtickNumber)], dataframe_temp['date'].loc[
                    [xticks_increment * (x) for x in range(xtickNumber)]])  # 设置x轴刻度
                plt.xlabel('Trade Date')
                plt.ylabel('Returns')
                plt.title("Factor = " + str(factorFilteredList[factorIndex]))

            plt.tight_layout()
            figure_temp = plt.figure(1)
            return figure_temp

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
                    #Draw Pic
                    plt.figure(2, figsize=(20, 30))
                    pic_temp = plt.subplot(len(marketFilteredList), 1, marketIndex + 1)
                    plt.plot(dataframe_temp['date'], dataframe_temp['profit'],
                             label="Factor=" + str(factorFilteredList[factorIndex]))

                pic_temp.legend()
                xticks_increment = int(
                    dataNumber / xtickNumber / len(factorFilteredList) / len(marketFilteredList))  # 设置画图步长
                plt.xticks([xticks_increment * (x) for x in range(xtickNumber)], dataframe_temp['date'].loc[
                    [xticks_increment * (x) for x in range(xtickNumber)]])  # 设置x轴刻度
                plt.title("Market = " + str(marketFilteredList[marketIndex]))
                plt.xlabel('Trade Date')
                plt.ylabel('Returns')

            plt.tight_layout()
            figure_temp = plt.figure(2)
            return figure_temp

    def table_visulization(self):
        outcome_dataframe = self.outcome_dataframe
        factor_filtered_list = list(set(outcome_dataframe["FactorPortfolio"]))
        market_filtered_list = list(set(outcome_dataframe["MarketValuePortfolio"]))
        factor_count, market_count = len(factor_filtered_list), len(market_filtered_list)

        indicator = ['annual_return', 'annual_volatility', 'annual_sharpe', 'max_retrace', 'max_retrace_date', 'max_win_streak_number',
                                                                            'max_lose_streak_number', 'profit_to_lose_percentage',
                                                                            'profit_to_lose', 'turnover_rate']

        data_storage = pd.DataFrame(np.zeros((factor_count*market_count,len(indicator))),
                                    index=[(x, y) for x in factor_filtered_list for y in market_filtered_list], columns = indicator)

        for factorIndex in range(factor_count):
            for marketIndex in range(market_count):
                profit_series = outcome_dataframe["ret"][
                    (outcome_dataframe["FactorPortfolio"] == factor_filtered_list[factorIndex]) & (
                        outcome_dataframe["MarketValuePortfolio"] == market_filtered_list[marketIndex])]

                symbol_series = outcome_dataframe["symbols"][
                    (outcome_dataframe["FactorPortfolio"] == factor_filtered_list[factorIndex]) & (
                        outcome_dataframe["MarketValuePortfolio"] == market_filtered_list[marketIndex])]

                return_temp = profit_series.apply(lambda x: (1+x)).product()
                data_storage['annual_return'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = \
                    return_temp**(252/len(profit_series)) - 1
                data_storage['annual_volatility'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = \
                    profit_series.std()*np.sqrt(252)
                data_storage['annual_sharpe'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = \
                    profit_series.mean()/profit_series.std()*np.sqrt(252)
                max_retrace = profit_series.min()
                data_storage['max_retrace'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = max_retrace
                data_storage['max_retrace_date'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = \
                    outcome_dataframe['trade_date'][(outcome_dataframe['ret'] == max_retrace)].to_string()

                data_storage['max_win_streak_number'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = \
                    max([len(list(v)) for k, v in itertools.groupby(profit_series > 0)])
                data_storage['max_lose_streak_number'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = \
                    max([len(list(v)) for k, v in itertools.groupby(profit_series < 0)])

                profit_count, lose_count = len(profit_series[profit_series > 0]), len(profit_series[profit_series < 0])
                profit_return, lose_return = profit_series[profit_series > 0].apply(lambda x: 1 + x).product() - 1, \
                                             profit_series[profit_series < 0].apply(lambda x: 1 + x).product() - 1

                data_storage['profit_to_lose_percentage'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] =\
                    (profit_return / profit_count) / (abs(lose_return) / lose_count)
                data_storage['profit_to_lose'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = profit_count/lose_count
                data_storage['turnover_rate'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = 0

        for tableIndex in range(len(data_storage.columns)):
            series_temp = data_storage[data_storage.columns[tableIndex]].values.reshape(factor_count, market_count)

            plt.figure(3, figsize=(20, 40))
            pic_temp = plt.subplot(len(data_storage.columns), 1, tableIndex+1)
            pic_temp.axis('off')
            pic_temp.axis('tight')
            table_temp = pic_temp.table(cellText=series_temp, loc='center', cellLoc='center', rowLabels=factor_filtered_list,
                                        colLabels = market_filtered_list, colLoc='center', colWidths=[0.1]*len(market_filtered_list))
            plt.title(data_storage.columns[tableIndex], loc='left', size=25)
            table_temp.set_fontsize(20)
            table_temp.scale(2, 2)

        plt.tight_layout()
        figure_temp = plt.figure(3)

        return figure_temp

    def generate_pdf(self):
        figure1 = self.daily_return_visulization(visionType="factor")
        figure2 = self.daily_return_visulization(visionType="market")
        figure3 = self.table_visulization()

        pp = PdfPages('Graph.pdf')
        pp.savefig(figure1)
        pp.savefig(figure2)
        pp.savefig(figure3)
        pp.close()

pd.options.mode.chained_assignment = None
a = DataToGraph()
#a.table_visulization()
a.generate_pdf()