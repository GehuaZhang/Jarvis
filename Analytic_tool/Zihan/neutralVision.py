# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:42:13 2018

@author: Alan
"""

import numpy as np
import pandas as pd
import pickle as pkl
import pylab as plt
#import itertools
from matplotlib.backends.backend_pdf import PdfPages
#from matplotlib.table import Table
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt2

'''
====================== next =============================
'''


pkl_file = open(r'G:\MultiFactor\Factor\Momentum\outcome_MV_21_10_10_5.pkl', 'rb')
outcome_dataframe = pkl.load(pkl_file)
file_name = pkl_file.name[-14:-4]
pkl_file.close()

def MarketNeutral_visualization(df = outcome_dataframe, visionType = "factor",
                               xtickNumber = 10, file_name = file_name ):
    global pic_temp
    outcome_dataframe = df.reset_index('trade_date')
    outcome_dataframe['trade_date'] = outcome_dataframe['trade_date'].apply(lambda x : int(x))
    
    dataNumber = len(outcome_dataframe)
    
    factorFilteredList = list(set(outcome_dataframe["FactorPortfolio"]))  #将factor指标压缩成只包含不同指标的数组，如[1,2,3,4,5]
    marketFilteredList = list(set(outcome_dataframe["MarketValuePortfolio"]))
        
    if visionType == "factor":
        for factorIndex in range(len(factorFilteredList)):
            for marketIndex in range(len(marketFilteredList)):
                profit_series = outcome_dataframe["ret"][
                        (outcome_dataframe["FactorPortfolio"] == factorFilteredList[
                        factorIndex]) & (outcome_dataframe[
                        "MarketValuePortfolio"] == marketFilteredList[
                        marketIndex])]
                date_series = outcome_dataframe["trade_date"][
                        (outcome_dataframe["FactorPortfolio"] == factorFilteredList[
                        factorIndex]) & (outcome_dataframe[
                        "MarketValuePortfolio"] == marketFilteredList[marketIndex])]

                profit_cumulate_series = np.exp(np.log(1 + profit_series).expanding(1).sum()) - 1  # 抽取累积收益Series

                dataframe_temp = pd.DataFrame({'date': date_series.apply(lambda x: str(x)).tolist(),
                                               'profit': profit_cumulate_series.tolist()})  # 生成一个新的dataframe来储存/此处只是为了update index

                #Draw Pic
                plt.figure(1, figsize=(20, 60))
                pic_temp = plt.subplot(len(factorFilteredList), 1, factorIndex + 1)
                plt.plot(dataframe_temp['date'], dataframe_temp['profit'],
                         label="MarketPortfolio = %s" % str(marketFilteredList[marketIndex]))
                plt.legend()
                #神奇的 plt.legend()，没有的话就打不出 label

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
        
        pp = PdfPages(r'G:\MultiFactor\Factor\Momentum\graph_mvNeutral%s_%s.pdf' % (file_name, visionType))
        
        #注意，这里没有 利用 outcome_MV_21_10_5_5.pkl
        
        pp.savefig(figure_temp)
        pp.close()

    if visionType == "market":
        for marketIndex in range(len(marketFilteredList)):
            for factorIndex in range(len(factorFilteredList)):

                profit_series = outcome_dataframe["ret"][(outcome_dataframe[
                        "FactorPortfolio"] == factorFilteredList[factorIndex]) & (
                        outcome_dataframe["MarketValuePortfolio"] == marketFilteredList[
                        marketIndex])]
                date_series = outcome_dataframe["trade_date"][(outcome_dataframe[
                        "FactorPortfolio"] == factorFilteredList[factorIndex]) & (
                        outcome_dataframe["MarketValuePortfolio"] == marketFilteredList[
                        marketIndex])]

                profit_cumulate_series = np.exp(np.log(1 + profit_series).expanding(1).sum()) - 1  # 抽取累积收益Series

                dataframe_temp = pd.DataFrame({'date': date_series.apply(lambda x: str(x)).tolist(),
                                                   'profit': profit_cumulate_series.tolist()})  # 生成一个新的dataframe来储存/此处只是为了update index
                #Draw Pic
                plt.figure(2, figsize=(20, 30))
                pic_temp = plt.subplot(len(marketFilteredList), 1, marketIndex + 1)
                plt.plot(dataframe_temp['date'], dataframe_temp['profit'],
                         label="FactorPortfolio = %s" % str(factorFilteredList[factorIndex]))
                plt.legend()
                
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
        
        pp = PdfPages(r'G:\MultiFactor\Factor\Momentum\graph_mvNeutral%s_%s.pdf' % (file_name, visionType))
        pp.savefig(figure_temp)
        pp.close()

MarketNeutral_visualization(df = outcome_dataframe, 
                           visionType = "factor",
                           xtickNumber = 10, 
                           file_name = file_name )

'''
====================== next =============================
'''


pkl_file = open(r'G:\MultiFactor\Factor\Momentum\outcome_Ind_21_10_5.pkl', 'rb')
outcome_dataframe = pkl.load(pkl_file)
file_name = pkl_file.name[-12:-4]
pkl_file.close()

def IndustryNeutral_visualization(df = outcome_dataframe, visionType = "factor",
                               xtickNumber = 10, file_name = file_name ):
    global pic_temp
    outcome_dataframe = df.reset_index('trade_date')
    outcome_dataframe['trade_date'] = outcome_dataframe['trade_date'].apply(lambda x : int(x))
    
    dataNumber = len(outcome_dataframe)
    
    factorFilteredList = list(set(outcome_dataframe["FactorPortfolio"]))  #将factor指标压缩成只包含不同指标的数组，如[1,2,3,4,5]
    industryFilteredList = list(set(outcome_dataframe["IndustryPortfolio"]))
        
    if visionType == "factor":
        for factorIndex in range(len(factorFilteredList)):
            for industryIndex in range(len(industryFilteredList)):
                profit_series = outcome_dataframe["ret"][
                        (outcome_dataframe["FactorPortfolio"] == factorFilteredList[
                        factorIndex]) & (outcome_dataframe[
                        "IndustryPortfolio"] == industryFilteredList[
                        industryIndex])]
                date_series = outcome_dataframe["trade_date"][
                        (outcome_dataframe["FactorPortfolio"] == factorFilteredList[
                        factorIndex]) & (outcome_dataframe[
                        "IndustryPortfolio"] == industryFilteredList[industryIndex])]

                profit_cumulate_series = np.exp(np.log(1 + profit_series).expanding(1).sum()) - 1  # 抽取累积收益Series

                dataframe_temp = pd.DataFrame({'date': date_series.apply(lambda x: str(x)).tolist(),
                                               'profit': profit_cumulate_series.tolist()})  # 生成一个新的dataframe来储存/此处只是为了update index

                #Draw Pic
                plt.figure(1, figsize=(20, 180))
                pic_temp = plt.subplot(len(factorFilteredList), 1, factorIndex + 1)
                plt.plot(dataframe_temp['date'], dataframe_temp['profit'],
                         label="IndustryPortfolio = %s" % str(industryFilteredList[industryIndex]))
                plt.legend()
                #神奇的 plt.legend()，没有的话就打不出 label

            pic_temp.legend()
            xticks_increment = int(
            dataNumber / xtickNumber / len(factorFilteredList) / len(industryFilteredList))  # 设置画图步长
            plt.xticks([xticks_increment * (x) for x in range(xtickNumber)], dataframe_temp['date'].loc[
                    [xticks_increment * (x) for x in range(xtickNumber)]])  # 设置x轴刻度
            plt.xlabel('Trade Date')
            plt.ylabel('Returns')
            plt.title("Factor = " + str(factorFilteredList[factorIndex]))

        plt.tight_layout()
        figure_temp = plt.figure(1)
        
        pp = PdfPages(r'G:\MultiFactor\Factor\Momentum\graph_industryNeutral%s_%s.pdf' % (file_name, visionType))
        
        #注意，这里没有 利用 outcome_MV_21_10_5_5.pkl
        
        pp.savefig(figure_temp)
        pp.close()

    if visionType == "industry":
        
        for industryIndex in range(len(industryFilteredList)):
            for factorIndex in range(len(factorFilteredList)):

                profit_series = outcome_dataframe["ret"][(outcome_dataframe[
                        "FactorPortfolio"] == factorFilteredList[factorIndex]) & (
                        outcome_dataframe["IndustryPortfolio"] == industryFilteredList[
                        industryIndex])]
                date_series = outcome_dataframe["trade_date"][(outcome_dataframe[
                        "FactorPortfolio"] == factorFilteredList[factorIndex]) & (
                        outcome_dataframe["IndustryPortfolio"] == industryFilteredList[
                        industryIndex])]

                profit_cumulate_series = np.exp(np.log(1 + profit_series).expanding(1).sum()) - 1  # 抽取累积收益Series

                dataframe_temp = pd.DataFrame({'date': date_series.apply(lambda x: str(x)).tolist(),
                                                   'profit': profit_cumulate_series.tolist()})  # 生成一个新的dataframe来储存/此处只是为了update index
                #Draw Pic
                plt.figure(2, figsize=(20, 180))
                pic_temp = plt.subplot(len(industryFilteredList), 1, industryIndex + 1)
                plt.plot(dataframe_temp['date'], dataframe_temp['profit'],
                         label="FactorPortfolio = %s" % str(factorFilteredList[factorIndex]))
                plt.legend()
                
            pic_temp.legend()
            xticks_increment = int(
                    dataNumber / xtickNumber / len(factorFilteredList) / len(industryFilteredList))  # 设置画图步长
            plt.xticks([xticks_increment * (x) for x in range(xtickNumber)], dataframe_temp['date'].loc[
                    [xticks_increment * (x) for x in range(xtickNumber)]])  # 设置x轴刻度

            plt.title("Industry = %s" % str(industryFilteredList[industryIndex]) )
            plt.xlabel('Trade Date')
            plt.ylabel('Returns')

        plt.tight_layout()
        figure_temp = plt.figure(2)
        
        pp = PdfPages(r'G:\MultiFactor\Factor\Momentum\graph_industryNeutral%s_%s.pdf' % (file_name, visionType))
        pp.savefig(figure_temp)
        pp.close()

IndustryNeutral_visualization(df = outcome_dataframe,
                               file_name = file_name )
'''
====================== next =============================
'''

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\outcome_Ind_21_10_5.pkl', 'rb')
outcome_dataframe = pkl.load(pkl_file)
file_name = pkl_file.name[-12:-4]
pkl_file.close()

def IndustryNeutral_portfolio(df = outcome_dataframe, file_name = file_name):

    
    df = df.reset_index('trade_date').copy()
    
    df_group = df.groupby(by = ['trade_date','FactorPortfolio'])[['ret']].sum() / 28
    # the constant 28, is to derive the mean
    # another way is to apply the weighted adopted by the 300 index over various industries
    
    df_group = df_group.reset_index()
    
    factorList = np.unique(df_group['FactorPortfolio'])
    
    plt2.figure(1, figsize = (30,10))
    for f in factorList :
        grouped = df_group[df_group['FactorPortfolio'] == f][['trade_date','ret']]
        grouped.index = range(len(grouped))
        grouped['ret'] = np.exp(grouped['ret'].apply(lambda x : np.log(1+x)).expanding(1).sum()) - 1
        plt2.plot(grouped['trade_date'], grouped['ret'], label = 'factorportfolio_%d' % f)
        plt2.legend()
        
    xticks_count = len(grouped) // 30
    xticks_increment = int(len(grouped) / xticks_count)
    plt2.xticks([xticks_increment * x for x in range(xticks_count)], 
                [grouped.iloc[x,0] for x in [xticks_increment * y for y in range(xticks_count)]])
    plt2.title('Industry Neutral Portfolio')
    plt2.tight_layout()

    pp = PdfPages(r'G:\MultiFactor\Factor\Momentum\graph_industryNeutral_Portfolio%s.pdf' % file_name)
    pp.savefig(plt2.figure(1))
    pp.close()
    
#IndustryNeutral_portfolio(df = outcome_dataframe, file_name = file_name)


'''
====================== next =============================
'''


pkl_file = open(r'G:\MultiFactor\Factor\Momentum\outcome_MV_21_10_5_5.pkl', 'rb')
outcome_dataframe = pkl.load(pkl_file)
file_name = pkl_file.name[-14:-4]
pkl_file.close()

def mvTable_visualization(df, file_name = file_name):
    
    outcome_dataframe = df.reset_index('trade_date')
    outcome_dataframe['trade_date'] = outcome_dataframe['trade_date'].apply(lambda x : int(x))
    
    factor_filtered_list = list(set(outcome_dataframe["FactorPortfolio"]))
    market_filtered_list = list(set(outcome_dataframe["MarketValuePortfolio"]))
    factor_count, market_count = len(factor_filtered_list), len(market_filtered_list)
    indicator = ['annual_return', 'annual_volatility', 'annual_sharpe', 'max_retrace', 'max_retrace_date', 'max_win_streak_number',
                 'max_lose_streak_number', 'margin_profit_to_lose',
                 'count_profit_to_lose', 'turnover_rate']   

    data_storage = pd.DataFrame(np.zeros((factor_count*market_count,len(indicator) + 2)),
                                index=[(x, y) for x in factor_filtered_list for y in market_filtered_list], 
                                columns = ['FactorPortfolio','MarketValuePortfolio'] + indicator)

    for factorIndex in range(factor_count):
        for marketIndex in range(market_count):
            profit_series = outcome_dataframe["ret"][(outcome_dataframe[
                    "FactorPortfolio"] == factor_filtered_list[factorIndex]) & (
                    outcome_dataframe["MarketValuePortfolio"] == market_filtered_list[
                    marketIndex])]

            symbol_series = outcome_dataframe[["symbols"]][(outcome_dataframe[
                    "FactorPortfolio"] == factor_filtered_list[factorIndex]) & (
                    outcome_dataframe["MarketValuePortfolio"] == market_filtered_list[
                    marketIndex])].loc[0,:]
            
            turnover_rate = []
            for row in range(1,len(symbol_series)):
                position_change = 1 - np.sum(pd.Series(symbol_series.iloc[row,0]).isin(pd.Series(symbol_series.iloc[row - 1,0]))) / len(symbol_series.iloc[row - 1,0])
                turnover_rate.append(position_change)
            
                
            return_temp = profit_series.apply(lambda x: (1+x)).product()
            data_storage['annual_return'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = \
                return_temp**(252/len(profit_series)) - 1
            data_storage['annual_volatility'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = \
                profit_series.std()*np.sqrt(252)
            data_storage['annual_sharpe'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = \
                data_storage['annual_return'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] / data_storage['annual_volatility'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])]
            
            cum_ret = np.exp(np.log(1 +  profit_series).cumsum())-1
            rolling_max = cum_ret.expanding(1).max()
            rolling_retrace = cum_ret - rolling_max
            rolling_retrace.index = range(len(rolling_retrace))
            max_retrace = - np.min(cum_ret - rolling_max)
            
            data_storage['max_retrace'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = max_retrace
            
            data_storage['max_retrace_date'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = \
                outcome_dataframe[(outcome_dataframe[
                        "FactorPortfolio"] == factor_filtered_list[factorIndex]) & (
                    outcome_dataframe["MarketValuePortfolio"] == market_filtered_list[
                    marketIndex])].iloc[rolling_retrace[
                    rolling_retrace == -max_retrace].index[0],0]

#            data_storage['max_win_streak_number'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = \
#                max([len(list(v)) for k, v in itertools.groupby(profit_series > 0)])
#             If you use groupby() on unorderd input you'll get a new group every time 
#             a different key is returned by the key function
#            
#            data_storage['max_lose_streak_number'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = \
#                max([len(list(v)) for k, v in itertools.groupby(profit_series < 0)])
#            上面的代码不能区分 win 或者 lose, itertools.groupby(>0) 与 itertools.groupby(<0) 无差
            
            win_countList = []
            win_count = 0
            for win in range(1, len(profit_series)):
                if (profit_series.iloc[win] * profit_series.iloc[win - 1] > 0) & (profit_series.iloc[win] > 0):
                    win_count += 1
                else:
                    win_count = 0
                win_countList.append(win_count)
            data_storage['max_win_streak_number'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = np.max(win_countList)
            
            lose_countList = []
            lose_count = 0
            for lose in range(1, len(profit_series)):
                if (profit_series.iloc[lose] * profit_series.iloc[lose - 1] > 0) & (profit_series.iloc[lose] < 0):
                    lose_count += 1
                else:
                    lose_count = 0
                lose_countList.append(lose_count)
            data_storage['max_lose_streak_number'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = np.max(lose_countList)
                        

            profit_count, lose_count = len(profit_series[profit_series > 0]), len(profit_series[profit_series < 0])
            profit_return, lose_return = np.mean(profit_series[profit_series > 0]), np.mean(profit_series[profit_series < 0])
            
            data_storage['margin_profit_to_lose'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] =\
                abs(profit_return / lose_return )
            data_storage['count_profit_to_lose'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = profit_count / lose_count
            data_storage['turnover_rate'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = np.mean(turnover_rate)
            
            data_storage['FactorPortfolio'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = factorIndex + 1
            data_storage['MarketValuePortfolio'][(factor_filtered_list[factorIndex], market_filtered_list[marketIndex])] = marketIndex + 1
    
    data_storage['FactorPortfolio'] = data_storage['FactorPortfolio'].apply(lambda x : str(int(x)))
    data_storage['MarketValuePortfolio'] = data_storage['MarketValuePortfolio'].apply(lambda x : str(int(x)))
    data_storage['annual_return'] = (data_storage['annual_return'] * 100).apply(lambda x : '%6.2f%%' %  x)
    data_storage['annual_volatility'] = (data_storage['annual_volatility'] * 100).apply(lambda x : '%6.2f%%' %  x)
    data_storage['annual_sharpe'] = data_storage['annual_sharpe'].apply(lambda x : '%6.2f' %  x)
    data_storage['max_retrace'] = (data_storage['max_retrace'] * 100).apply(lambda x : '%6.2f%%' %  x)
    data_storage['max_retrace_date'] = data_storage['max_retrace_date'].apply(lambda x : str(int(x)))
    data_storage['max_win_streak_number'] = data_storage['max_win_streak_number'].apply(lambda x : str(int(x)))
    data_storage['max_lose_streak_number'] = data_storage['max_lose_streak_number'].apply(lambda x : str(int(x)))
    data_storage['margin_profit_to_lose'] = data_storage['margin_profit_to_lose'].apply(lambda x : '%6.2f' %  x)
    data_storage['count_profit_to_lose'] = data_storage['count_profit_to_lose'].apply(lambda x : '%6.2f' %  x)
    data_storage['turnover_rate'] = (data_storage['turnover_rate']*100).apply(lambda x : '%6.2f%%' %  x)

#    哥华的版本
#    for tableIndex in range(len(data_storage.columns)):
#        series_temp = data_storage[data_storage.columns[tableIndex]].values.reshape(factor_count, market_count)
#        
#        plt.figure(3, figsize=(20, 40))
#        pic_temp = plt.subplot(len(data_storage.columns), 1, tableIndex+1)
#        pic_temp.axis('off')
#        pic_temp.axis('tight')
#        table_temp = pic_temp.table(cellText=series_temp, loc='center', cellLoc='center', rowLabels=factor_filtered_list,
#                                    colLabels = market_filtered_list, colLoc='center', colWidths=[0.1]*len(market_filtered_list))
#        plt.title(data_storage.columns[tableIndex], loc='left', size=25)
#        table_temp.set_fontsize(20)
#        table_temp.scale(2, 2)
#
#    plt.tight_layout()
#    figure_temp = plt.figure(3)
    
#    我的版本
#    plt.figure(3,figsize=(25,16))
#    plt.axis('off')
#    table_temp = plt.table(cellText = data_storage.values, loc = 'upper center', cellLoc = 'center',
#              colLabels = data_storage.columns, colLoc = 'center')
#    plt.title('Table for MarketValueNeutral%s' % file_name,fontsize = 40)
#    table_temp.set_fontsize(30)
#    table_temp.scale(0.8,2)
#    plt.tight_layout()
#    figure_temp = plt.figure(3)
#
#    pp = PdfPages(r'G:\MultiFactor\Factor\Momentum\table_mvNeutral%s.pdf' % file_name)
#    pp.savefig(figure_temp)
#    pp.close()

    data_storage.to_excel(r'G:\MultiFactor\Factor\Momentum\table_mvNeutral%s.xlsx' % file_name)
    
#mvTable_visualization(df = outcome_dataframe)


'''
====================== next =============================
'''



pkl_file = open(r'G:\MultiFactor\Factor\Momentum\outcome_Ind_21_10_10.pkl', 'rb')
outcome_dataframe = pkl.load(pkl_file)
file_name = pkl_file.name[-12:-4]
pkl_file.close()


def industryTable_visualization(df, file_name = file_name):
    
    outcome_dataframe = df.reset_index('trade_date')
    outcome_dataframe['trade_date'] = outcome_dataframe['trade_date'].apply(lambda x : int(x))
    
    factor_filtered_list = list(set(outcome_dataframe["FactorPortfolio"]))
    industry_filtered_list = list(set(outcome_dataframe["IndustryPortfolio"]))
    factor_count, industry_count = len(factor_filtered_list), len(industry_filtered_list)
    indicator = ['annual_return', 'annual_volatility', 'annual_sharpe', 'max_retrace', 'max_retrace_date', 'max_win_streak_number',
                 'max_lose_streak_number', 'margin_profit_to_lose',
                 'count_profit_to_lose', 'turnover_rate']   

    data_storage = pd.DataFrame(np.zeros((factor_count*industry_count,len(indicator) + 2)),
                                index=[(x, y) for x in industry_filtered_list for y in factor_filtered_list], 
                                columns = ['IndustryPortfolio','FactorPortfolio'] + indicator)

    for industryIndex in range(industry_count):
        for factorIndex in range(factor_count):
            profit_series = outcome_dataframe["ret"][(outcome_dataframe[
                    "FactorPortfolio"] == factor_filtered_list[factorIndex]) & (
                    outcome_dataframe["IndustryPortfolio"] == industry_filtered_list[
                    industryIndex])]

            symbol_series = outcome_dataframe[["symbols"]][(outcome_dataframe[
                    "FactorPortfolio"] == factor_filtered_list[factorIndex]) & (
                    outcome_dataframe["IndustryPortfolio"] == industry_filtered_list[
                    industryIndex])].loc[0,:]
            
            turnover_rate = []
            for row in range(1,len(symbol_series)):
                position_change = 1 - np.sum(pd.Series(symbol_series.iloc[row,0]).isin(pd.Series(symbol_series.iloc[row - 1,0]))) / len(symbol_series.iloc[row - 1,0])
                turnover_rate.append(position_change)
            
                
            return_temp = profit_series.apply(lambda x: (1+x)).product()
            data_storage['annual_return'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] = \
                return_temp**(252/len(profit_series)) - 1
            data_storage['annual_volatility'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] = \
                profit_series.std()*np.sqrt(252)
            data_storage['annual_sharpe'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] = \
                data_storage['annual_return'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] / data_storage['annual_volatility'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])]
            
            cum_ret = np.exp(np.log(1 +  profit_series).cumsum())-1
            rolling_max = cum_ret.expanding(1).max()
            rolling_retrace = cum_ret - rolling_max
            rolling_retrace.index = range(len(rolling_retrace))
            max_retrace = - np.min(cum_ret - rolling_max)
            
            data_storage['max_retrace'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] = max_retrace
            
            data_storage['max_retrace_date'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] = \
                outcome_dataframe[(outcome_dataframe[
                        "FactorPortfolio"] == factor_filtered_list[factorIndex]) & (
                    outcome_dataframe["IndustryPortfolio"] == industry_filtered_list[
                    industryIndex])].iloc[rolling_retrace[
                    rolling_retrace == -max_retrace].index[0],0]

#            data_storage['max_win_streak_number'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] = \
#                max([len(list(v)) for k, v in itertools.groupby(profit_series > 0)])
#             If you use groupby() on unorderd input you'll get a new group every time 
#             a different key is returned by the key function
#            
#            data_storage['max_lose_streak_number'][(factor_filtered_list[factorIndex], industry_filtered_list[industryIndex])] = \
#                max([len(list(v)) for k, v in itertools.groupby(profit_series < 0)])
#            上面的代码不能区分 win 或者 lose, itertools.groupby(>0) 与 itertools.groupby(<0) 无差
            
            win_countList = []
            win_count = 0
            for win in range(1, len(profit_series)):
                if (profit_series.iloc[win] * profit_series.iloc[win - 1] > 0) & (profit_series.iloc[win] > 0):
                    win_count += 1
                else:
                    win_count = 0
                win_countList.append(win_count)
            data_storage['max_win_streak_number'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] = np.max(win_countList)
            
            lose_countList = []
            lose_count = 0
            for lose in range(1, len(profit_series)):
                if (profit_series.iloc[lose] * profit_series.iloc[lose - 1] > 0) & (profit_series.iloc[lose] < 0):
                    lose_count += 1
                else:
                    lose_count = 0
                lose_countList.append(lose_count)
            data_storage['max_lose_streak_number'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] = np.max(lose_countList)
                        

            profit_count, lose_count = len(profit_series[profit_series > 0]), len(profit_series[profit_series < 0])
            profit_return, lose_return = np.mean(profit_series[profit_series > 0]), np.mean(profit_series[profit_series < 0])
            
            data_storage['margin_profit_to_lose'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] =\
                abs(profit_return / lose_return )
            data_storage['count_profit_to_lose'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] = profit_count / lose_count
            data_storage['turnover_rate'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] = np.mean(turnover_rate)
            
            data_storage['FactorPortfolio'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] = factorIndex + 1
            data_storage['IndustryPortfolio'][(industry_filtered_list[industryIndex], factor_filtered_list[factorIndex])] = industryIndex + 1
    
    data_storage['FactorPortfolio'] = data_storage['FactorPortfolio'].apply(lambda x : str(int(x)))
    data_storage['IndustryPortfolio'] = data_storage['IndustryPortfolio'].apply(lambda x : str(int(x)))
    data_storage['annual_return'] = (data_storage['annual_return'] * 100).apply(lambda x : '%6.2f%%' %  x)
    data_storage['annual_volatility'] = (data_storage['annual_volatility'] * 100).apply(lambda x : '%6.2f%%' %  x)
    data_storage['annual_sharpe'] = data_storage['annual_sharpe'].apply(lambda x : '%6.2f' %  x)
    data_storage['max_retrace'] = (data_storage['max_retrace'] * 100).apply(lambda x : '%6.2f%%' %  x)
    data_storage['max_retrace_date'] = data_storage['max_retrace_date'].apply(lambda x : str(int(x)))
    data_storage['max_win_streak_number'] = data_storage['max_win_streak_number'].apply(lambda x : str(int(x)))
    data_storage['max_lose_streak_number'] = data_storage['max_lose_streak_number'].apply(lambda x : str(int(x)))
    data_storage['margin_profit_to_lose'] = data_storage['margin_profit_to_lose'].apply(lambda x : '%6.2f' %  x)
    data_storage['count_profit_to_lose'] = data_storage['count_profit_to_lose'].apply(lambda x : '%6.2f' %  x)
    data_storage['turnover_rate'] = (data_storage['turnover_rate']*100).apply(lambda x : '%6.2f%%' %  x)


    data_storage.to_excel(r'G:\MultiFactor\Factor\Momentum\table_industryNeutral%s.xlsx' % file_name)

#industryTable_visualization(df = outcome_dataframe, file_name = file_name)






'''
====================== next =============================
'''


df = pd.read_excel(r'G:\MultiFactor\Factor\Momentum\table_mvNeutral_21_10_10_5.xlsx')
file_name = 'heatmap_mvNeutral_21_10_10_5'

def heatmap(df, file_name):
    
    # series: a DataFrame with one column and an index of str(trade_date)
    
    column = df.columns
    
    if 'FactorPortfolio' in column:
        series = df[['FactorPortfolio','MarketValuePortfolio','annual_return','annual_sharpe']]
    elif 'IndustryPortfolio' in column:
        series = df[['IndustryPortfolio','FactorPortfolio','annual_return','annual_sharpe']]
    else:
        print('check columns!')
    
    series.index = range(len(series))
    
    vertical_axis = []
    horizontal_axis = []
    
    for i in range(len(series)):
        vertical_axis.append(series.iloc[i,0])
        horizontal_axis.append(series.iloc[i,1])
    
    series.index = pd.MultiIndex.from_arrays(
            [vertical_axis, horizontal_axis],
            names = series.columns[:2])
    
    series_1 = series[['annual_return']]
    series_1['annual_return'] = series_1['annual_return'].apply(lambda x : float(x[:-1]) / 100)

    series_2 = series[['annual_sharpe']]
    series_1 = series_1['annual_return'].unstack()
    series_2 = series_2['annual_sharpe'].unstack()

    f, (ax1, ax2) = plt2.subplots(1, 2, figsize=(2 * len(series_1) + 4, 1 * len(series_1)))
#    ax = ax.flatten()
    
    ax_1 = sns.heatmap(
            series_1,
            annot = True,
            alpha = 1.0,
            center = 0.0,
            linewidths = 0.01,
            linecolor = 'white',
            cmap = cm.get_cmap('RdBu'),
            cbar = True,
            ax = ax1)
    ax_1.set_title('annual return')
#    fig_1 = ax_1.get_figure()
#    fig_1.savefig(r'G:\MultiFactor\Factor\Momentum\return_%s.pdf' % file_name)
    
    ax_2 = sns.heatmap(
            series_2,
            annot = True,
            alpha = 1.0,
            center = 0.0,
            linewidths = 0.01,
            linecolor = 'white',
            cmap = cm.get_cmap('RdBu'),
            cbar = True,
            ax = ax2)
    ax_2.set_title('annual sharpe')
#    fig_2 = ax_2.get_figure()
#    fig_2.savefig(r'G:\MultiFactor\Factor\Momentum\sharpe_%s.pdf' % file_name)
    
    f.savefig(r'G:\MultiFactor\Factor\Momentum\%s.pdf' % file_name)

#heatmap(df, file_name)
