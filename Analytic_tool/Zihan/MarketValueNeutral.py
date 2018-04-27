# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:32:02 2018

@author: Alan
"""

import pandas as pd
import numpy as np
import pickle
import os
from glob import glob



def read_tradingDate():
    pkl_file = open(r'G:\MultiFactor\datetimeTools\trade_date.pkl','rb')
    trading_date = pickle.load(pkl_file)
    trading_date = trading_date.rename(columns={'date':'trade_date'})
    pkl_file.close()
    return trading_date

def MarketValueMatrix(exposureMatrix,trailingDays = 21):
    
    trading_date = read_tradingDate() 
    first_date = exposureMatrix.iloc[0,0]
    start_date = trading_date.iloc[trading_date.index[trading_date['trade_date']==first_date][0] - trailingDays,0]
    end_date = exposureMatrix.iloc[-1,0]    
    
    # first initialize the MarketValueMatrix, with similar format with exposureMatrix
    colList = np.array(exposureMatrix.columns.tolist()[1:])
    
    #initiate market value exposure matrix, similar to factor exposure matrix
    dates = exposureMatrix[['trade_date']]
    marketValueMatrix = dates.copy()
    
    myFolder =r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv'
    fileList = glob(os.path.join(myFolder,'*.h5'))
    fileList.sort()
    fileName = fileList[-1][-23:-3]  
    hdf_mv = pd.HDFStore(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\%s.h5' % fileName,'r')

    for key in colList:
        mv = getattr(hdf_mv,key)[['trade_date','float_mv']]
        mv = mv[(mv['trade_date'] >= start_date) & (mv['trade_date'] <= end_date)]
        if len(mv) == 0:
            continue
        mv['mv_avg'] = pd.rolling_mean(mv['float_mv'],window = trailingDays).shift(1)
        # pd.rolling_mean, window = 21, then the first non-nan value is the 21st
        # as default, this func calculates including current one
        # so, shift(1) is applied
        
        mv = mv[['trade_date','mv_avg']].rename(columns={'mv_avg':'%s' % key})
        marketValueMatrix = pd.merge(marketValueMatrix,mv, how='left', on='trade_date')
    
    ['trade_date'] = marketValueMatrix['trade_date'].apply(lambda x: str(x))
    marketValueMatrix = marketValueMatrix.set_index('trade_date')
    
    pkl_file = open(r'G:\MultiFactor\Factor\Momentum\marketValueMatrix_%d.pkl' % trailingDays,'wb')
    pickle.dump(marketValueMatrix,pkl_file)
    pkl_file.close()
    
    hdf_mv.close()

def statusPanel(exposureMatrix,trailingDays = 21):
    
    trading_date = read_tradingDate()  
    
    # first of all, find the start_date from exposureMatrix
    first_date = exposureMatrix.iloc[0,0]
    start_date = trading_date.iloc[trading_date.index[trading_date['trade_date']==first_date][0] - trailingDays,0]
    end_date = exposureMatrix.iloc[-1,0]
    
    myFolder =r'G:\MultiFactor\Data\TimeSeriesDaily'
    fileList = glob(os.path.join(myFolder,'*.h5'))
    fileList.sort()
    fileName = fileList[-1][-20:-3]
    hdf_price = pd.HDFStore(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.h5' % fileName,'r')
    
    colList = np.array(exposureMatrix.columns.tolist()[1:])
    
    statusList = pd.DataFrame([])
    for key in colList:
        status = getattr(hdf_price,key)[['trade_date','symbol','trade_status']]
        status = status[(status['trade_date'] >= start_date) & (status['trade_date'] <= end_date)]
        statusList = pd.concat([statusList,status],axis = 0)
    
    # statusList  是以面板形式存储的数据, 
    statusList['trade_date'] = statusList['trade_date'].apply(lambda x: str(x))
    
    pkl_file = open(r'G:\MultiFactor\Factor\Momentum\statusList_%d.pkl' % trailingDays,'wb')
    pickle.dump(statusList,pkl_file)
    pkl_file.close()
    
    hdf_price.close()

# load exposureMatrix
trailingDays = 21
holdingDays = 10
halfLife=10
pkl_file = open(r'G:\MultiFactor\Factor\Momentum\exposureMatrix_%d_%d_%d.pkl' % (trailingDays,holdingDays,halfLife),'rb')
exposureMatrix = pickle.load(pkl_file)
pkl_file.close()

# initialize marketValueMatrix and statusList
MarketValueMatrix(exposureMatrix,trailingDays = 21)
statusPanel(exposureMatrix,trailingDays = 21)

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\marketValueMatrix_%d.pkl' % trailingDays,'rb')
marketValueMatrix = pickle.load(pkl_file)
pkl_file.close()  

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\statusList_%d.pkl' % trailingDays,'rb')
statusList = pickle.load(pkl_file)
pkl_file.close()

def mvNeutral(exposureMatrix, marketValueMatrix, statusList, 
              trailingDays = 21, holdingDays = 10, n = 5, m =5):
    # n 是因子的分组
    # m 是市值大小的分组
    # exposureMatrix, marketValueMatrix, statusList 都是分开展开得到的，减小程序压力
    
    trading_date = read_tradingDate()
    marketValueMatrix = marketValueMatrix.reset_index('trade_date')
    marketValueMatrix['trade_date'] = marketValueMatrix['trade_date'].apply(lambda x: int(x))
    
    statusList['trade_date'] = statusList['trade_date'].apply(lambda x: int(x))
    
    #这个程序最开始是把 trade_date 当做 列 来写的;
    # 还有 trade_date 是整型；
    # 然而现在 input 3个 df 都是 把 trade_date 当做 str 来存储
    
    # first of all, find the start_date from exposureMatrix
    exposureMatrix = exposureMatrix.reset_index('trade_date')
    exposureMatrix['trade_date'] = exposureMatrix['trade_date'].apply(lambda x: int(x))
    first_date = exposureMatrix.iloc[0,0]
    start_date = trading_date.iloc[trading_date.index[trading_date['trade_date']==first_date][0] - trailingDays,0]
    end_date = exposureMatrix.iloc[-1,0]
    
    myFolder =r'G:\MultiFactor\Data\TimeSeriesDaily'
    fileList = glob(os.path.join(myFolder,'*.h5'))
    fileList.sort()
    fileName = fileList[-1][-20:-3]
    hdf_price = pd.HDFStore(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.h5' % fileName,'r')
    
    colList = np.array(exposureMatrix.columns.tolist()[1:])
    
    outcome = pd.DataFrame([])
    for date_index in range(0,len(exposureMatrix)):
        # util len() due to the fact that we build positions each date_index
        # and at the same time hold for a while until next date_index
        # so last date_index can be considered util date_index.last + holdingDays
        
        # transpose every index, including columns of codes 
        sample_exposure = pd.DataFrame([colList,exposureMatrix.iloc[date_index,:].values[1:]]).T
        sample_exposure.columns = ['symbol','exposure']
        
        #sample_expusre has na values since exposureMatrix has na values
        sample_exposure = sample_exposure.dropna(axis = 0)
        
        sample_exposure['rank'] = sample_exposure['exposure'].rank(method='dense')
        
        sample_mv = pd.DataFrame([colList,marketValueMatrix.iloc[date_index,:].values[1:]]).T
        sample_mv.columns = ['symbol','float_mv']
        sample_mv = sample_mv.dropna(axis = 0)
        sample_mv['rank'] = sample_mv['float_mv'].rank(method='dense')
        
        for i in range(1,n+1):
            locals()['portFactor_'+str(i)] = sample_exposure[(sample_exposure['rank'] <= np.ceil(len(sample_exposure['rank'].unique()) * i / n )) & (sample_exposure['rank'] >= np.floor(len(sample_exposure['rank'].unique()) * (i-1) / n ))]
            locals()['portFactor_'+str(i)] = locals()['portFactor_'+str(i)].sort_values(by='rank')
            locals()['portFactor_'+str(i)]['weight_factor'] = 1
            if len(locals()['portFactor_'+str(i)]) == 0:
                continue
            
            if int(len(sample_exposure) * i / n) != len(sample_exposure) * i / n:
                locals()['portFactor_'+str(i)].iloc[-2,-1] = len(sample_exposure) * i / n - np.floor(len(sample_exposure) * i / n )
                locals()['portFactor_'+str(i)].iloc[-1,-1] = np.ceil(len(sample_exposure) * i / n ) - len(sample_exposure) * i / n
            if int(len(sample_exposure) * (i-1) / n) != len(sample_exposure) * (i-1) / n:
                locals()['portFactor_'+str(i)].iloc[0,-1] = len(sample_exposure) * (i-1) / n - np.floor(len(sample_exposure) * (i-1) / n )
                locals()['portFactor_'+str(i)].iloc[1,-1] = np.ceil(len(sample_exposure) * (i-1) / n ) - len(sample_exposure) * (i-1) / n
            if (i > 1) & (int(len(sample_exposure) * (i-1) / n) == len(sample_exposure) * (i-1) / n):
                locals()['portFactor_'+str(i)].iloc[0,-1] = 0
            locals()['portFactor_'+str(i)]['weight_factor'] = locals()['portFactor_'+str(i)]['weight_factor'] / np.sum(locals()['portFactor_'+str(i)]['weight_factor'])

            for j in range(1,m+1):
                locals()['portMv_'+str(j)] = sample_mv[(sample_mv['rank'] <= np.ceil(len(sample_mv['rank'].unique()) * j / m )) & (sample_mv['rank'] >= np.floor(len(sample_mv['rank'].unique()) * (j-1) / m ))]
                locals()['portMv_'+str(j)] = locals()['portMv_'+str(j)].sort_values(by='rank')
                locals()['portMv_'+str(j)]['weight_mv'] = 1
                if len(locals()['portMv_'+str(j)]) == 0:
                    continue
            
                if int(len(sample_mv) * j / m) != len(sample_mv) * j / m:
                    locals()['portMv_'+str(j)].iloc[0,-1] = len(sample_mv) * (j-1) / m - np.floor(len(sample_mv) * (j-1) / m )
                    locals()['portMv_'+str(j)].iloc[1,-1] = np.ceil(len(sample_mv) * (j-1) / m ) - len(sample_mv) * (j-1) / m            
                if int(len(sample_mv) * (j-1) / m) != len(sample_mv) * (j-1) / m:
                    locals()['portMv_'+str(j)].iloc[-2,-1] = len(sample_mv) * j / m - np.floor(len(sample_mv) * j / m )
                    locals()['portMv_'+str(j)].iloc[-1,-1] = np.ceil(len(sample_mv) * j / m ) - len(sample_mv) * j / m
                if (j > 1) & (int(len(sample_mv) * (j-1) / m) == len(sample_mv) * (j-1) / m):
                    locals()['portMv_'+str(j)].iloc[0,-1] = 0
                
                locals()['port_'+str(i)+'_'+str(j)] = pd.merge(locals()['portFactor_'+str(i)][['symbol','weight_factor']],
                     locals()['portMv_'+str(j)][['symbol','weight_mv']],how='inner',on='symbol')
                locals()['port_'+str(i)+'_'+str(j)]['weight'] = locals()['port_'+str(i)+'_'+str(j)]['weight_factor'] * locals()['port_'+str(i)+'_'+str(j)]['weight_mv']
                locals()['port_'+str(i)+'_'+str(j)] = locals()['port_'+str(i)+'_'+str(j)][['symbol','weight']]
                
                #then to buy these stocks, being purchasable should be considered
                # do not forget to standardalize weight in the end
                purchasable = statusList[(statusList['trade_date'] == exposureMatrix.iloc[date_index,0]) & (statusList['trade_status'] == 1)]
                locals()['port_'+str(i)+'_'+str(j)] = locals()['port_'+str(i)+'_'+str(j)].loc[locals()['port_'+str(i)+'_'+str(j)]['symbol'].isin(purchasable['symbol'])]
                locals()['port_'+str(i)+'_'+str(j)]['weight'] = locals()['port_'+str(i)+'_'+str(j)]['weight'] / np.sum(locals()['port_'+str(i)+'_'+str(j)]['weight'])
                
                if len(locals()['port_'+str(i)+'_'+str(j)]) == 0:
                    continue
                
                sample_start_date = exposureMatrix.iloc[date_index,0]
                sample_end_date = trading_date.iloc[trading_date.index[trading_date['trade_date']==exposureMatrix.iloc[date_index,0]][0] + holdingDays - 1,0]
                
                # port_panel describes T * N matrix, to multiply N * 1 weight vector
                port_panel = pd.DataFrame([])
                for code in locals()['port_'+str(i)+'_'+str(j)]['symbol']:
                    sample_price = getattr(hdf_price,code)[['trade_date','symbol','trade_status','close','adjust_factor']]
                    sample_price['adjust_factor'] = sample_price['adjust_factor'] / sample_price.iloc[-1,-1]
                    sample_price['close_pre'] = sample_price['close'] * sample_price['adjust_factor']
                    sample_price['ret'] = sample_price['close_pre'] / sample_price['close_pre'].shift(1) - 1
                    sample_price = sample_price[(sample_price['trade_date'] >= sample_start_date) & (sample_price['trade_date'] <= sample_end_date)]
                    sample_price = sample_price[['trade_date','ret']]
                    sample_price = sample_price.dropna(axis = 0)                    
                    if len(port_panel) > 0:
                        port_panel = pd.merge(port_panel,sample_price,how='left',
                                              on='trade_date')
                    else:
                        port_panel = pd.concat([port_panel,sample_price],axis=1)
                # .fillna() is important, cuz np.dot(na,matrix) achieves np.nan
                port_panel = port_panel.fillna(0)
                sample_outcome = port_panel[['trade_date']]
                sample_outcome['FactorPortfolio'] = i
                sample_outcome['MarketValuePortfolio'] = j
                sample_outcome['count'] = len(locals()['port_'+str(i)+'_'+str(j)]['symbol'])
                sample_outcome['symbols'] =  1
                positionStock = list(locals()['port_'+str(i)+'_'+str(j)]['symbol'])
                sample_outcome['symbols'] = sample_outcome['symbols'].agg(lambda x :  positionStock)
                port_panel = port_panel.set_index('trade_date')
                sample_outcome['ret'] = np.dot(port_panel,locals()['port_'+str(i)+'_'+str(j)]['weight'])
                
                print(date_index,'\t',i,'\t',j)
                
                outcome = pd.concat([outcome,sample_outcome],axis = 0)
    
    hdf_price.close()
    
    outcome['trade_date'] = outcome['trade_date'].apply(lambda x : str(x))
    outcome = outcome.set_index('trade_date')
    
    pkl_file = open(r'G:\MultiFactor\Factor\Momentum\outcome_MV_%d_%d_%d_%d.pkl' % (trailingDays, holdingDays, n, m),'wb')
    pickle.dump(outcome,pkl_file)
    pkl_file.close()
    
    return outcome

#outcome = mvNeutral(exposureMatrix, marketValueMatrix, statusList, 
#                    trailingDays = 21, holdingDays = 10, n = 10, m =5)