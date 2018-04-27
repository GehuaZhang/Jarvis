# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:42:48 2018

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

def IndustryMatrix(exposureMatrix,trailingDays = 63):
    # 参照了 MarketValueNeutral.py 的写法，但是不需要传入 trailingDays 作为参数
    
    trading_date = read_tradingDate() 

    exposureMatrix = exposureMatrix.reset_index('trade_date')
    exposureMatrix['trade_date'] = exposureMatrix['trade_date'].apply(lambda x: int(x))
    
    first_date = exposureMatrix.iloc[0,0]
    start_date = trading_date.iloc[trading_date.index[trading_date['trade_date']==first_date][0] - trailingDays,0]
    end_date = exposureMatrix.iloc[-1,0]    
    
    # first initialize the MarketValueMatrix, with similar format with exposureMatrix
    colList = np.array(exposureMatrix.columns.tolist()[1:])
    
    #initiate market value exposure matrix, similar to factor exposure matrix
    dates = exposureMatrix[['trade_date']]
    industryMatrix = dates.copy()
    
    myFolder =r'G:\MultiFactor\Data\TimeSeriesDaily'
    fileList = glob(os.path.join(myFolder,'*.h5'))
    fileList.sort()
    fileName = fileList[-1][-20:-3]  
    hdf = pd.HDFStore(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.h5' % fileName,'r')

    for key in colList:
        ind = getattr(hdf,key)[['trade_date','industry_1']]
        ind = ind[(ind['trade_date'] >= start_date) & (ind['trade_date'] <= end_date)]
        if len(ind) == 0:
            continue
        ind = ind[['trade_date','industry_1']].rename(columns={'industry_1':'%s' % key})
        industryMatrix = pd.merge(industryMatrix, ind, how='left', on='trade_date')
        print(key)
    
    industryMatrix['trade_date'] = industryMatrix['trade_date'].apply(lambda x: str(x))
    industryMatrix = industryMatrix.set_index('trade_date')
    
    pkl_file = open(r'G:\MultiFactor\Factor\Momentum\industryMatrix_%d.pkl' % trailingDays,'wb')
    pickle.dump(industryMatrix,pkl_file)
    pkl_file.close()
    
    hdf.close()
    
trailingDays = 21
holdingDays = 10
halfLife = 10

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\exposureMatrix_%d_%d_%d.pkl' % (trailingDays,holdingDays,halfLife),'rb')
exposureMatrix = pickle.load(pkl_file)
pkl_file.close()

#IndustryMatrix(exposureMatrix,trailingDays = 63)

# initialize marketValueMatrix and statusList

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\industryMatrix_%d.pkl' % trailingDays,'rb')
industryMatrix = pickle.load(pkl_file)
pkl_file.close()  

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\statusList_%d.pkl' % trailingDays,'rb')
statusList = pickle.load(pkl_file)
pkl_file.close()

#pkl_file = open(r'G:\MultiFactor\Factor\Momentum\dailyReturnMatrix_%d_%d_%d.pkl' % (trailingDays,holdingDays,halfLife),'rb')
#dailyReturnMatrix = pickle.load(pkl_file)
#pkl_file.close()

def industryNeutral(exposureMatrix = exposureMatrix, 
                    industryMatrix = industryMatrix ,
                    statusList = statusList,
                    trailingDays = 63,
                    holdingDays = 10 , n = 5):
    
    # n 是因子的分组
    # m 本来是行业的分组，但是现在 m 可以直接从 industryMatrix 读出来

    trading_date = read_tradingDate()  
    
    exposureMatrix = exposureMatrix.reset_index('trade_date')
    exposureMatrix['trade_date'] = exposureMatrix['trade_date'].apply(lambda x: int(x))

    industryMatrix = industryMatrix.reset_index('trade_date')
    industryMatrix['trade_date'] = industryMatrix['trade_date'].apply(lambda x: int(x))    
    
    myFolder =r'G:\MultiFactor\Data\TimeSeriesDaily'
    fileList = glob(os.path.join(myFolder,'*.h5'))
    fileList.sort()
    fileName = fileList[-1][-20:-3]
    hdf_price = pd.HDFStore(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.h5' % fileName,'r')
    
    
#    m = len(np.unique(industryMatrix.dropna(axis=1).iloc[0,1:].values.tolist()))
    industryList = np.unique(industryMatrix.dropna(axis=1).iloc[0,1:].values.tolist())
    
    colList = np.array(exposureMatrix.columns.tolist()[1:])
    outcome = pd.DataFrame([])
    
    for date_index in range(0,len(exposureMatrix)):
        
        sample_exposure = pd.DataFrame([colList,exposureMatrix.iloc[date_index,:].values[1:]]).T
        sample_exposure.columns = ['symbol','exposure']
        sample_exposure = sample_exposure.dropna(axis = 0)        
        
        sample_industry = pd.DataFrame([colList, industryMatrix.iloc[date_index,:].values[1:]]).T
        sample_industry.columns = ['symbol','industry_1']
        sample_industry = sample_industry.dropna(axis = 0)
        
        for ind in industryList:
            sample_ind = sample_industry[sample_industry['industry_1'] == ind]
            if len(sample_ind) == 0:
                continue
            
            locals()['portIndustry_'+str(ind)] = pd.merge(sample_ind, sample_exposure,
                  how='left', on='symbol')
            locals()['portIndustry_'+str(ind)] = locals()['portIndustry_'+str(ind)].dropna(axis = 0)
            locals()['portIndustry_'+str(ind)]['rank'] = locals()['portIndustry_'+str(ind)]['exposure'].rank(method='dense')
            
            for i in range(1,n+1):
                locals()['port_'+str(ind) + '_' +str(i)] = locals()['portIndustry_'+str(ind)][(locals()['portIndustry_'+str(ind)]['rank'] <= np.ceil(len(locals()['portIndustry_'+str(ind)]['rank'].unique()) * i / n )) & (locals()['portIndustry_'+str(ind)]['rank'] >= np.floor(len(locals()['portIndustry_'+str(ind)]['rank'].unique()) * (i-1) / n ))]
                if len(locals()['port_'+str(ind) + '_' +str(i)]) == 0:
                    continue
                locals()['port_'+str(ind) + '_' +str(i)] = locals()['port_'+str(ind) + '_' +str(i)].sort_values(by='rank')
                locals()['port_'+str(ind) + '_' +str(i)]['weight'] = 1
                
                if int(len(locals()['portIndustry_'+str(ind)]) * i / n) != len(locals()['portIndustry_'+str(ind)]) * i / n:
                    locals()['port_'+str(ind) + '_' +str(i)].iloc[-2,-1] = len(locals()['portIndustry_'+str(ind)]) * i / n - np.floor(len(locals()['portIndustry_'+str(ind)]) * i / n )
                    
                    locals()['port_'+str(ind) + '_' +str(i)].iloc[-1,-1] = np.ceil(len(locals()['portIndustry_'+str(ind)]) * i / n ) - len(locals()['portIndustry_'+str(ind)]) * i / n
                
                if int(len(locals()['portIndustry_'+str(ind)]) * (i-1) / n) != len(locals()['portIndustry_'+str(ind)]) * (i-1) / n:
                    locals()['port_'+str(ind) + '_' +str(i)].iloc[0,-1] = len(locals()['portIndustry_'+str(ind)]) * (i-1) / n - np.floor(len(locals()['portIndustry_'+str(ind)]) * (i-1) / n )
                    locals()['port_'+str(ind) + '_' +str(i)].iloc[1,-1] = np.ceil(len(locals()['portIndustry_'+str(ind)]) * (i-1) / n ) - len(locals()['portIndustry_'+str(ind)]) * (i-1) / n
                if (i > 1) & (int(len(locals()['portIndustry_'+str(ind)]) * (i-1) / n) == len(locals()['portIndustry_'+str(ind)]) * (i-1) / n):
                    locals()['port_'+str(ind) + '_' +str(i)].iloc[0,-1] = 0
                
                locals()['port_'+str(ind) + '_' +str(i)] = locals()['port_'+str(ind) + '_' +str(i)].dropna(axis = 0)
                locals()['port_'+str(ind) + '_' +str(i)]['weight'] = locals()['port_'+str(ind) + '_' +str(i)]['weight'] / np.sum(locals()['port_'+str(ind) + '_' +str(i)]['weight'])
                
                purchasable = statusList[(statusList['trade_date'] == exposureMatrix.iloc[date_index,0]) & (statusList['trade_status'] == 1)]
                locals()['port_'+str(ind) + '_' +str(i)] = locals()['port_'+str(ind) + '_' +str(i)].loc[locals()['port_'+str(ind) + '_' +str(i)]['symbol'].isin(purchasable['symbol'])]
                locals()['port_'+str(ind) + '_' +str(i)] = locals()['port_'+str(ind) + '_' +str(i)].dropna(axis = 0)
                locals()['port_'+str(ind) + '_' +str(i)]['weight'] = locals()['port_'+str(ind) + '_' +str(i)]['weight'] / np.sum(locals()['port_'+str(ind) + '_' +str(i)]['weight'])
                
                if len(locals()['port_'+str(ind) + '_' +str(i)]) == 0:
                    continue
                
                sample_start_date = exposureMatrix.iloc[date_index,0]
                sample_end_date = trading_date.iloc[trading_date.index[trading_date['trade_date']==exposureMatrix.iloc[date_index,0]][0] + holdingDays - 1,0]
                
                # port_panel describes T * N matrix, to multiply N * 1 weight vector
                port_panel = pd.DataFrame([])
                for code in locals()['port_'+str(ind) + '_' +str(i)]['symbol']:
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
                sample_outcome['IndustryPortfolio'] = ind
                sample_outcome['FactorPortfolio'] = i
                sample_outcome['count'] = len(locals()['port_'+str(ind) + '_' +str(i)]['symbol'])
                sample_outcome['symbols'] =  1
                positionStock = list(locals()['port_'+str(ind) + '_' +str(i)]['symbol'])
                sample_outcome['symbols'] = sample_outcome['symbols'].agg(lambda x :  positionStock)
                port_panel = port_panel.set_index('trade_date')
                sample_outcome['ret'] = np.dot(port_panel,locals()['port_'+str(ind) + '_' +str(i)]['weight'])
                
                print(date_index,'\t',ind,'\t',i)
                
                outcome = pd.concat([outcome,sample_outcome],axis = 0)    
    
    hdf_price.close()
    
    outcome['trade_date'] = outcome['trade_date'].apply(lambda x : str(x))
    outcome = outcome.set_index('trade_date')    
    
    pkl_file = open(r'G:\MultiFactor\Factor\Momentum\outcome_Ind_%d_%d_%d.pkl' % (trailingDays,holdingDays, n),'wb')
    pickle.dump(outcome,pkl_file)
    pkl_file.close()
    
    return outcome

#outcome = industryNeutral()
#outcome = industryNeutral(exposureMatrix = exposureMatrix, 
#                    industryMatrix = industryMatrix ,
#                    statusList = statusList,
#                    trailingDays = 21,
#                    holdingDays = 10 , n = 10)