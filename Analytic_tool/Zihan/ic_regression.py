# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:17:47 2018

@author: Alan
"""

import pandas as pd
import pickle as pkl
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

trailingDays = 21
holdingDays = 10
halfLife = 10
n = 10
m = 5

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\exposureMatrix_%d_%d_%d.pkl' % (trailingDays,holdingDays,halfLife),'rb')
exposureMatrix = pkl.load(pkl_file)
pkl_file.close()

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\dailyReturnMatrix_%d_%d_%d.pkl' % (trailingDays,holdingDays,halfLife),'rb')
dailyReturnMatrix = pkl.load(pkl_file)
pkl_file.close()

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\marketValueMatrix_%d.pkl' % (trailingDays),'rb')
marketValueMatrix = pkl.load(pkl_file)
pkl_file.close()

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\outcome_MV_%d_%d_%d_%d.pkl' % (trailingDays,holdingDays, n, m),'rb')
outcome_MV = pkl.load(pkl_file)
file_name = pkl_file.name[-14:-4]
pkl_file.close()

def ic_regression_mvNeutral(df_expos = exposureMatrix ,
                  df_return = dailyReturnMatrix,
                  df_mv = marketValueMatrix,
                  df_outcome = outcome_MV):

  
    icList = []
    irList = []
    index = []
    
    for date_index in range(len(df_expos.index) - 1):
        
        # process data - concat for each cross-section
        ret_temp = df_return.iloc[date_index+1,:]
        exposure_temp = df_expos.iloc[date_index,:]
        mv_temp = df_mv.iloc[date_index,:]
        mv_temp = pd.DataFrame(mv_temp)
        mv_temp.columns = ['market value']
        
        df_temp = pd.concat([ret_temp,exposure_temp],axis = 1)
        df_temp.columns = ['ret','exposure']
        
        # process data - dropna and cut off and so on
        df_temp = df_temp.dropna(axis = 0)
        expos_dev_sup, expos_dev_inf = df_temp['exposure'].median() + 5*(abs(df_temp['exposure'] - df_temp['exposure'].median())).median(), \
                                       df_temp['exposure'].median() - 5*(abs(df_temp['exposure'] - df_temp['exposure'].median())).median()
                                       # dev = median(|x-median(x)|)
                                       
        df_temp.loc[df_temp[df_temp['exposure'] > expos_dev_sup].index,'exposure'] = expos_dev_sup
        df_temp.loc[df_temp[df_temp['exposure'] < expos_dev_inf].index,'exposure'] = expos_dev_inf

        df_temp['exposure'] = ((df_temp['exposure'] - df_temp['exposure'].mean())/df_temp['exposure'].std(ddof=1))     
        df_temp = df_temp.dropna(axis = 0)
        
        # to initialize the market dummy
        market_temp = df_outcome.loc[exposureMatrix.index[date_index]][['MarketValuePortfolio','symbols']]
        mvCodeList = []
        mvPortList = []
        
        for market_index in range(len(market_temp)):
            mvCodeList = mvCodeList + market_temp.iloc[market_index,1]
            mvPortList = mvPortList + [market_temp.iloc[market_index,0],] * len(market_temp.iloc[market_index,1])
        
        marketDict = pd.concat([pd.DataFrame(mvPortList),pd.DataFrame(mvCodeList)],axis = 1)
        marketDict.columns = ['mvPort','code']
        marketDict = marketDict.drop_duplicates('code')
        
        dummyMatrix = pd.get_dummies(marketDict['mvPort'],prefix = 'mvPortfolio_')
        dummyMatrix = pd.concat([marketDict[['code']],dummyMatrix], axis = 1)
        
        # merge back to X, Y to avoid missiong code
        df_reg = pd.merge(df_temp,dummyMatrix, how ='inner', left_index = True, right_on = 'code')
        
        df_reg = df_reg.set_index('code')
        # check for the dummies: np.sum(df_reg.iloc[:,-5:].sum(axis = 1))
        
        df_reg = pd.merge(df_reg,mv_temp,how='inner',left_index = True, right_index = True)
        df_reg = df_reg.dropna(axis = 0)
        
        Y = df_reg.iloc[:,1].values.astype('float')
        X = df_reg.iloc[:,1:-1].values.astype('float')
        weights = 1 / df_reg.iloc[:,-1].values.astype('float')
        modelNeutral = sm.WLS(Y, X, weights = weights, hasconst = False)
        results = modelNeutral.fit()
        resid = results.resid
        
        ic = np.corrcoef([resid,df_reg.iloc[:,0].values.astype('float')])[0][1]
        ir = np.mean(resid) / np.std(resid,ddof = 1)
        
        icList.append(ic)
        irList.append(ir)
        index.append(df_expos.index[date_index])
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize = (30,5))
    
    plt.sca(ax1)
    plt.plot(index, icList)
    plt.axhline(y = 0, c = "r", linestyle = '--', linewidth = 1.5, zorder = 0)
    xticks_count = len(icList) // 4
    xticks_increment = int(len(icList) / xticks_count)
    plt.xticks([xticks_increment * x for x in range(xticks_count)], 
                [index[x] for x in [xticks_increment * y for y in range(xticks_count)]])   
    plt.ylabel('IC')
    plt.title('IC_mv_%s' % file_name)
    plt.legend()
    
    plt.sca(ax2)
    plt.plot(index, irList)
    plt.axhline(y = 0, c = "r", linestyle = '--', linewidth = 1.5, zorder = 0)
    xticks_count = len(icList) // 4
    xticks_increment = int(len(icList) / xticks_count)
    plt.xticks([xticks_increment * x for x in range(xticks_count)], 
                [index[x] for x in [xticks_increment * y for y in range(xticks_count)]])   
    plt.ylabel('IR')
    plt.title('IR_mv_%s' % file_name)
    plt.legend()
    
    plt.tight_layout()
    pp = PdfPages(r'G:\MultiFactor\Factor\Momentum\graph_Information_mvNeutral_%s.pdf' % file_name)      
        
    pp.savefig(f)
    pp.close()

ic_regression_mvNeutral(df_expos = exposureMatrix ,
                  df_return = dailyReturnMatrix,
                  df_mv = marketValueMatrix,
                  df_outcome = outcome_MV)


'''
==================== next comes IC regression for industryNeutral ============
'''

trailingDays = 21
holdingDays = 10
halfLife = 10
n = 10              # 因子分组


pkl_file = open(r'G:\MultiFactor\Factor\Momentum\exposureMatrix_%d_%d_%d.pkl' % (trailingDays,holdingDays,halfLife),'rb')
exposureMatrix = pkl.load(pkl_file)
pkl_file.close()

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\dailyReturnMatrix_%d_%d_%d.pkl' % (trailingDays,holdingDays,halfLife),'rb')
dailyReturnMatrix = pkl.load(pkl_file)
pkl_file.close()

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\industryMatrix_%d.pkl' % trailingDays,'rb')
industryMatrix = pkl.load(pkl_file)
pkl_file.close()

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\marketValueMatrix_%d.pkl' % (trailingDays),'rb')
marketValueMatrix = pkl.load(pkl_file)
pkl_file.close()

pkl_file = open(r'G:\MultiFactor\Factor\Momentum\outcome_Ind_%d_%d_%d.pkl' % (trailingDays,holdingDays, n),'rb')
outcome_Ind = pkl.load(pkl_file)
file_name = pkl_file.name[-12:-4]
pkl_file.close()

def ic_regression_industryNeutral(df_expos = exposureMatrix ,
                  df_return = dailyReturnMatrix,
                  df_ind = industryMatrix,
                  df_mv = marketValueMatrix, 
                  df_outcome = outcome_Ind):

  
    icList = []
    irList = []
    index = []
    
    for date_index in range(len(df_expos.index) - 1):
        
        # process data - concat for each cross-section
        ret_temp = df_return.iloc[date_index+1,:]
        exposure_temp = df_expos.iloc[date_index,:]
        mv_temp = df_mv.iloc[date_index,:]
        mv_temp = pd.DataFrame(mv_temp)
        mv_temp.columns = ['market value']
        
        df_temp = pd.concat([ret_temp,exposure_temp],axis = 1)
        df_temp.columns = ['ret','exposure']
        
        # process data - dropna and cut off and so on
        df_temp = df_temp.dropna(axis = 0)
        expos_dev_sup, expos_dev_inf = df_temp['exposure'].median() + 5*(abs(df_temp['exposure'] - df_temp['exposure'].median())).median(), \
                                       df_temp['exposure'].median() - 5*(abs(df_temp['exposure'] - df_temp['exposure'].median())).median()
                                       # dev = median(|x-median(x)|)
                                       
        df_temp.loc[df_temp[df_temp['exposure'] > expos_dev_sup].index,'exposure'] = expos_dev_sup
        df_temp.loc[df_temp[df_temp['exposure'] < expos_dev_inf].index,'exposure'] = expos_dev_inf

        df_temp['exposure'] = ((df_temp['exposure'] - df_temp['exposure'].mean())/df_temp['exposure'].std(ddof=1))     
        df_temp = df_temp.dropna(axis = 0)
        
        # to initialize the market dummy
        industry_temp = df_outcome.loc[exposureMatrix.index[date_index]][['IndustryPortfolio','symbols']]
        indCodeList = []
        indPortList = []
        
        for market_index in range(len(industry_temp)):
            indCodeList = indCodeList + industry_temp.iloc[market_index,1]
            indPortList = indPortList + [industry_temp.iloc[market_index,0],] * len(industry_temp.iloc[market_index,1])
        
        industryDict = pd.concat([pd.DataFrame(indPortList),pd.DataFrame(indCodeList)],axis = 1)
        industryDict.columns = ['indPort','code']
        industryDict = industryDict.drop_duplicates('code')
        
        dummyMatrix = pd.get_dummies(industryDict['indPort'],prefix = 'indPortfolio_')
        dummyMatrix = pd.concat([industryDict[['code']],dummyMatrix], axis = 1)
        
        # merge back to X, Y to avoid missiong code
        df_reg = pd.merge(df_temp,dummyMatrix, how ='inner', left_index = True, right_on = 'code')
        
        df_reg = df_reg.set_index('code')
        # check for the dummies: np.sum(df_reg.iloc[:,-5:].sum(axis = 1))
        
        df_reg = pd.merge(df_reg, mv_temp, how='inner',left_index = True, right_index = True)
        df_reg = df_reg.dropna(axis = 0)
        
        Y = df_reg.iloc[:,1].values.astype('float')
        X = df_reg.iloc[:,1:-1].values.astype('float')
        weights = 1 / df_reg.iloc[:,-1].values.astype('float')
        modelNeutral = sm.WLS(Y, X, weights = weights, hasconst = False)
        results = modelNeutral.fit()
        resid = results.resid
        
        ic = np.corrcoef([resid,df_reg.iloc[:,0].values.astype('float')])[0][1]
        ir = np.mean(resid) / np.std(resid,ddof = 1)
        
        icList.append(ic)
        irList.append(ir)
        index.append(df_expos.index[date_index])
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize = (30,5))
    
    plt.sca(ax1)
    plt.plot(index, icList)
    plt.axhline(y = 0, c = "r", linestyle = '--', linewidth = 1.5, zorder = 0)
    xticks_count = len(icList) // 4
    xticks_increment = int(len(icList) / xticks_count)
    plt.xticks([xticks_increment * x for x in range(xticks_count)], 
                [index[x] for x in [xticks_increment * y for y in range(xticks_count)]])   
    plt.ylabel('IC')
    plt.title('IC_ind_%s' % file_name)
    plt.legend()
    
    plt.sca(ax2)
    plt.plot(index, irList)
    plt.axhline(y = 0, c = "r", linestyle = '--', linewidth = 1.5, zorder = 0)
    xticks_count = len(icList) // 4
    xticks_increment = int(len(icList) / xticks_count)
    plt.xticks([xticks_increment * x for x in range(xticks_count)], 
                [index[x] for x in [xticks_increment * y for y in range(xticks_count)]])   
    plt.ylabel('IR')
    plt.title('IR_ind_%s' % file_name)
    plt.legend()
    
    plt.tight_layout()
    pp = PdfPages(r'G:\MultiFactor\Factor\Momentum\graph_Information_indNeutral_%s.pdf' % file_name)      
        
    pp.savefig(f)
    pp.close()

ic_regression_industryNeutral(df_expos = exposureMatrix ,
                  df_return = dailyReturnMatrix,
                  df_ind = industryMatrix,
                  df_mv = marketValueMatrix, 
                  df_outcome = outcome_Ind)