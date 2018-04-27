# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:57:14 2018

@author: Alan
"""
from jaqs.data import DataApi
from jaqs.data import RemoteDataService
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH

import pandas as pd
from datetime import datetime as ddt
import pickle
from glob import glob
import os

def download_to_local_hdf(start_date = 20100101, end_date = 20180228):
    api = DataApi(addr='tcp://data.quantos.org:8910')
    api.login('18011342046','eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjA5MDQxNzY1NDciLCJpc3MiOiJhdXRoMCIsImlkIjoiMTgwMTEzNDIwNDYifQ.cvUCM4y_btI8W_zbUHPbzSS_vxqMjcZR22lESjb5eQY')
    data_config = jutil.read_json(DATA_CONFIG_PATH)
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    codeList, msg = api.query(
                view="jz.instrumentInfo", #这是基础信息表
                fields="status,list_date, fullname_en, market", 
                filter="inst_type=1&status=1&symbol=", 
                data_format='pandas')
    codeList = codeList[codeList['symbol'].apply(lambda x : (x[-2:]=='SZ') | (x[-2:]=='SH'))]['symbol']
    codeList = list(set(codeList))
#    codeStr = ','.join(codeList)

    p = {}
    
    for code in codeList:
        sample = api.daily(code,start_date=start_date, end_date=end_date,
                           fields='', adjust_mode = None)[0]
        try:
            assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
        except AssertionError:
            print('%s is of shiness' % code)
            continue
        sample = sample[['trade_date','symbol','close','vwap','volume','turnover',
                         'trade_status']]

        trade_status_convert = {'交易':1,
                                '停牌':-1}
        sample['trade_status'] = sample['trade_status'].map(trade_status_convert)
        adj_factor = api.query(view='lb.secAdjFactor',field = '',
                               filter='symbol=%s&start_date=%d&end_date=%d' %(code,start_date,end_date),
                               data_format='pandas')[0]
        sample = pd.concat([sample,adj_factor['adjust_factor']],axis = 1 )
#        sample['adjust_factor'] = sample['adjust_factor'] / sample.iloc[-1,-3]
#        sample['close_pre'] = sample['close'] * sample['adjust_factor']
#        sample.iloc[-1,-1] = sample.iloc[-1,2]
        
        industry_1 = ds.query_industry_daily(code,start_date=start_date,
                                           end_date=end_date,type_='SW')
        industry_2 = ds.query_industry_daily(code,start_date=start_date,
                                           end_date=end_date,type_='SW',
                                           level=2)
        industry_1.columns = ['industry_1']
        industry_2.columns = ['industry_2']
        industry_1.index = range(len(industry_1))
        industry_2.index = range(len(industry_2))
        sample = pd.concat([sample,industry_1['industry_1']],axis=1)
        sample = pd.concat([sample,industry_2['industry_2']],axis=1)
        sample = sample.dropna(subset=['trade_date','symbol'])
        sample['trade_date'] = sample['trade_date'].apply(lambda x : int(x))
        sample['trade_status'] = sample['trade_status'].apply(lambda x : int(x))
        
        p[code] = sample
        
    date_max = max(sample['trade_date'])
    hdf = pd.HDFStore(r'G:\MultiFactor\Data\TimeSeriesDaily\ts_Daily_%d.h5' % date_max,
                      mode='w',format='table',complevel=9, complib='blosc')
    
    for key,value in p.items():
        hdf.put(key=key,value=value,format='table',columns=True)
#        print(key)
    hdf.close()
    
#    panel = pd.Panel(p)
#    return panel 
    
    pkl_file = open(r'G:\MultiFactor\Data\TimeSeriesDaily\ts_Daily_%d.pkl' % date_max, 'wb')
    pickle.dump(p,pkl_file)
    pkl_file.close()    
    
    return p

'''--------------download_to_local_hdf is done-------------------'''



def download_to_local_pkl(start_date = 20100101, end_date = 20180228):
    api = DataApi(addr='tcp://data.quantos.org:8910')
    api.login('18011342046','eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjA5MDQxNzY1NDciLCJpc3MiOiJhdXRoMCIsImlkIjoiMTgwMTEzNDIwNDYifQ.cvUCM4y_btI8W_zbUHPbzSS_vxqMjcZR22lESjb5eQY')
    data_config = jutil.read_json(DATA_CONFIG_PATH)
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    codeList, msg = api.query(
                view="jz.instrumentInfo", #这是基础信息表
                fields="status,list_date, fullname_en, market", 
                filter="inst_type=1&status=1&symbol=", 
                data_format='pandas')
    codeList = codeList[codeList['symbol'].apply(lambda x : (x[-2:]=='SZ') | (x[-2:]=='SH'))]['symbol']
    codeList = list(set(codeList))
#    codeStr = ','.join(codeList)
 
    for code in codeList:
#        print(code)
        sample = api.daily(code,start_date=start_date, end_date=end_date,
                           fields='', adjust_mode = None)[0]
        try:
            assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
        except AssertionError:
            print('%s is of shiness' % code)
            continue
        sample = sample[['trade_date','symbol','close','vwap','volume','turnover',
                         'trade_status']]

        trade_status_convert = {'交易':1,
                                '停牌':-1}
        sample['trade_status'] = sample['trade_status'].map(trade_status_convert)
        adj_factor = api.query(view='lb.secAdjFactor',field = '',
                               filter='symbol=%s&start_date=%d&end_date=%d' %(code,start_date,end_date),
                               data_format='pandas')[0]
        try:
            sample = pd.concat([sample,adj_factor['adjust_factor']],axis = 1 )
        except TypeError:
            print('%s encounters None-Type Error!' % code)
            continue
        
#        sample['adjust_factor'] = sample['adjust_factor'] / sample.iloc[-1,-1]
#        sample['close_pre'] = sample['close'] * sample['adjust_factor']
#        sample.iloc[-1,-1] = sample.iloc[-1,2]
        
        industry_1 = ds.query_industry_daily(code,start_date=start_date,
                                           end_date=end_date,type_='SW')
        industry_2 = ds.query_industry_daily(code,start_date=start_date,
                                           end_date=end_date,type_='SW',
                                           level=2)
        industry_1.columns = ['industry_1']
        industry_2.columns = ['industry_2']
        industry_1.index = range(len(industry_1))
        industry_2.index = range(len(industry_2))
        sample = pd.concat([sample,industry_1['industry_1']],axis=1)
        sample = pd.concat([sample,industry_2['industry_2']],axis=1)
        sample = sample.dropna(subset=['trade_date','symbol'])
        sample['trade_date'] = sample['trade_date'].apply(lambda x : int(x))
        sample['trade_status'] = sample['trade_status'].apply(lambda x : int(x))
        
        pkl_file = open(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.pkl' % code, 'wb')
        pickle.dump(sample,pkl_file)
        pkl_file.close()
        
def update_local_dict_hdf(update_hdf = True):
    api = DataApi(addr='tcp://data.quantos.org:8910')
    api.login('18011342046','eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjA5MDQxNzY1NDciLCJpc3MiOiJhdXRoMCIsImlkIjoiMTgwMTEzNDIwNDYifQ.cvUCM4y_btI8W_zbUHPbzSS_vxqMjcZR22lESjb5eQY')
    data_config = jutil.read_json(DATA_CONFIG_PATH)
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    myFolder =r'G:\MultiFactor\Data\TimeSeriesDaily'
    fileList = glob(os.path.join(myFolder,'*.pkl'))
    file = fileList[-1][-21:-4]
    update_date = int(file[-8:])
    today = ddt.today().year * 10000 + ddt.today().month * 100 + ddt.today().day

    pkl_file = open(r'G:\MultiFactor\Data\TimeSeriesDaily\ts_Daily_%d.pkl' % update_date, 'rb')
    sample_original = pickle.load(pkl_file)
    pkl_file.close()
    
    keyList = list(sample_original.keys())
    
    codeList, msg = api.query(
                view="jz.instrumentInfo", #这是基础信息表
                fields="status,list_date, fullname_en, market", 
                filter="inst_type=1&status=1&symbol=", 
                data_format='pandas')
    codeList = codeList[codeList['symbol'].apply(lambda x : (x[-2:]=='SZ') | (x[-2:]=='SH'))]['symbol']
    codeList = list(set(codeList))

    diffList = list([x for x in codeList if x not in keyList])

    if len(diffList) > 0:
        
        original_date = 20100101
        
        for code in diffList:
#            print(code)
            sample = api.daily(code,start_date=original_date, end_date=today,
                               fields='', adjust_mode = None)[0]
            try:
                assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
            except AssertionError:
                print('%s is of shiness' % code)
                continue
            sample = sample[['trade_date','symbol','close','vwap','volume','turnover',
                             'trade_status']]

            trade_status_convert = {'交易':1,
                                    '停牌':-1}
            sample['trade_status'] = sample['trade_status'].map(trade_status_convert)
            adj_factor = api.query(view='lb.secAdjFactor',field = '',
                                   filter='symbol=%s&start_date=%d&end_date=%d' %(code,original_date,today),
                                   data_format='pandas')[0]
            try:
                sample = pd.concat([sample,adj_factor['adjust_factor']],axis = 1 )
            except TypeError:
                print('%s encounters None-Type Error!' % code)
                continue
        
#            sample['adjust_factor'] = sample['adjust_factor'] / sample.iloc[-1,-1]
#            sample['close_pre'] = sample['close'] * sample['adjust_factor']
#            sample.iloc[-1,-1] = sample.iloc[-1,2]
        
            industry_1 = ds.query_industry_daily(code,start_date=original_date,
                                                 end_date=today,type_='SW')
            industry_2 = ds.query_industry_daily(code,start_date=original_date,
                                                 end_date=today,type_='SW',
                                                 level=2)
            industry_1.columns = ['industry_1']
            industry_2.columns = ['industry_2']
            industry_1.index = range(len(industry_1))
            industry_2.index = range(len(industry_2))
            sample = pd.concat([sample,industry_1['industry_1']],axis=1)
            sample = pd.concat([sample,industry_2['industry_2']],axis=1)
            sample = sample.dropna(subset=['trade_date','symbol'])
            sample['trade_date'] = sample['trade_date'].apply(lambda x : int(x))
            sample['trade_status'] = sample['trade_status'].apply(lambda x : int(x))
        
            sample_original[code] = sample
            print('%s is a new arrival!' % code)
    
    original_date = 20100101

    for code in keyList:
        try:
            sample = sample_original[code]
        except:
            print('For code in keyList, %s cannot be abstracted from .pkl dict format' % code)
            continue
        
        if (sample is None) | (len(sample) <= 1):
            sample = api.daily(code,start_date=original_date, end_date=today,
                               fields='', adjust_mode = None)[0]
            try:
                assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
            except AssertionError:
                print('%s is of shiness' % code)
                continue

            sample = sample[['trade_date','symbol','close','vwap','volume','turnover',
                             'trade_status']]
            trade_status_convert = {'交易':1,
                                    '停牌':-1}
            sample['trade_status'] = sample['trade_status'].map(trade_status_convert)
            adj_factor = api.query(view='lb.secAdjFactor',field = '',
                                   filter='symbol=%s&start_date=%d&end_date=%d' %(code,original_date,today),
                                   data_format='pandas')[0]
            try:
                sample = pd.concat([sample,adj_factor['adjust_factor']],axis = 1 )
            except TypeError:
                print('%s encounters None-Type Error!' % code)
                continue
            
#            sample['adjust_factor'] = sample['adjust_factor'] / sample.iloc[-1,-1]
#            sample['close_pre'] = sample['close'] * sample['adjust_factor']
#            sample.iloc[-1,-1] = sample.iloc[-1,2]
            
            industry_1 = ds.query_industry_daily(code,start_date=original_date,
                                                 end_date=today,type_='SW')
            industry_2 = ds.query_industry_daily(code,start_date=original_date,
                                                 end_date=today,type_='SW',
                                                 level=2)
            industry_1.columns = ['industry_1']
            industry_2.columns = ['industry_2']
            industry_1.index = range(len(industry_1))
            industry_2.index = range(len(industry_2))
            sample = pd.concat([sample,industry_1['industry_1']],axis=1)
            sample = pd.concat([sample,industry_2['industry_2']],axis=1)
            sample = sample.dropna(subset=['trade_date','symbol'])
            sample['trade_date'] = sample['trade_date'].apply(lambda x : int(x))
            sample['trade_status'] = sample['trade_status'].apply(lambda x : int(x))                
            
            sample[code] = sample
            print('%s has something wrong but now it is settled!' % code)
        
        else:
            try:
                sample = sample_original[code]
            except:
                print('For code in keyList, %s cannot be abstracted from .pkl format' % code)
                pkl_file.close()
                continue 
            
            date_max = max(sample['trade_date'])
            start_date = date_max
            end_date = today
            sample_append = api.daily(code,start_date=start_date, end_date=end_date,
                               fields='', adjust_mode = None)[0]
            try:
                assert ((isinstance(sample_append,pd.DataFrame)) & (len(sample_append) > 0))
            except AssertionError:
                print('For updated data, %s is of shiness' % code)
                continue
            sample_append = sample_append[['trade_date','symbol','close','vwap','volume','turnover',
                             'trade_status']]
            
            trade_status_convert = {'交易':1,
                                    '停牌':-1}
            sample_append['trade_status'] = sample_append['trade_status'].map(trade_status_convert)
            adj_factor = api.query(view='lb.secAdjFactor',field = '',
                                   filter='symbol=%s&start_date=%d&end_date=%d' %(code,start_date,end_date),
                                   data_format='pandas')[0]
            try:
                sample_append = pd.concat([sample_append,adj_factor['adjust_factor']],axis = 1 )
            except TypeError:
                print('%s encounters None-Type Error!' % code)
                continue
    
            sample_append['adjust_factor'] = sample_append['adjust_factor'] / sample_append.iloc[-1,-1]
            sample_append['close_pre'] = sample['close'] * sample['adjust_factor']
            sample_append.iloc[-1,-1] = sample_append.iloc[-1,2]
            
            industry_1 = ds.query_industry_daily(code,start_date=start_date,
                                                 end_date=end_date,type_='SW')
            industry_2 = ds.query_industry_daily(code,start_date=start_date,
                                                 end_date=end_date,type_='SW',
                                                 level=2)
            industry_1.columns = ['industry_1']
            industry_2.columns = ['industry_2']
            industry_1.index = range(len(industry_1))
            industry_2.index = range(len(industry_2))
            sample_append = pd.concat([sample_append,industry_1['industry_1']],axis=1)
            sample_append = pd.concat([sample_append,industry_2['industry_2']],axis=1)
            sample_append = sample_append.dropna(subset=['trade_date','symbol'])
            sample_append['trade_date'] = sample_append['trade_date'].apply(lambda x : int(x))
            sample_append['trade_status'] = sample_append['trade_status'].apply(lambda x : int(x))
            
            sample = pd.concat([sample,sample_append],axis=0)
            sample = sample.drop_duplicates(subset=['trade_date','symbol'])
            sample.index = range(len(sample))
            
            sample_original[code] = sample
    
    date_max = today
            
    pkl_file = open(r'G:\MultiFactor\Data\TimeSeriesDaily\ts_Daily_%d.pkl' % date_max, 'wb')
    pickle.dump(sample_original,pkl_file)
    pkl_file.close()
    
    if update_hdf:
        p = sample_original.copy()
        today = ddt.today().year * 10000 + ddt.today().month * 100 + ddt.today().day
        hdf = pd.HDFStore(r'G:\MultiFactor\Data\TimeSeriesDaily\ts_Daily_%d.h5' % today,
                          mode='w',format='table',complevel=9, complib='blosc')
        for key,value in p.items():
            hdf.put(key=key,value=value,format='table',columns=True)   
        hdf.close()
    
def update_local_pkl():
    api = DataApi(addr='tcp://data.quantos.org:8910')
    api.login('18011342046','eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjA5MDQxNzY1NDciLCJpc3MiOiJhdXRoMCIsImlkIjoiMTgwMTEzNDIwNDYifQ.cvUCM4y_btI8W_zbUHPbzSS_vxqMjcZR22lESjb5eQY')
    data_config = jutil.read_json(DATA_CONFIG_PATH)
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    codeList, msg = api.query(
                view="jz.instrumentInfo", #这是基础信息表
                fields="status,list_date, fullname_en, market", 
                filter="inst_type=1&status=1&symbol=", 
                data_format='pandas')
    codeList = codeList[codeList['symbol'].apply(lambda x : (x[-2:]=='SZ') | (x[-2:]=='SH'))]['symbol']
    codeList = list(set(codeList))
#    codeStr = ','.join(codeList)
    
    myFolder =r'G:\MultiFactor\Data\TimeSeriesDaily'
    fileList = glob(os.path.join(myFolder,'*.pkl'))
    fileList = list((x[-13:-4] for x in fileList))
    diffList = list([x for x in codeList if x not in fileList])
    
    today = ddt.today().year * 10000 + ddt.today().month * 100 + ddt.today().day
    
    if len(diffList) > 0:
        
        original_date = 20100101
    
        for code in diffList:
#            print(code)
            sample = api.daily(code,start_date=original_date, end_date=today,
                               fields='', adjust_mode = None)[0]
            try:
                assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
            except AssertionError:
                print('%s is of shiness' % code)
                continue
            sample = sample[['trade_date','symbol','close','vwap','volume','turnover',
                             'trade_status']]

            trade_status_convert = {'交易':1,
                                    '停牌':-1}
            sample['trade_status'] = sample['trade_status'].map(trade_status_convert)
            adj_factor = api.query(view='lb.secAdjFactor',field = '',
                                   filter='symbol=%s&start_date=%d&end_date=%d' %(code,original_date,today),
                                   data_format='pandas')[0]
            try:
                sample = pd.concat([sample,adj_factor['adjust_factor']],axis = 1 )
            except TypeError:
                print('%s encounters None-Type Error!' % code)
                continue
        
#            sample['adjust_factor'] = sample['adjust_factor'] / sample.iloc[-1,-1]
#            sample['close_pre'] = sample['close'] * sample['adjust_factor']
#            sample.iloc[-1,-1] = sample.iloc[-1,2]
        
            industry_1 = ds.query_industry_daily(code,start_date=original_date,
                                                 end_date=today,type_='SW')
            industry_2 = ds.query_industry_daily(code,start_date=original_date,
                                                 end_date=today,type_='SW',
                                                 level=2)
            industry_1.columns = ['industry_1']
            industry_2.columns = ['industry_2']
            industry_1.index = range(len(industry_1))
            industry_2.index = range(len(industry_2))
            sample = pd.concat([sample,industry_1['industry_1']],axis=1)
            sample = pd.concat([sample,industry_2['industry_2']],axis=1)
            sample = sample.dropna(subset=['trade_date','symbol'])
            sample['trade_date'] = sample['trade_date'].apply(lambda x : int(x))
            sample['trade_status'] = sample['trade_status'].apply(lambda x : int(x))
        
            pkl_file = open(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.pkl' % code, 'wb')
            pickle.dump(sample,pkl_file)
            pkl_file.close()
    
    original_date = 20100101
    for code in fileList:
        pkl_file = open(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.pkl' % code, 'rb')
        try:
            sample = pickle.load(pkl_file)
            pkl_file.close()
        except:
            print('For code in fileList, %s cannot be abstracted from .pkl format' % code)
            pkl_file.close()
            continue
        
        if (sample is None) | (len(sample) <= 1):
            sample = api.daily(code,start_date=original_date, end_date=today,
                               fields='', adjust_mode = None)[0]
            try:
                assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
            except AssertionError:
                print('%s is of shiness' % code)
                continue
            sample = sample[['trade_date','symbol','close','vwap','volume','turnover',
                             'trade_status']]
            
            trade_status_convert = {'交易':1,
                                    '停牌':-1}
            sample['trade_status'] = sample['trade_status'].map(trade_status_convert)
            adj_factor = api.query(view='lb.secAdjFactor',field = '',
                                   filter='symbol=%s&start_date=%d&end_date=%d' %(code,original_date,today),
                                   data_format='pandas')[0]
            try:
                sample = pd.concat([sample,adj_factor['adjust_factor']],axis = 1 )
            except TypeError:
                print('%s encounters None-Type Error!' % code)
                continue
            
#            sample['adjust_factor'] = sample['adjust_factor'] / sample.iloc[-1,-1]
#            sample['close_pre'] = sample['close'] * sample['adjust_factor']
#            sample.iloc[-1,-1] = sample.iloc[-1,2]
            
            industry_1 = ds.query_industry_daily(code,start_date=original_date,
                                                 end_date=today,type_='SW')
            industry_2 = ds.query_industry_daily(code,start_date=original_date,
                                                 end_date=today,type_='SW',
                                                 level=2)
            industry_1.columns = ['industry_1']
            industry_2.columns = ['industry_2']
            industry_1.index = range(len(industry_1))
            industry_2.index = range(len(industry_2))
            sample = pd.concat([sample,industry_1['industry_1']],axis=1)
            sample = pd.concat([sample,industry_2['industry_2']],axis=1)
            sample = sample.dropna(subset=['trade_date','symbol'])
            sample['trade_date'] = sample['trade_date'].apply(lambda x : int(x))
            sample['trade_status'] = sample['trade_status'].apply(lambda x : int(x))                
            
            pkl_file = open(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.pkl' % code, 'wb')
            pickle.dump(sample,pkl_file)
            pkl_file.close()
            print('%s has something wrong but now it is settled!' % code)
        
        else:
            pkl_file = open(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.pkl' % code, 'rb')
            try:
                sample = pickle.load(pkl_file)
                pkl_file.close()
            except:
                print('For code in fileList, %s cannot be abstracted from .pkl format' % code)
                pkl_file.close()
                continue 
            
            date_max = max(sample['trade_date'])
            start_date = date_max
            end_date = today
            sample_append = api.daily(code,start_date=start_date, end_date=end_date,
                               fields='', adjust_mode = None)[0]
            try:
                assert ((isinstance(sample_append,pd.DataFrame)) & (len(sample_append) > 0))
            except AssertionError:
                print('For updated data, %s is of shiness' % code)
                continue
            sample_append = sample_append[['trade_date','symbol','close','vwap','volume','turnover',
                             'trade_status']]
            
            trade_status_convert = {'交易':1,
                                    '停牌':-1}
            sample_append['trade_status'] = sample_append['trade_status'].map(trade_status_convert)
            adj_factor = api.query(view='lb.secAdjFactor',field = '',
                                   filter='symbol=%s&start_date=%d&end_date=%d' %(code,start_date,end_date),
                                   data_format='pandas')[0]
            try:
                sample_append = pd.concat([sample_append,adj_factor['adjust_factor']],axis = 1 )
            except TypeError:
                print('%s encounters None-Type Error!' % code)
                continue
    
            sample_append['adjust_factor'] = sample_append['adjust_factor'] / sample_append.iloc[-1,-1]
            sample_append['close_pre'] = sample['close'] * sample['adjust_factor']
            sample_append.iloc[-1,-1] = sample_append.iloc[-1,2]
            
            ds = RemoteDataService()
            ds.init_from_config(data_config)
            '''ds 放在这里是因为程序跑太久了，失去了 connection '''
            
            industry_1 = ds.query_industry_daily(code,start_date=start_date,
                                                 end_date=end_date,type_='SW')
            industry_2 = ds.query_industry_daily(code,start_date=start_date,
                                                 end_date=end_date,type_='SW',
                                                 level=2)
            industry_1.columns = ['industry_1']
            industry_2.columns = ['industry_2']
            industry_1.index = range(len(industry_1))
            industry_2.index = range(len(industry_2))
            sample_append = pd.concat([sample_append,industry_1['industry_1']],axis=1)
            sample_append = pd.concat([sample_append,industry_2['industry_2']],axis=1)
            sample_append = sample_append.dropna(subset=['trade_date','symbol'])
            sample_append['trade_date'] = sample_append['trade_date'].apply(lambda x : int(x))
            sample_append['trade_status'] = sample_append['trade_status'].apply(lambda x : int(x))
            
            sample = pd.concat([sample,sample_append],axis=0)
            sample = sample.drop_duplicates(subset=['trade_date','symbol'])
            sample.index = range(len(sample))
            
            pkl_file = open(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.pkl' % code, 'wb')
            pickle.dump(sample,pkl_file)
            pkl_file.close()
#            print(code)

'''
def update_local_hdf():
    p = update_local_dict()
    today = ddt.today().year * 10000 + ddt.today().month * 100 + ddt.today().day
    hdf = pd.HDFStore(r'G:\MultiFactor\Data\TimeSeriesDaily\ts_Daily_%d.h5' % today,
                      mode='w',format='table',complevel=9, complib='blosc')
    for key,value in p.items():
        hdf.put(key=key,value=value,format='table',columns=True)   
    hdf.close()

    这个函数要在 update_local_hdf 之后运行才行！
'''

def select_ts_pkl(code=None,start_date=20170101,end_date=20171231):
    if not isinstance(start_date,int):
        print('start_date and end_date must be int type')
        return False
    if not isinstance(end_date,int):
        print('start_date and end_date must be int type')
        return False
    
    if isinstance(code,list):
        df = pd.DataFrame([])
        for c in code:
            try:
                pkl_file = open(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.pkl' % c, 'rb')
                sample = pickle.load(pkl_file)
                pkl_file.close()
                sample = sample[(sample['trade_date'] >= start_date) & (sample['trade_date'] <= end_date)]
                sample['adjust_factor'] = sample['adjust_factor'] / sample.iloc[-1,-3]
                sample['close_pre'] = sample['close'] * sample['adjust_factor']
            except:
                pkl_file.close()
                print('%s cannot be abstracted from .pkl format!')
                continue
            df = pd.concat([df,sample],axis = 0)
        return df
    
    elif isinstance(code,str):
        try:
            pkl_file = open(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.pkl' % code, 'rb')
            df = pickle.load(pkl_file)
            pkl_file.close()
            df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
            df['adjust_factor'] = df['adjust_factor'] / df.iloc[-1,-3]
            df['close_pre'] = df['close'] * df['adjust_factor']
            return df
        except:
            pkl_file.close()
            print('%s cannot be abstracted from .pkl format!')
            return False
        
    elif code is None:
        myFolder =r'G:\MultiFactor\Data\TimeSeriesDaily'
        fileList = glob(os.path.join(myFolder,'*.pkl'))
        fileList = list((x[-13:-4] for x in fileList))
        df = pd.DataFrame([])
        for code in fileList:
            try:
                pkl_file = open(r'G:\MultiFactor\Data\TimeSeriesDaily\%s.pkl' % code, 'rb')
                sample = pickle.load(pkl_file)
                sample = sample[(sample['trade_date'] >= start_date) & (sample['trade_date'] <= end_date)]
                sample['adjust_factor'] = sample['adjust_factor'] / sample.iloc[-1,-3]
                sample['close_pre'] = sample['close'] * sample['adjust_factor']
                pkl_file.close()
            except:
                pkl_file.close()
                print('%s cannot be abstracted from .pkl format!')
                continue
            df = pd.concat([df,sample],axis = 0)
        return df
    
    else:
        print('Something wrong about the param of code you input!')
        return False

def select_ts_hdf(code=None,start_date=20170101,end_date=20171231):
    if not isinstance(start_date,int):
        print('start_date and end_date must be int type')
        return False
    if not isinstance(end_date,int):
        print('start_date and end_date must be int type')
        return False
    
    myFolder =r'G:\MultiFactor\Data\TimeSeriesDaily'
    try:
        fileList = glob(os.path.join(myFolder,'*.h5'))
        fileList.sort()
        date_max = int(fileList[-1][-11:-3])
    except IndexError:
        print('There is no *.h5 file in the folder! Initialize it first!')
        return False
    
    if isinstance(code,list):
        df = pd.DataFrame([])
        hdf = pd.HDFStore(r'G:\MultiFactor\Data\TimeSeriesDaily\ts_Daily_%d.h5' % date_max,'r')
        for c in code:
            try:
                sample = getattr(hdf,c)
                sample['adjust_factor'] = sample['adjust_factor'] / sample.iloc[-1,-3]
                sample['close_pre'] = sample['close'] * sample['adjust_factor']
            except AttributeError:
                print('object has no attribute %s' % c)
                continue
            df = pd.concat([df,sample],axis=0)
        hdf.close()
        return df
        
    elif isinstance(code,str):
        hdf = pd.HDFStore(r'G:\MultiFactor\Data\TimeSeriesDaily\ts_Daily_%d.h5' % date_max,'r')
        try:
            df = getattr(hdf,code)
            df['adjust_factor'] = df['adjust_factor'] / df.iloc[-1,-3]
            df['close_pre'] = df['close'] * df['adjust_factor']
            hdf.close()
        except AttributeError:
            print('object has no attribute %s' % c)
            hdf.close()
            return False
        return df        
    
    elif code is None:
        df = pd.DataFrame([])
        hdf = pd.HDFStore(r'G:\MultiFactor\Data\TimeSeriesDaily\ts_Daily_%d.h5' % date_max,'r')
        for key in hdf.keys():
            try:
                sample = getattr(hdf,key)
                sample['adjust_factor'] = sample['adjust_factor'] / sample.iloc[-1,-3]
                sample['close_pre'] = sample['close'] * sample['adjust_factor']                
            except AttributeError:
                print('object has no attribute %s' % c)
                continue
            df = pd.concat([df,sample],axis=0)
        return df
    
    else:
        print('Something wrong about the param of code you input!')
        return False    

#download_to_local_hdf()
download_to_local_pkl()
#update_local_pkl()
#update_local_dict_hdf()
#sample = select_ts_pkl(code='000001.SZ',start_date=20170101,end_date=20171231)
#sample = select_ts_hdf(code='000001.SZ',start_date=20170101,end_date=20171231)

'''
----------------------------------------------------------------------
----------------------------------------------------------------------
===============以上都是市场数据，以下的数据有市值等等数据================
----------------------------------------------------------------------
----------------------------------------------------------------------'''


def download_ref_daily_pkl(start_date = 20100101, end_date = 20180228):
    api = DataApi(addr='tcp://data.quantos.org:8910')
    api.login('18011342046','eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjA5MDQxNzY1NDciLCJpc3MiOiJhdXRoMCIsImlkIjoiMTgwMTEzNDIwNDYifQ.cvUCM4y_btI8W_zbUHPbzSS_vxqMjcZR22lESjb5eQY')
    data_config = jutil.read_json(DATA_CONFIG_PATH)
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    codeList, msg = api.query(
                view="jz.instrumentInfo", #这是基础信息表
                fields="status,list_date, fullname_en, market", 
                filter="inst_type=1&status=1&symbol=", 
                data_format='pandas')
    codeList = codeList[codeList['symbol'].apply(lambda x : (x[-2:]=='SZ') | (x[-2:]=='SH'))]['symbol']
    codeList = list(set(codeList))     
    
    for code in codeList:
        ds = RemoteDataService()
        ds.init_from_config(data_config)
        sample = ds.query_lb_dailyindicator(code,start_date=start_date,end_date=end_date,
                fields='trade_date,total_mv,float_mv,pe,pb,turnover_ratio,free_turnover_ratio')[0]
        try:
            assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
        except AssertionError:
            print('%s is of shiness in ref_daily data' % code)
            continue
        sample = sample[['trade_date','symbol','total_mv','float_mv','pe','pb',
                         'turnover_ratio','free_turnover_ratio']]
        sample['trade_date'] = sample['trade_date'].apply(lambda x : int(x))
        
        pkl_file = open(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\mv_%s.pkl' % code, 'wb')
        pickle.dump(sample,pkl_file)
        pkl_file.close()

def download_ref_daily_hdf(start_date = 20100101, end_date = 20180228):
    api = DataApi(addr='tcp://data.quantos.org:8910')
    api.login('18011342046','eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjA5MDQxNzY1NDciLCJpc3MiOiJhdXRoMCIsImlkIjoiMTgwMTEzNDIwNDYifQ.cvUCM4y_btI8W_zbUHPbzSS_vxqMjcZR22lESjb5eQY')
    data_config = jutil.read_json(DATA_CONFIG_PATH)
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    codeList, msg = api.query(
                view="jz.instrumentInfo", #这是基础信息表
                fields="status,list_date, fullname_en, market", 
                filter="inst_type=1&status=1&symbol=", 
                data_format='pandas')
    codeList = codeList[codeList['symbol'].apply(lambda x : (x[-2:]=='SZ') | (x[-2:]=='SH'))]['symbol']
    codeList = list(set(codeList))     

    p = {}
 
    for code in codeList:
#        print(code)
        sample = ds.query_lb_dailyindicator(code,start_date=start_date,end_date=end_date,
                fields='trade_date,total_mv,float_mv,pe,pb,turnover_ratio,free_turnover_ratio')[0]
        try:
            assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
        except AssertionError:
            print('%s is of shiness in ref_daily data' % code)
            continue
        sample = sample[['trade_date','symbol','total_mv','float_mv','pe','pb',
                         'turnover_ratio','free_turnover_ratio']]
        sample['trade_date'] = sample['trade_date'].apply(lambda x : int(x))
        p[code] = sample
        
    hdf = pd.HDFStore(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\ts_RefDaily_20180228.h5',
                      mode='w',format='table',complevel=9, complib='blosc') 
    for key,value in p.items():
        hdf.put(key=key,value=value,format='table',columns=True)
    hdf.close()
    
    pkl_file = open(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\ts_RefDaily_20180228.pkl', 'wb')
    pickle.dump(p,pkl_file)
    pkl_file.close()
    
def update_RefDaily_dict_hdf(update_hdf = True):
    api = DataApi(addr='tcp://data.quantos.org:8910')
    api.login('18011342046','eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjA5MDQxNzY1NDciLCJpc3MiOiJhdXRoMCIsImlkIjoiMTgwMTEzNDIwNDYifQ.cvUCM4y_btI8W_zbUHPbzSS_vxqMjcZR22lESjb5eQY')
    data_config = jutil.read_json(DATA_CONFIG_PATH)
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    myFolder =r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv'
    fileList = glob(os.path.join(myFolder,'*.pkl'))
    fileList.sort()
    file = fileList[-1]
    update_date = int(file[-12:-4])
    today = ddt.today().year * 10000 + ddt.today().month * 100 + ddt.today().day

    pkl_file = open(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\ts_RefDaily_%d.pkl' % update_date, 'rb')
    sample_original = pickle.load(pkl_file)
    pkl_file.close()
    
    keyList = list(sample_original.keys())
    
    codeList, msg = api.query(
                view="jz.instrumentInfo", #这是基础信息表
                fields="status,list_date, fullname_en, market", 
                filter="inst_type=1&status=1&symbol=", 
                data_format='pandas')
    codeList = codeList[codeList['symbol'].apply(lambda x : (x[-2:]=='SZ') | (x[-2:]=='SH'))]['symbol']
    codeList = list(set(codeList))

    diffList = list([x for x in codeList if x not in keyList])
    
    original_date = 20100101
    
    if len(diffList) > 0:   
        for code in diffList:
            sample = ds.query_lb_dailyindicator(code,start_date=original_date,end_date=today,
                     fields='trade_date,total_mv,float_mv,pe,pb,turnover_ratio,free_turnover_ratio')[0]
            try:
                assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
            except AssertionError:
                print('%s is of shiness in ref_daily data' % code)
                continue
            sample = sample[['trade_date','symbol','total_mv','float_mv','pe','pb',
                             'turnover_ratio','free_turnover_ratio']]
            sample['trade_date'] = sample['trade_date'].apply(lambda x : int(x))
        
            sample_original[code] = sample
            print('%s is a new arrival for RefDaily List!' % code)
    
    for code in keyList:
        try:
            sample = sample_original[code]
        except:
            print('For code in keyList, %s cannot be abstracted from .pkl dict format in RefDaily data' % code)
            continue
        if (sample is None) | (len(sample) <= 1):
            sample = ds.query_lb_dailyindicator(code,start_date=original_date,end_date=today,
                    fields='trade_date,total_mv,float_mv,pe,pb,turnover_ratio,free_turnover_ratio')[0]
            try:
                assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
            except AssertionError:
                print('%s is of shiness in ref_daily data' % code)
                continue
            sample = sample[['trade_date','symbol','total_mv','float_mv','pe','pb',
                             'turnover_ratio','free_turnover_ratio']]
            sample['trade_date'] = sample['trade_date'].apply(lambda x : int(x))
            
            sample[code] = sample
            print('%s has something wrong but now it is settled in ref_daily data!' % code)
        
        else:
            try:
                sample = sample_original[code]
            except:
                print('For code in keyList, %s cannot be abstracted from .pkl format in ref_daily data' % code)
                pkl_file.close()
                continue 
            
            date_max = max(sample['trade_date'])
            start_date = date_max
            end_date = today
            sample_append = ds.query_lb_dailyindicator(code,start_date=start_date,end_date=end_date,
                            fields='trade_date,total_mv,float_mv,pe,pb,turnover_ratio,free_turnover_ratio')[0]
            try:
                assert ((isinstance(sample_append,pd.DataFrame)) & (len(sample_append) > 0))
            except AssertionError:
                print('For updated data, %s is of shiness' % code)
                continue
            sample_append = sample_append[['trade_date','symbol','total_mv',
                                           'float_mv','pe','pb','turnover_ratio',
                                           'free_turnover_ratio']]
            sample_append['trade_date'] = sample_append['trade_date'].apply(lambda x : int(x))
            
            sample = pd.concat([sample,sample_append],axis=0)
            sample = sample.drop_duplicates(subset=['trade_date','symbol'])
            sample.index = range(len(sample))
            
            sample_original[code] = sample
    
    date_max = today
            
    pkl_file = open(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\ts_RefDaily_%d.pkl' % date_max, 'wb')
    pickle.dump(sample_original,pkl_file)
    pkl_file.close()
    
    if update_hdf:
        p = sample_original.copy()
        today = ddt.today().year * 10000 + ddt.today().month * 100 + ddt.today().day
        hdf = pd.HDFStore(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\ts_RefDaily_%d.h5' % today,
                          mode='w',format='table',complevel=9, complib='blosc')
        for key,value in p.items():
            hdf.put(key=key,value=value,format='table',columns=True)   
        hdf.close()

def update_RefDaily_pkl():
    api = DataApi(addr='tcp://data.quantos.org:8910')
    api.login('18011342046','eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjA5MDQxNzY1NDciLCJpc3MiOiJhdXRoMCIsImlkIjoiMTgwMTEzNDIwNDYifQ.cvUCM4y_btI8W_zbUHPbzSS_vxqMjcZR22lESjb5eQY')
    data_config = jutil.read_json(DATA_CONFIG_PATH)
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    codeList, msg = api.query(
                view="jz.instrumentInfo", #这是基础信息表
                fields="status,list_date, fullname_en, market", 
                filter="inst_type=1&status=1&symbol=", 
                data_format='pandas')
    codeList = codeList[codeList['symbol'].apply(lambda x : (x[-2:]=='SZ') | (x[-2:]=='SH'))]['symbol']
    codeList = list(set(codeList))
#    codeStr = ','.join(codeList)
    
    myFolder =r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv'
    fileList = glob(os.path.join(myFolder,'*.pkl'))
    fileList = list((x[-13:-4] for x in fileList))
    diffList = list([x for x in codeList if x not in fileList])
    
    today = ddt.today().year * 10000 + ddt.today().month * 100 + ddt.today().day
    original_date = 20100101
    
    if len(diffList) > 0:
    
        for code in diffList:
            sample = ds.query_lb_dailyindicator(code,start_date=original_date,end_date=today,
                     fields='trade_date,total_mv,float_mv,pe,pb,turnover_ratio,free_turnover_ratio')[0]
            try:
                assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
            except AssertionError:
                print('%s is of shiness in ref_daily data' % code)
                continue
            sample = sample[['trade_date','symbol','total_mv','float_mv','pe','pb',
                             'turnover_ratio','free_turnover_ratio']]
            sample['trade_date'] = sample['trade_date'].apply(lambda x : int(x))
        
            pkl_file = open(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\mv_%s.pkl' % code, 'wb')
            pickle.dump(sample,pkl_file)
            pkl_file.close()

    for code in fileList:
        try:
            pkl_file = open(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\mv_%s.pkl' % code, 'rb')
            sample = pickle.load(pkl_file)
            pkl_file.close()
        except:
            print('For code in fileList, %s cannot be abstracted from .pkl format' % code)
            pkl_file.close()
            continue
        
        if (sample is None) | (len(sample) <= 1):
            sample = ds.query_lb_dailyindicator(code,start_date=original_date,end_date=today,
                     fields='trade_date,total_mv,float_mv,pe,pb,turnover_ratio,free_turnover_ratio')[0]
            try:
                assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
            except AssertionError:
                print('%s is of shiness in ref_daily data' % code)
                continue
            sample = sample[['trade_date','symbol','total_mv','float_mv','pe','pb',
                             'turnover_ratio','free_turnover_ratio']]       
            sample['trade_date'] = sample['trade_date'].apply(lambda x : int(x))
            
            pkl_file = open(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\mv_%s.pkl' % code, 'wb')
            pickle.dump(sample,pkl_file)
            pkl_file.close()
            print('%s has something wrong but now it is settled!' % code)
        
        else:
            try:
                pkl_file = open(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\mv_%s.pkl' % code, 'rb')
                sample = pickle.load(pkl_file)
                pkl_file.close()
            except:
                print('For code in fileList, %s cannot be abstracted from .pkl format' % code)
                pkl_file.close()
                continue 
            
            date_max = max(sample['trade_date'])
            start_date = date_max
            end_date = today
            sample_append = ds.query_lb_dailyindicator(code,start_date=start_date,end_date=end_date,
                     fields='trade_date,total_mv,float_mv,pe,pb,turnover_ratio,free_turnover_ratio')[0]
            
            try:
                assert ((isinstance(sample_append,pd.DataFrame)) & (len(sample_append) > 0))
            except AssertionError:
                print('%s is of shiness in ref_daily data' % code)
                continue
            sample_append = sample_append[['trade_date','symbol','total_mv','float_mv','pe','pb',
                             'turnover_ratio','free_turnover_ratio']]
            sample_append['trade_date'] = sample_append['trade_date'].apply(lambda x : int(x))
                        
            sample = pd.concat([sample,sample_append],axis=0)
            sample = sample.drop_duplicates(subset=['trade_date','symbol'])
            sample.index = range(len(sample))
            
            pkl_file = open(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\mv_%s.pkl' % code, 'wb')
            pickle.dump(sample,pkl_file)
            pkl_file.close()

def select_RefDaily_pkl(code=None,start_date=20170101,end_date=20171231):
    if not isinstance(start_date,int):
        print('start_date and end_date must be int type')
        return False
    if not isinstance(end_date,int):
        print('start_date and end_date must be int type')
        return False
    
    if isinstance(code,list):
        df = pd.DataFrame([])
        for c in code:
            try:
                pkl_file = open(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\mv_%s.pkl' % c, 'rb')
                sample = pickle.load(pkl_file)
                pkl_file.close()
                sample['trade_date'] = sample['trade_date'].apply(lambda x: int(x))
                sample = sample[(sample['trade_date'] >= start_date) & (sample['trade_date'] <= end_date)]
            except:
                pkl_file.close()
                print('%s cannot be abstracted from .pkl format for RefDaily data!' % c)
                continue
            df = pd.concat([df,sample],axis = 0)
        return df
    
    elif isinstance(code,str) :
        try:
            pkl_file = open(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\mv_%s.pkl' % code, 'rb')
            df = pickle.load(pkl_file)
            pkl_file.close()
            df['trade_date'] = df['trade_date'].apply(lambda x: int(x))
            df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
            return df
        except:
            pkl_file.close()
            print('%s cannot be abstracted from .pkl format for RefDaily data!' % code)
            return False
        
    elif code is None:
        myFolder =r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv'
        fileList = glob(os.path.join(myFolder,'*.pkl'))
        fileList = list((x[-13:-4] for x in fileList))
        df = pd.DataFrame([])
        for code in fileList:
            try:
                pkl_file = open(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\mv_%s.pkl' % code, 'rb')
                sample = pickle.load(pkl_file)
                sample['trade_date'] = sample['trade_date'].apply(lambda x: int(x))
                sample = sample[(sample['trade_date'] >= start_date) & (sample['trade_date'] <= end_date)]
                pkl_file.close()
            except:
                pkl_file.close()
                print('%s cannot be abstracted from .pkl format for RefDaily data!' % code)
                continue
            df = pd.concat([df,sample],axis = 0)
        return df
    
    else:
        print('Something wrong about the param of code you input for RefDaily data!')
        return False

def select_RefDaily_hdf(code=None,start_date=20170101,end_date=20171231):
    if not isinstance(start_date,int):
        print('start_date and end_date must be int type')
        return False
    if not isinstance(end_date,int):
        print('start_date and end_date must be int type')
        return False
    
    myFolder =r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv'
    try:
        fileList = glob(os.path.join(myFolder,'*.h5'))
        fileList.sort()
        date_max = int(fileList[-1][-11:-3])
        
    except IndexError:
        print('There is no *.h5 file in the folder for RefDaily data! Initialize it first!')
        return False
    
    if isinstance(code,list):
        df = pd.DataFrame([])
        hdf = pd.HDFStore(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\ts_RefDaily_%d.h5' % date_max,'r')
        for c in code:
            try:
                sample = getattr(hdf,c)
                sample['trade_date'] = sample['trade_date'].apply(lambda x: int(x))
            except AttributeError:
                print('object has no attribute %s' % c)
                continue
            df = pd.concat([df,sample],axis=0)
        hdf.close()
        return df
        
    elif isinstance(code,str):
        hdf = pd.HDFStore(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\ts_RefDaily_%d.h5' % date_max,'r')
        try:
            df = getattr(hdf,code)
            df['trade_date'] = df['trade_date'].apply(lambda x: int(x))
            hdf.close()
        except AttributeError:
            print('object has no attribute %s' % c)
            hdf.close()
            return False
        return df        
    
    elif code is None:
        df = pd.DataFrame([])
        hdf = pd.HDFStore(r'G:\MultiFactor\Data\TimesSeriesRefDaily_mv\ts_RefDaily__%d.h5' % date_max,'r')
        for key in hdf.keys():
            try:
                sample = getattr(hdf,key)
                sample['trade_date'] = sample['trade_date'].apply(lambda x: int(x))
            except AttributeError:
                print('object has no attribute %s' % c)
                continue
            df = pd.concat([df,sample],axis=0)
        return df
    
    else:
        print('Something wrong about the param of code you input!')
        return False    


def merge_mv(price,mv):
    if not isinstance(price,pd.DataFrame):
        print('Only DataFrame should be input!')
        return
    if not isinstance(mv,pd.DataFrame):
        print('Only DataFrame should be input!')
        return
    price_mv = pd.merge(left=price,right=mv,how='left',left_on=['trade_date','symbol'],
                        right_on = ['trade_date','symbol'])
    '''
    -----------------------------------------------------------------------
    the DataFrame mv incorporates fields such as :
        'total_mv','float_mv','pe','pb','turnover_ratio','free_turnover_ratio'
    could use pd.merge(left,right=mv[['pe','pb']], ...) to merge specific columns
    -----------------------------------------------------------------------
    ''' 
    return price_mv


#
download_ref_daily_pkl(start_date = 20100101, end_date = 20180228)
download_ref_daily_hdf(start_date = 20100101, end_date = 20180228)
#update_RefDaily_dict_hdf(update_hdf = True)
#update_RefDaily_pkl()
#select_RefDaily_pkl(code='000001.SZ',start_date=20170101,end_date=20171231)
'''
----------------------------------------------------------------------
----------------------------------------------------------------------
===============以上都是市值数据，以下的数据有三表数据===================
----------------------------------------------------------------------
----------------------------------------------------------------------''' 


'''
balance_sheet = ["monetary_cap", "tradable_assets", "notes_rcv", "acct_rcv", "other_rcv", "pre_pay",
             "dvd_rcv", "int_rcv", "inventories", "consumptive_assets", "deferred_exp",
             "noncur_assets_due_1y", "settle_rsrv", "loans_to_banks", "prem_rcv", "rcv_from_reinsurer",
             "rcv_from_ceded_insur_cont_rsrv", "red_monetary_cap_for_sale", "other_cur_assets",
             "tot_cur_assets", "fin_assets_avail_for_sale", "held_to_mty_invest", "long_term_eqy_invest",
             "invest_real_estate", "time_deposits", "other_assets", "long_term_rec", "fix_assets",
             "const_in_prog", "proj_matl", "fix_assets_disp", "productive_bio_assets",
             "oil_and_natural_gas_assets", "intang_assets", "r_and_d_costs", "goodwill",
             "long_term_deferred_exp", "deferred_tax_assets", "loans_and_adv_granted",
             "oth_non_cur_assets", "tot_non_cur_assets", "cash_deposits_central_bank",
             "asset_dep_oth_banks_fin_inst", "precious_metals", "derivative_fin_assets",
             "agency_bus_assets", "subr_rec", "rcv_ceded_unearned_prem_rsrv", "rcv_ceded_claim_rsrv",
             "rcv_ceded_life_insur_rsrv", "rcv_ceded_lt_health_insur_rsrv", "mrgn_paid",
             "insured_pledge_loan", "cap_mrgn_paid", "independent_acct_assets", "clients_cap_deposit",
             "clients_rsrv_settle", "incl_seat_fees_exchange", "rcv_invest", "tot_assets", "st_borrow",
             "borrow_central_bank", "deposit_received_ib_deposits", "loans_oth_banks", "tradable_fin_liab",
             "notes_payable", "acct_payable", "adv_from_cust", "fund_sales_fin_assets_rp",
             "handling_charges_comm_payable", "empl_ben_payable", "taxes_surcharges_payable", "int_payable",
             "dvd_payable", "other_payable", "acc_exp", "deferred_inc", "st_bonds_payable", "payable_to_reinsurer",
             "rsrv_insur_cont", "acting_trading_sec", "acting_uw_sec", "non_cur_liab_due_within_1y", "other_cur_liab",
             "tot_cur_liab", "lt_borrow", "bonds_payable", "lt_payable", "specific_item_payable", "provisions",
             "deferred_tax_liab", "deferred_inc_non_cur_liab", "other_non_cur_liab", "tot_non_cur_liab",
             "liab_dep_other_banks_inst", "derivative_fin_liab", "cust_bank_dep", "agency_bus_liab", "other_liab",
             "prem_received_adv", "deposit_received", "insured_deposit_invest", "unearned_prem_rsrv", "out_loss_rsrv",
             "life_insur_rsrv", "lt_health_insur_v", "independent_acct_liab", "incl_pledge_loan", "claims_payable",
             "dvd_payable_insured", "total_liab", "capital_stk", "capital_reser", "special_rsrv", "surplus_rsrv",
             "undistributed_profit", "less_tsy_stk", "prov_nom_risks", "cnvd_diff_foreign_curr_stat",
             "unconfirmed_invest_loss", "minority_int", "tot_shrhldr_eqy_excl_min_int", "tot_shrhldr_eqy_incl_min_int",
             "tot_liab_shrhldr_eqy", "spe_cur_assets_diff", "tot_cur_assets_diff", "spe_non_cur_assets_diff",
             "tot_non_cur_assets_diff", "spe_bal_assets_diff", "tot_bal_assets_diff", "spe_cur_liab_diff",
             "tot_cur_liab_diff", "spe_non_cur_liab_diff", "tot_non_cur_liab_diff", "spe_bal_liab_diff",
             "tot_bal_liab_diff", "spe_bal_shrhldr_eqy_diff", "tot_bal_shrhldr_eqy_diff", "spe_bal_liab_eqy_diff",
             "tot_bal_liab_eqy_diff", "lt_payroll_payable", "other_comp_income", "other_equity_tools",
             "other_equity_tools_p_shr", "lending_funds", "accounts_receivable", "st_financing_payable", "payables"]

balance_field = str()
for x in balance_sheet:
    balance_field = balance_field + str(',') + str(x)
balance_field = balance_field[1:]

sample = ds.query_lb_fin_stat(type_='balance_sheet',symbol='000001.SZ',
                         start_date=20100101,end_date=20171231,fields=balance_field)[0]

这样可以提取要的东西

1）列名的含义是一个问题
2）很多值是空值也是一个问题
3）ann_date 和 report_date 的条件也是一个问题

讲道理，三表的数据可以本地化，但是没有必要实现更新，因为每次重新下载一次就好了，反正都有可能更改。。。
数据不多，下起来也快！

'''

def download_bal_sheet_pkl():
    api = DataApi(addr='tcp://data.quantos.org:8910')
    api.login('18011342046','eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MjA5MDQxNzY1NDciLCJpc3MiOiJhdXRoMCIsImlkIjoiMTgwMTEzNDIwNDYifQ.cvUCM4y_btI8W_zbUHPbzSS_vxqMjcZR22lESjb5eQY')
    data_config = jutil.read_json(DATA_CONFIG_PATH)
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    balance_sheet = ["monetary_cap", "tradable_assets", "notes_rcv", "acct_rcv", "other_rcv", "pre_pay",
             "dvd_rcv", "int_rcv", "inventories", "consumptive_assets", "deferred_exp",
             "noncur_assets_due_1y", "settle_rsrv", "loans_to_banks", "prem_rcv", "rcv_from_reinsurer",
             "rcv_from_ceded_insur_cont_rsrv", "red_monetary_cap_for_sale", "other_cur_assets",
             "tot_cur_assets", "fin_assets_avail_for_sale", "held_to_mty_invest", "long_term_eqy_invest",
             "invest_real_estate", "time_deposits", "other_assets", "long_term_rec", "fix_assets",
             "const_in_prog", "proj_matl", "fix_assets_disp", "productive_bio_assets",
             "oil_and_natural_gas_assets", "intang_assets", "r_and_d_costs", "goodwill",
             "long_term_deferred_exp", "deferred_tax_assets", "loans_and_adv_granted",
             "oth_non_cur_assets", "tot_non_cur_assets", "cash_deposits_central_bank",
             "asset_dep_oth_banks_fin_inst", "precious_metals", "derivative_fin_assets",
             "agency_bus_assets", "subr_rec", "rcv_ceded_unearned_prem_rsrv", "rcv_ceded_claim_rsrv",
             "rcv_ceded_life_insur_rsrv", "rcv_ceded_lt_health_insur_rsrv", "mrgn_paid",
             "insured_pledge_loan", "cap_mrgn_paid", "independent_acct_assets", "clients_cap_deposit",
             "clients_rsrv_settle", "incl_seat_fees_exchange", "rcv_invest", "tot_assets", "st_borrow",
             "borrow_central_bank", "deposit_received_ib_deposits", "loans_oth_banks", "tradable_fin_liab",
             "notes_payable", "acct_payable", "adv_from_cust", "fund_sales_fin_assets_rp",
             "handling_charges_comm_payable", "empl_ben_payable", "taxes_surcharges_payable", "int_payable",
             "dvd_payable", "other_payable", "acc_exp", "deferred_inc", "st_bonds_payable", "payable_to_reinsurer",
             "rsrv_insur_cont", "acting_trading_sec", "acting_uw_sec", "non_cur_liab_due_within_1y", "other_cur_liab",
             "tot_cur_liab", "lt_borrow", "bonds_payable", "lt_payable", "specific_item_payable", "provisions",
             "deferred_tax_liab", "deferred_inc_non_cur_liab", "other_non_cur_liab", "tot_non_cur_liab",
             "liab_dep_other_banks_inst", "derivative_fin_liab", "cust_bank_dep", "agency_bus_liab", "other_liab",
             "prem_received_adv", "deposit_received", "insured_deposit_invest", "unearned_prem_rsrv", "out_loss_rsrv",
             "life_insur_rsrv", "lt_health_insur_v", "independent_acct_liab", "incl_pledge_loan", "claims_payable",
             "dvd_payable_insured", "total_liab", "capital_stk", "capital_reser", "special_rsrv", "surplus_rsrv",
             "undistributed_profit", "less_tsy_stk", "prov_nom_risks", "cnvd_diff_foreign_curr_stat",
             "unconfirmed_invest_loss", "minority_int", "tot_shrhldr_eqy_excl_min_int", "tot_shrhldr_eqy_incl_min_int",
             "tot_liab_shrhldr_eqy", "spe_cur_assets_diff", "tot_cur_assets_diff", "spe_non_cur_assets_diff",
             "tot_non_cur_assets_diff", "spe_bal_assets_diff", "tot_bal_assets_diff", "spe_cur_liab_diff",
             "tot_cur_liab_diff", "spe_non_cur_liab_diff", "tot_non_cur_liab_diff", "spe_bal_liab_diff",
             "tot_bal_liab_diff", "spe_bal_shrhldr_eqy_diff", "tot_bal_shrhldr_eqy_diff", "spe_bal_liab_eqy_diff",
             "tot_bal_liab_eqy_diff", "lt_payroll_payable", "other_comp_income", "other_equity_tools",
             "other_equity_tools_p_shr", "lending_funds", "accounts_receivable", "st_financing_payable", "payables"]

    balance_field = str()
    for x in balance_sheet:
        balance_field = balance_field + str(',') + str(x)
    balance_field = balance_field[1:]

    codeList, msg = api.query(
                view="jz.instrumentInfo", #这是基础信息表
                fields="status,list_date, fullname_en, market", 
                filter="inst_type=1&status=1&symbol=", 
                data_format='pandas')
    codeList = codeList[codeList['symbol'].apply(lambda x : (x[-2:]=='SZ') | (x[-2:]=='SH'))]['symbol']
    codeList = list(set(codeList))

    start_date = 20100101
    today = ddt.today().year * 10000 + ddt.today().month * 100 + ddt.today().day
    end_date = today
    
    for code in codeList:
        sample = ds.query_lb_fin_stat(type_='balance_sheet',symbol=code,
                         start_date=start_date,end_date=end_date,fields=balance_field)[0]
        try:
            assert ((isinstance(sample,pd.DataFrame)) & (len(sample) > 0))
        except AssertionError:
            print('%s is of shiness for balance_sheet data' % code)
            continue
        
        pkl_file = open(r'G:\MultiFactor\Data\Balance_sheet\bal_sheet_%s.pkl' % code, 'wb')
        pickle.dump(sample,pkl_file)
        pkl_file.close()    

def select_bal_sheet_pkl(code=None,start_date=20170101,end_date=20171231):
    '''
    三表里面的 start_date ， end_date 都是宣告日，在报告期之后
    '''
    if not isinstance(start_date,int):
        print('start_date and end_date must be int type')
        return False
    if not isinstance(end_date,int):
        print('start_date and end_date must be int type')
        return False
    
    if isinstance(code,list):
        df = pd.DataFrame([])
        for c in code:
            pkl_file = open(r'G:\MultiFactor\Data\Balance_sheet\bal_sheet_%s.pkl' % c, 'rb')
            try:
                sample = pickle.load(pkl_file)
                pkl_file.close()
                sample['ann_date'] = sample['ann_date'].apply(lambda x: int(x))
                sample = sample[(sample['ann_date'] >= start_date) & (sample['ann_date'] <= end_date)]
            except:
                pkl_file.close()
                print('%s cannot be abstracted from .pkl format for bal_sheet data!' % c)
                continue
            df = pd.concat([df,sample],axis = 0)
        return df
    
    elif isinstance(code,str):
        pkl_file = open(r'G:\MultiFactor\Data\Balance_sheet\bal_sheet_%s.pkl' % code, 'rb')
        try:
            df = pickle.load(pkl_file)
            pkl_file.close()
            df['ann_date'] = df['ann_date'].apply(lambda x: int(x))
            df = df[(df['ann_date'] >= start_date) & (df['ann_date'] <= end_date)]
            return df
        except:
            pkl_file.close()
            print('%s cannot be abstracted from .pkl format for ann_date data!' % code)
            return False
        
    elif code is None:
        myFolder =r'G:\MultiFactor\Data\Balance_sheet'
        fileList = glob(os.path.join(myFolder,'*.pkl'))
        fileList = list((x[-13:-4] for x in fileList))
        df = pd.DataFrame([])
        for code in fileList:
            pkl_file = open(r'G:\MultiFactor\Data\Balance_sheet\bal_sheet_%s.pkl' % code, 'rb')
            try:
                sample = pickle.load(pkl_file)
                sample['ann_date'] = sample['ann_date'].apply(lambda x: int(x))
                sample = sample[(sample['ann_date'] >= start_date) & (sample['ann_date'] <= end_date)]
                pkl_file.close()
            except:
                pkl_file.close()
                print('%s cannot be abstracted from .pkl format for bal_sheet data!' % code)
                continue
            df = pd.concat([df,sample],axis = 0)
        return df
    
    else:
        print('Something wrong about the param of code you input for bal_sheet data!')
        return False

download_bal_sheet_pkl()
#select_bal_sheet_pkl(code=None,start_date=20170101,end_date=20171231)
