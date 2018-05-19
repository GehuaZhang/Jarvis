import pandas as pd
import pickle as pkl
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import svm



class SVM_train:
    def __init__(self):
        self.dict_raw = self.load_batch_expos("../Data/DataExpos")
        self.df_return = self.load_pnl("../Data")

    def load_batch_expos(self, directory=""):
        dict_data = {}
        files = os.listdir(directory)
        for file in files:
            if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                pkl_file = open(directory+'/'+file, 'rb')
                dict_data[file] = self.reset_index(pkl.load(pkl_file), "trade_date")
                pkl_file.close()
        return dict_data

    def load_pnl(self, directory=""):
        pkl_file = open(directory + '/dailyReturnMatrix_21_10_10.pkl', 'rb')
        df_return = self.reset_index(pkl.load(pkl_file), "trade_date")
        pkl_file.close()
        return df_return

    def reset_index(self, df_temp, index_name):  # Reset index and remove it from previous dataframe
        df_temp.index = df_temp[index_name].apply(lambda x: str(x)).tolist()
        df_temp = df_temp.drop(labels=index_name, axis=1)
        return df_temp

    def general_initialize(self):
        dict_data = {}
        column_temp = (list(self.dict_raw.values())[0]).columns

        for factor_name, df_raw in self.dict_raw.items():
            df_init = self.expos_initialize(df_expos=df_raw)
            dict_data[factor_name] = df_init
            column_temp = column_temp.intersection(df_init.columns)   #求列的交集

        num_feature = len(dict_data.keys())
        num_timespan = len(list(self.dict_raw.values())[0].index)
        num_sample = len(column_temp)
        arr_feature = np.zeros(shape=[num_timespan*num_sample, num_feature])


        # 统一特征值
        for i in range(len(dict_data.values())):
            df_temp = list(dict_data.values())[i]
            list_flat = df_temp[column_temp].T.values.flatten()
            for j in range(len(list_flat)):
                arr_feature[j][i] = list_flat[j]

        df_label = self.label_initialize(list_column=column_temp, percentage=30)
        arr_label = df_label.T.values.flatten()

        return arr_feature, arr_label

    def expos_initialize(self, df_expos):
        df_expos = df_expos.dropna(axis=1, how='any')
        expos_size = df_expos.shape

        # 去极值
        expos_dev_sup, expos_dev_inf = df_expos.median() + 5 * (abs(df_expos - df_expos.median())).median(), df_expos.median() - 5 * (abs(df_expos - df_expos.median())).median()  # dev = median(|x-median(x)|)
        df_expos[df_expos > expos_dev_sup] = pd.DataFrame([expos_dev_sup.tolist()] * expos_size[0], index=df_expos.index, columns=df_expos.columns)
        df_expos[df_expos < expos_dev_inf] = pd.DataFrame([expos_dev_inf.tolist()] * expos_size[0], index=df_expos.index, columns=df_expos.columns)
        df_expos = ((df_expos - df_expos.mean()) / df_expos.std(ddof=1)).dropna(axis=1, how='any')

        # 正态化
        df_expos = df_expos.apply(lambda x: ((x-np.mean(x))/np.std(x)))

        return df_expos

    def label_initialize(self, list_column, percentage=30):

        if percentage > 50:
            print("Percentage cannot be greater than 50")  # 因为对两端进行处理 每端占比不能超过一半
            return

        # 收益df与暴露度df统一；然后向上平移一格(下期收益率)；然后把最后一行的NaN替换为0
        df_pnl = self.df_return[list_column].shift(-1).fillna(0)

        # 需要对两端值进行初始化

        # 计算极端值的排名位置
        extremum_inf = np.floor(len(df_pnl.index)*percentage/100)
        extremum_sup = len(df_pnl.index) - extremum_inf

        df_rank = df_pnl.apply(lambda x: x.rank())
        df_rank[df_rank <= extremum_inf] = -1
        df_rank[df_rank >= extremum_sup] = 1
        df_rank[(df_rank != -1) & (df_rank != 1)] = 0

        return df_rank

    def split_data(self, percentage = 90):
        arr_feature, arr_label = self.general_initialize()
        test_percentage = 1-percentage/100
        train_feature, test_feature, train_label, test_label = train_test_split(arr_feature, arr_label, test_size=test_percentage, random_state=0)
        return train_feature, test_feature, train_label, test_label

    def svm_train(self):
        train_feature, test_feature, train_label, test_label = self.split_data()

        #  decision_function_shape "One VS One" "One VS Rest"
        #  cache_size 内存调用大小 1000是指1000mb, 默认200
        #  更多参数说明 http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        clf = svm.SVC(C=0.8, kernel='rbf', gamma="auto", decision_function_shape='ovr', cache_size=5000)
        clf.fit(train_feature, train_label)
        score = clf.score(test_feature, test_label)
        print("Prediction Score: " + str(score))
        self.save_model(model=clf)
        print("Successfully Saved! ")
        return


    #  储存模型
    #  命名方式 SVM2018051523.pkl: 2018年5月15日23点 没保存分钟数
    def save_model(self, model, file_name="SVM", directory="../SVM/SVM_model/"):
        local_time = time.strftime('%Y%m%d%H', time.localtime(time.time()))
        file_path = directory+file_name+local_time+".pkl"
        joblib.dump(model, file_path)
        print("Model Saved to: "+directory)
        print("Name: "+file_path)
        return

    #  导入上次模型
    def load_model(self, model_name, directory="../SVM/SVM_model/"):
        clf = joblib.load(model_name)
        return clf

    #  通过导入的模型进行分类
    def svm_load(self):
        return

pd.options.mode.chained_assignment = None
a = SVM_train()
b = a.svm_train()
print(b)