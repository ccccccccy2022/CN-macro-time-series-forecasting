# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/06/16 30:27
# @Author  : TJD
# @FileName: feature_engine2.py
import sys
import time
import warnings

import pandas as pd
import numpy as np
import math
import ray

from feature_engineering.model_tools.Multi import column_matrix_corr_matrix, del_dupli


class Filter:
    def __init__(self, strainx, train_y, test_x, n_for, ray_params, nnn=1, t=0):
        '''
        :param filter_name: 特征工程名字，'CORT'，'Pearson'，'HMImage_1_a'/'HMImage_1_p'/'HMImage_1_d'/'HMImage_2_a'/
        'HMImage_2_p'/'HMImage_2_d'/'HMImage_3_a'/ 'HMImage_3_p'/ 'HMImage_3_d'/ 'Image_4_a'/ 'HMImage_4_p'/ 'HMImage_4_d'，'DTWfilter'，'Causalfilter'（ANM)
        :param strainx:特征x，[dataframe]
        :param train_y:特征y,[numpy.array]
        :param test_x:特征x验证集部分，[dataframe]
        :param k:阈值或者特征个数，CORT和Pearson给的是阈值，其余方法控制的是特征的个数
        :param t:再最新的DTWfilter和Causalfilter里可以作为特征得分的截取点,默认t=0，那就是取排名靠前的前k个特征，如果t=n,那么截取的是（n,n+k)的特征
        :param nnn:适配大程序的多期滚动，集群将多期循环工作已做拆解，默认为1期滚动
        :param n_for:预测长度
        '''
        self.strainx = strainx
        self.train_y = train_y
        self.test_x = test_x
        self.t = int(t)
        self.nnn = nnn
        self.n_for = n_for
        self.train_x2 = list(np.zeros((self.nnn, 1)))
        self.test_x2 = list(np.zeros((self.nnn, 1)))
        self.ray_params = ray_params

    def __getdata(self, ynum):
        columns = self.strainx[0].columns.to_list()
        row, col = self.strainx[0].shape[0], self.strainx[0].shape[1]
        ylen = int(row / ynum)
        nfor = int(self.test_x[0].shape[0] / ynum)
        train_x_data, train_y_data, test_x_data = [], [], []
        for i in range(ynum):
            train_x_data.append(self.strainx[0].iloc[i * ylen:(i + 1) * ylen, :])
            train_y_data.append(pd.DataFrame(self.train_y[0][i * ylen:(i + 1) * ylen]))
            test_x_data.append(self.test_x[0].iloc[i * nfor:(i + 1) * nfor, :])
        return train_x_data, train_y_data, test_x_data

    def ray_source_management(self):
        """
        针对并行计算需要启动ray而进行资源管理
        :return:
        """
        pass

    def causal_filter(self, ynum, method='anm', k=50):
        """
        实例化CausalFilter，默认使用anm方法
        :return:
        """
        train_x, train_y, test_x = self.__getdata(ynum)
        train_x = [x.iloc[:, :] for x in train_x]
        test_x = [x.iloc[:, :] for x in test_x]
        train_y = [y[:] for y in train_y]
        filter = CausalFilter(train_x=train_x,
                              train_y=train_y,
                              test_x=test_x,
                              nnn=self.nnn,
                              ray_params=self.ray_params)
        if method == 'anm-p':
            print('启用并行anm')
            rest_features = filter.anm_parallel(k=k)
        elif method == 'anm':
            print('启用anm v1')
            rest_features = filter.anm(k=k)
        elif method == 'anm-1':
            print('启用anm v2')
            rest_features = filter.anm_serial(k=k)
        elif method == 'icalingam':
            # raise ValueError('icalingam还在调试，并未启用')
            rest_features = filter.lingam(mode='ica', max_feature=50)
        elif method == 'directlingam':
            # raise ValueError('directlingam还在调试，并未启用')
            rest_features = filter.lingam(mode='direct', max_feature=50)
        elif method == 'varlingam':
            raise ValueError('varlingam还在调试，并未启用')
            # rest_features = filter.lingam(mode='var')
        elif method == 'sgnn':
            raise ValueError('sgnn还在调试，并未启用')
            # rest_features = filter.sgnn(k=k)
        elif method == 'pc':
            print('启用pc')
            rest_features = filter.pc(k=k)
        elif method == 'hsic':
            print('启用hsiclasso')
            rest_features = filter.hsiclasso(max_feature=k)
        else:
            print('默认启用并行anm方法')
            rest_features = filter.anm_parallel(k=k)
        # 对trainx和testx进行提取
        train_x2 = [df[rest_features] for df in self.strainx]
        test_x2 = [df[rest_features] for df in self.test_x]
        return train_x2, test_x2

    def correlation_filter(self, filter_name, ynums, k):
        """
        实例化CorrelationFilter
        """
        strainx = [x.iloc[:1000, :] for x in self.strainx]
        test_x = [x.iloc[:1000, :] for x in self.test_x]
        train_y = [y[:1000] for y in self.train_y]

        filter = CorrelationFilter(strainx=strainx,
                                   train_y=train_y,
                                   test_x=test_x,
                                   k=k,
                                   n_for=self.n_for,
                                   nnn=self.nnn,
                                   t=self.t,
                                   ray_params=self.ray_params)

        return filter.run(filter_name=filter_name, ynums=ynums)

    def run(self, filter_name, ynums=1, k=10):
        if filter_name[:6] != 'causal':
            # 相关性方法
            train_x2, test_x2 = self.correlation_filter(filter_name, ynums, k)
        elif filter_name[:6] == 'causal':
            # 因果发现方法
            train_x2, test_x2 = self.causal_filter(ynums, method=filter_name[7:], k=k)
        else:
            raise ValueError('输入特征工程未找到：{}'.format(filter_name))
        return train_x2, test_x2


class CorrelationFilter:
    """
    所有相关性过滤方法集合
    """

    def __init__(self, strainx, train_y, test_x, k, ray_params, n_for, nnn=1, t=0):
        self.strainx = strainx
        self.train_y = train_y
        self.test_x = test_x
        self.k = k
        self.t = int(t)
        self.nnn = nnn
        self.n_for = n_for
        self.train_x2 = list(np.zeros((self.nnn, 1)))
        self.test_x2 = list(np.zeros((self.nnn, 1)))
        self.ray_params = ray_params

    def cort(self):
        """
        CORT相关性方法
        :return:
        """

        # cort辅助计算
        def _cort_cal(d2):
            d22 = d2[:-1]
            d2_ = d2[1:]
            d2_d2 = d2_ - d22
            res = np.abs(np.dot(d2_d2.T, d1_d1) / (np.sqrt(np.dot(d2_d2.T, d2_d2)) * np.sqrt(np.dot(d1_d1.T, d1_d1))))
            return res

        # ----------- cort -------------
        for j in range(self.nnn):
            d1 = pd.DataFrame(self.train_y[j]).iloc[:-1, :]
            d1_ = pd.DataFrame(pd.DataFrame(self.train_y[j]).iloc[1:, :].values,
                               index=range(self.train_y[j].shape[0] - 1))
            d1_d1 = d1_ - d1
            index = []
            i = 0
            while np.array(index).shape[0] == 0 and self.k - i * 0.1 > 0:
                print(str(self.k - i * 0.1))
                cort = np.apply_along_axis(_cort_cal, 0, np.array(self.strainx[j]))
                cortm = pd.DataFrame(cort.reshape(1, len(cort.flatten())),
                                     columns=pd.DataFrame(self.strainx[j]).columns)
                cortaddname = cortm[cortm > (self.k - i * 0.1)].dropna(axis=1).columns
                index = index + list(cortaddname)
                i = i + 1
        return index

    def _findcorr(self, Ytrain_roll, Xtrain_roll, k1_1, X_num):
        """

        :param Ytrain_roll:
        :param Xtrain_roll:
        :param k1_1:
        :param X_num:
        :return:
        """
        while 1:
            chooseXmatrix2 = column_matrix_corr_matrix(Ytrain_roll, Xtrain_roll, k1_1)
            k1_1 -= 0.05
            if chooseXmatrix2.shape[1] > X_num:
                break
        return chooseXmatrix2.columns, k1_1

    def pearson(self):
        """
        trainx和testx分别计算与y的相关性，并剔留下两个结果的交集
        :return:
        """
        for j in range(self.nnn):
            k2 = 0.8
            X_num = 30

            # 分别计算trainx，testx和y的相关性筛出符合要求的特征名称并取交集
            chooseX1, k1_1 = self._findcorr(self.train_y[j], self.strainx[j], self.k, X_num)
            tmptestx = pd.DataFrame(self.test_x[j], columns=self.strainx[j].columns)
            chooseX2, k1_2 = self._findcorr(self.train_y[j][-self.n_for:], tmptestx, self.k, X_num)
            XnameAll0 = chooseX1 & chooseX2

            # 在trainx中选择的特征列表长度小于100时，需要补齐到100个以上
            while len(chooseX1) < 100:
                X_num += 100
                chooseX1, k1_1 = self._findcorr(self.train_y[j], self.strainx[j], k1_1, X_num)
                chooseX2, k1_2 = self._findcorr(self.train_y[j].iloc[-self.n_for:, ], self.test_x[j], k1_2, X_num)
                XnameAll0 = chooseX1 & chooseX2

            # 当选择特征列表长度大于1时，剔除重复
            if len(XnameAll0) > 1:
                chooseXmatrix3 = del_dupli(self.strainx[j][XnameAll0], k2)

        return chooseXmatrix3.columns

    def _bivariate_pearson(self, v1, v2):
        """
        两个变量x和y的pearson计算, 输入v1和v2是两个等长的[1, n]列表。
        :return:
        """
        n = len(v1)
        # simple sums
        sum1 = sum(float(v1[i]) for i in range(n))
        sum2 = sum(float(v2[i]) for i in range(n))
        # sum up the squares
        sum1_pow = sum([pow(v, 2.0) for v in v1])
        sum2_pow = sum([pow(v, 2.0) for v in v2])
        # sum up the products
        p_sum = sum([v1[i] * v2[i] for i in range(n)])
        # 分子sum，分母demoninator
        num = p_sum - (sum1 * sum2 / n)
        den = math.sqrt((sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))
        if den == 0:
            return 0.0
        return num / den

    def hmimage(self, filter_name):
        """

        :param filter_name:
        :return:
        """
        if 'step' in filter_name:
            print('image_step')
            pic_len = int(64 / 72 * self.strainx[0].shape[0])
            if pic_len % 2 != 0:
                pic_len += 1
            image_method = int(filter_name[8])
            Hash_method = filter_name[10]
            feature_step = 1
            try:
                feature_step = int(filter_name[-1])
            except:
                pass
            print('feature_step={}'.format(feature_step))
            feature_index = list(set(
                np.linspace(0, np.floor(self.nnn / feature_step) * feature_step, int(np.floor(self.nnn / feature_step)),
                            endpoint=False, dtype=int)))
            if len(feature_index) == 0:
                feature_index = [0]
            print('要计算的期数为{}'.format(feature_index))
            feature_index.sort()
            strainx_cal = []
            train_y_cal = []
            test_x_cal = []
            for ff in feature_index:
                strainx_cal.append(self.strainx[ff])
                train_y_cal.append(self.train_y[ff])
                test_x_cal.append(self.test_x[ff])
            nnn_cal = len(feature_index)
            train_x2_cal = list(map(lambda x: cal_HMD(strainx_cal[x], train_y_cal[x], test_x_cal[x], list([pic_len]),
                                                      list([image_method]), list([int(self.k)]), list([Hash_method]))[
                0],
                                    range(nnn_cal)))
            test_x2_cal = list(map(lambda x: cal_HMD(strainx_cal[x], train_y_cal[x], test_x_cal[x], list([pic_len]),
                                                     list([image_method]), list([int(self.k)]), list([Hash_method]))[1],
                                   range(nnn_cal)))
            train_x2 = []
            test_x2 = []
            for j in range(self.nnn):
                if j in feature_index:
                    train_x2.append(train_x2_cal[feature_index.index(j)])
                    test_x2.append(test_x2_cal[feature_index.index(j)])
                    print("第{}个滚动期，计算特征工程".format(j + 1))
                else:
                    train_x2.append(self.strainx[j][train_x2[-1].columns])
                    tmptestx2 = pd.DataFrame(self.test_x[j], columns=self.strainx[j].columns)
                    test_x2.append(tmptestx2[test_x2[-1].columns])
                    print("第{}个滚动期，不计算特征工程".format(j + 1))
        else:
            print('image')
            pic_len = int(64 / 72 * self.strainx[0].shape[0])
            if pic_len % 2 != 0:
                pic_len += 1
            image_method = int(filter_name[8])
            Hash_method = filter_name[-1]
            from multiprocessing import Pool
            from feature_engineering.model_tools.Multi import cal_HMD
            # zip_args = list(zip(strainx, train_y, test_x, [list([pic_len]) for x in range(nnn)],
            #                     [list([image_method]) for x in range(nnn)], [list([k]) for x in range(nnn)],
            #                     [list([Hash_method]) for x in range(nnn)]))
            # pool = Pool(processes=6)
            # res = pool.starmap(cal_HMD, zip_args)
            # pool.close()
            # pool.join()
            train_x2 = list(map(lambda x:
                                cal_HMD(self.strainx[x], self.train_y[x], self.test_x[x], list([pic_len]),
                                        list([image_method]),
                                        list([self.k]), list([Hash_method]))[0], range(self.nnn)))
            print(train_x2)
            test_x2 = list(map(lambda x:
                               cal_HMD(self.strainx[x], self.train_y[x], self.test_x[x], list([pic_len]),
                                       list([image_method]),
                                       list([self.k]), list([Hash_method]))[1], range(self.nnn)))
        return train_x2, test_x2

    def dtw_filter(self):
        from feature_engineering.model_tools.Multi import dtw_filter_parallel
        start_t = time.time()
        res = list(map(lambda x: dtw_filter_parallel(self.strainx[x], self.train_y[x], self.test_x[x], self.k, self.t,
                                                     ray_params=self.ray_params),
                       range(self.nnn)))
        end_t = time.time()
        print('dtw ray运行耗时：{}'.format(end_t - start_t))

        train_x2 = list(map(lambda x: res[x][0], range(self.nnn)))
        test_x2 = list(map(lambda x: res[x][1], range(self.nnn)))
        return train_x2, test_x2

    def dtw_filter_cpp(self):
        from feature_engineering.model_tools.Multi import dtw_cpp
        res = []
        start_t = time.time()
        for j in range(self.nnn):
            res.append(dtw_cpp(self.strainx[j], self.train_y[j], self.test_x[j], self.k, self.t))
        end_t = time.time()
        print('dtw c++ 运行耗时：{}'.format(end_t - start_t))

        train_x2 = list(map(lambda x: res[x][0], range(self.nnn)))
        test_x2 = list(map(lambda x: res[x][1], range(self.nnn)))
        return train_x2, test_x2

    def run(self, filter_name, ynums=1):
        if filter_name == 'CORT':
            index = self.cort()
            for j in range(self.nnn):
                if np.array(index).shape[0] == 0:
                    print("阈值提取特征失败")
                    self.train_x2[j] = self.strainx[j]
                    self.test_x2[j] = self.test_x[j]
                else:
                    self.train_x2[j] = pd.DataFrame(pd.DataFrame(self.strainx[j])[index])
                    if self.test_x[j].shape[0] != 0:
                        self.test_x2[j] = pd.DataFrame(
                            pd.DataFrame(self.test_x[j], columns=self.strainx[j].columns)[index])
            return self.train_x2, self.test_x2

        elif filter_name == 'Pearson':
            index = self.pearson()
            for j in range(self.nnn):
                if np.array(index).shape[0] == 0:
                    print("阈值提取特征失败")
                    self.train_x2[j] = self.strainx[j]
                    self.test_x2[j] = self.test_x[j]
                else:
                    self.train_x2[j] = pd.DataFrame(pd.DataFrame(self.strainx[j])[index])
                    if self.test_x[j].shape[0] != 0:
                        self.test_x2[j] = pd.DataFrame(
                            pd.DataFrame(self.test_x[j], columns=self.strainx[j].columns)[index])

        elif filter_name[0:7] == "HMImage":
            # raise ValueError('目前hmimage太慢暂不能使用')
            return self.hmimage(filter_name)
        elif filter_name == 'DTWfilter':
            return self.dtw_filter()
        elif filter_name == 'dtw_cpp':
            return self.dtw_filter_cpp()
        else:
            return self.dtw_filter()


class CausalFilter:
    """
    目前可用的因果推断模型方法用于筛选特征
    """

    def __init__(self, train_x, train_y, test_x, ray_params, nnn=1, t=0):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.nnn = nnn
        self.t = int(t)
        self.ray_params = ray_params

    # ---------------------- pairwise based approach ------------------------
    def __data_auto_splite(self, k):
        """
        这个方法目前适用于pairwise的因果方法（anm, gnn），对于高维数据集遍历过慢的问题的优化，进行行列切分
        TODO: 对分块进行并行分组
        :return:
        """
        cut_x_list = []
        for j in range(self.nnn):
            part = math.ceil(self.train_x[j].shape[1] / int(k))
            i = 0
            if part <= self.train_x[j].shape[0] * 1 / 4:
                i = 1
            else:
                while part > self.train_x[j].shape[0] * 1 / 4:
                    i = i + 1
                    part = math.ceil(self.train_x[j].shape[1] / (i * int(k)))

            for cut in range(i * k):
                try:
                    cut_x_list.append(self.train_x[j].iloc[:, part * cut:part * (cut + 1)])
                except:
                    cut_x_list.append(self.train_x[j].iloc[:, part * cut:])
        return cut_x_list, part

    def __pairwise_assist(self, cutx, train_y, graph, model):
        """
        :param cutx:
        :param train_y:
        :param graph:
        :param model:
        :return:
        """
        y_name = train_y.columns.tolist()[-1]
        skeleton = graph.predict(pd.concat([train_y, cutx], axis=1))
        output_graph = model.predict(pd.concat([train_y, cutx], axis=1), skeleton)
        ans1 = pd.DataFrame(list(output_graph.edges(data='weight')), columns=['Cause', 'Effect', 'Score'])
        ans2 = ans1.sort_values(by='Score', ascending=False)
        score = ans2[ans2['Effect'] == y_name].iloc[0, :]
        nameindex = ans2[ans2['Effect'] == y_name]['Cause'].iloc[0]
        return score, nameindex

    def anm(self, k=10):
        """
        用到了对形式模型additive noise model，以及用glasso求data中的逆邻接矩阵。分组求跟y相关的最高anm得分的特征。
        :param k: 切割块的每个块的特征个数，默认10
        :return:
        """
        from cdt.independence.graph import Glasso
        from cdt.causality.pairwise import ANM
        glasso = Glasso()
        anm_model = ANM()

        # 数据集分割处理
        cut_x_list, part = self.__data_auto_splite(k=k)
        # 对每个cut_x_list里面的x分区执行
        for j in range(self.nnn):
            score = pd.DataFrame()
            for cutx in cut_x_list:
                try:
                    score1, _ = self.__pairwise_assist(cutx, self.train_y[j], glasso, anm_model)
                    score = pd.concat([score, score1], axis=1)
                except:
                    pass
            score.columns = score.loc['Cause']
        return list(score.loc['Score'].sort_values(ascending=False).index[int(self.t):int(self.t) + 2])

    def anm_serial(self, k=10):
        from causallearn.search.FCMBased.ANM.ANM import ANM
        anm_model = ANM()

        def _anm_assist(cutx, train_y, anm):
            df = pd.concat([cutx, train_y], axis=1)
            index_list = df.columns[:-1]
            anm_result = pd.DataFrame(columns=['Score'], index=index_list)
            n_row, n_len = df.shape[0], df.shape[1]
            for i in range(n_len - 1):
                # if i==181 or i==122:
                #     continue
                # print(i)
                p_forward, p_backward = anm.cause_or_effect(data_x=df.iloc[:, i].values.reshape(n_row, 1),
                                                            data_y=df.iloc[:, -1].values.reshape(n_row, 1))
                anm_result.loc[index_list[i]] = [p_forward - p_backward]
            return anm_result

        # 数据集分割处理
        cut_x_list, part = self.__data_auto_splite(k=k)
        cut_x_list = [x.iloc[:500, :] for x in cut_x_list]
        train_y = [x.iloc[:500, :] for x in self.train_y]

        # 对每个cut_x_list分区执行
        score = pd.DataFrame()
        ray_tmp = [_anm_assist(cutx, train_y[0], anm_model) for cutx in cut_x_list]
        for item in ray_tmp:
            score = pd.concat([score, item], axis=0)

        return list(score.sort_values(by=['Score'], ascending=False).index[int(self.t):int(self.t) + int(k)])

    def anm_parallel(self, cpus=8, k=10):
        """
        由于pairwise本身的特征，可以将这个过程做并行计算。使用基于python的Ray包。
        :param num_cpus: 调用的cpu个数
        :return:
        """
        from causallearn.search.FCMBased.ANM.ANM import ANM
        anm_model = ANM()

        @ray.remote(num_cpus=0.5)
        def _anm_assist_ray(cutx, train_y, anm, task_id):
            df = pd.concat([cutx, train_y], axis=1)
            index_list = df.columns[:-1]
            anm_result = pd.DataFrame(columns=['Score'], index=index_list)
            n_row, n_len = df.shape[0], df.shape[1]
            for i in range(n_len - 1):
                p_forward, p_backward = anm.cause_or_effect(data_x=df.iloc[:, i].values.reshape(n_row, 1),
                                                            data_y=df.iloc[:, -1].values.reshape(n_row, 1))
                anm_result.loc[index_list[i]] = [p_forward - p_backward]
            print('计算完毕：{}/{}'.format(task_id[0], task_id[1]))
            return anm_result

        # 启动ray
        if self.ray_params['cluster_address'] is None:
            print('未检测到集群地址，使用本地资源并行')
            ray.init(num_cpus=cpus)
        else:
            print('检测到集群地址，使用集群资源并行')
            ray.init(address=self.ray_params['cluster_address'])

        # 数据集分割处理
        cut_x_list, part = self.__data_auto_splite(k=k)
        cut_x_list = [x.iloc[:, :] for x in cut_x_list]
        train_y = [x.iloc[:, :] for x in self.train_y]

        # 对每个cut_x_list分区执行
        score = pd.DataFrame()
        ray_id_store = []
        for i in range(len(cut_x_list)):
            task_id = (i, len(cut_x_list))
            ray_id_store.append(_anm_assist_ray.remote(cut_x_list[i], train_y[0], anm_model, task_id))
        ray_tmp = ray.get(ray_id_store)
        for item in ray_tmp:
            score = pd.concat([score, item], axis=0)

        # 停止ray
        ray.shutdown()

        return list(score.sort_values(by=['Score'], ascending=False).index[int(self.t):int(self.t) + 2])

    def bivariate_fit(self, k=10):
        """
        双变量fit方法。原理为
        :param k:
        :return:
        """
        pass

    def sgnn(self, k=10):
        """
        shallow generative netural networks，一个cgnn的变形方法。
        TODO:sgnn速度太慢
        :param k:选择特征个数，默认10
        :return:
        """
        from cdt.causality.pairwise import GNN
        from cdt.independence.graph import Glasso

        glasso = Glasso()
        gnn_model = GNN()
        # 数据集分割处理
        cut_x_list, part = self.__data_auto_splite(k=k)
        # 对每个cut_x_list里面的x分区执行
        for j in range(self.nnn):
            score = pd.DataFrame()
            nameindex = []
            for cutx in cut_x_list:
                try:
                    score1, nameindex1 = self.__pairwise_assist(cutx, self.train_y[j], glasso, gnn_model)
                    score = pd.concat([score, score1], axis=1)
                    nameindex.append(nameindex1)
                except:
                    pass
            score.columns = score.loc['Cause']
        return list(score.loc['Score'].sort_values(ascending=False).index[int(self.t):int(self.t) + 5])

    # ---------------------- graph based approach -------------------------
    def __lingam_causal_generator(self, model, df):
        """
        辅助进行两步操作：1.计算整个df的特征的相互因果排序；2.查询y的邻接矩阵。
        然后进行特征交集。
        :param model: lingam输入模型
        :param df:
        :return:
        """
        y_index = df.shape[1] - 1
        feature_names = df.columns.tolist()
        causal_order = model.causal_order_
        adj_matrix = model.adjacency_matrix_
        # 找到所有排在y之前的原因x的index1
        temp_order = causal_order[:causal_order.index(y_index) + 1]
        # 从邻接矩阵中找到和y相连的x的index2
        causal_index = np.nonzero(adj_matrix[y_index, :])
        # index1和index2进行交集
        cross_index = list(set(causal_index[0]).intersection(set(temp_order)))
        if len(cross_index) == len(causal_index[0]):
            causal_feature = []
            for i in causal_index[0]:
                causal_feature.append(feature_names[i])
        else:
            raise ValueError('邻接矩阵和因果排序中出现错误')
        return causal_feature

    def lingam(self, mode='ica', max_feature=None):
        """
        一个基于方程的线性非高斯无环模型，使用ica求解，并得到一个因果排序。本方法并不是pairwise方法，
        所以不进行行列切割的方法对数据分块。
        在每个回测期都会计算一次当前期x与y的一个因果情况，并返回记录在字典里。最终字典内对每个滚动期
        每个有因果关系的特征都会做一次计数。根据出现次数排序，并根据num_features来选择最终的特征数。
        TODO：varlingam会出现奇异矩阵报错。
        :param k: 需要的特征数
        :return:
        """
        from causallearn.search.FCMBased import lingam
        causal_count_dict = {}
        for j in range(self.nnn):
            xy_combined = pd.concat([self.train_x[j], self.train_y[j]], axis=1)
            if mode == 'ica':
                print('启用icalingam')
                model = lingam.ICALiNGAM()
            elif mode == 'var':
                raise ValueError('varlingam目前不支持，请使用ica或者direct')
                # model = lingam.VARLiNGAM(prune=True, criterion='bic')
            elif mode == 'direct':
                print('启用direct-lingam')
                model = lingam.DirectLiNGAM()
            else:
                print('启用icalingam')
                model = lingam.ICALiNGAM()

            model.fit(xy_combined)
            causal_feature = self.__lingam_causal_generator(model=model, df=xy_combined)

            for item in causal_feature:
                if item not in causal_count_dict.keys():
                    causal_count_dict[item] = 1
                else:
                    causal_count_dict[item] += 1

        causal_sorted = sorted(causal_count_dict.items(), key=lambda x: x[1], reverse=True)
        if max_feature is None or max_feature == len(causal_sorted):
            return [key for (key, v) in causal_sorted]
        else:
            if max_feature > len(causal_sorted):
                raise ValueError('输出max_feature大于待选x的长度')
            else:
                return [key for i, (key, v) in enumerate(causal_sorted) if i < max_feature]

    def causalgraph_plot(self, g, dpi=200):
        """
        用于各种graph方法的绘图
        :return:
        """
        from causallearn.utils.GraphUtils import GraphUtils
        import io
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        pyd = GraphUtils.to_pydot(g.G, dpi=dpi)
        tmp_png = pyd.create_png(f="png")
        pyd.write_png("result.png")
        fp = io.BytesIO(tmp_png)

    def pc(self, k=None):
        """
        peter-clark方法，基本思想是统计（条件）独立的变量之间没有因果链接，是基于约束的算法的典型代表。
        利用数据中的条件独立关系来发掘潜在的因果结构。
        # TODO: 计算过程中产生奇异矩阵，目前没解决这个问题。
        :param k:
        :return:
        """
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz

        for j in range(self.nnn):
            # xy_combined = pd.concat([self.train_x[j], self.train_y[j]], axis=1)
            # np.linalg.det(self.train_x[j].values)
            cg = pc(data=pd.concat([self.train_x[0], self.train_y[0]], axis=1).values,
                    alpha=0.05,
                    indep_test=fisherz,
                    stable=0,
                    uc_rule=0)

            self.causalgraph_plot(cg, dpi=100)
            print('pc plot is done.')
        return None

    def sam(self, k=None):
        """
        structural agnostic modeling. 对抗性建模方法
        :param k:
        :return:
        """
        import networkx as nx
        from cdt.causality.graph import SAM
        obj = SAM()
        output = obj.predict(pd.concat([self.train_x[0], self.train_y[0]], axis=1))
        nx.to_pandas_adjacency(output).to_excel('./sam_adjacency.xlsx')
        return output

    def hsiclasso(self, max_feature=None):
        """
        graphical lasso
        :param k:
        :return:
        """
        from cdt.independence.graph import HSICLasso
        obj = HSICLasso()
        score = obj.predict_features(self.train_x[0], self.train_y[0])
        tmp = pd.DataFrame(data=np.array(score), index=self.train_x[0].columns, columns=['Score'])
        return list(tmp.sort_values(by=['Score'], ascending=False).index[:max_feature])

    def pc_gpu(self):
        """
        pc方法的gpu版本，需要调用cdt包。
        TODO:cdt调用pc方法时需要同时调用r包，目前有几个R包在install中装不上。
        """
        pass

    def linear_granger(self):
        """
        格兰杰因果
        TODO:待开发。
        :return:
        """
        pass

    # ----------------- continious optimizing method -------------------
    def no_tears(self):
        pass
