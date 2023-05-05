try:
    import ray
except:
    pass
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from platform import platform

def normalize_Y(img):
    vmin = np.min(img)
    vmax = np.max(img)
    return (img - vmin) / (vmax - vmin)


def cal_aHash_Hanming(x, tmpY, freq, image_method=1):
    # x = tmp_trainx.iloc[:,0]
    # y = tmp_trainy
    # freq = 6
    if image_method == 1:
        image_ = GramianAngularField(image_size=freq, method='summation')
    elif image_method == 2:
        image_ = GramianAngularField(image_size=freq, method='difference')
    elif image_method == 3:
        image_ = MarkovTransitionField(image_size=freq, n_bins=5)
    elif image_method == 4:
        image_ = RecurrencePlot()
    image_X_fit = image_.fit_transform(np.transpose(np.array(x).reshape(-1, 1)))
    # image_Y_fit = image_.fit_transform(np.transpose(np.array(y).reshape(-1, 1)))
    tmpX = image_X_fit[0]
    # tmpY = image_Y_fit[0]
    tmpX_m = tmpX - np.mean(tmpX)
    tmpY_m = tmpY - np.mean(tmpY)
    dotXY = tmpX_m * tmpY_m
    return sum(sum(dotXY > 0)) / dotXY.shape[0] / dotXY.shape[1]


import cv2


def cal_pHash_Hanming(x, tmpY, freq, image_method=1):
    if image_method == 1:
        image_ = GramianAngularField(image_size=freq, method='summation')
    elif image_method == 2:
        image_ = GramianAngularField(image_size=freq, method='difference')
    elif image_method == 3:
        image_ = MarkovTransitionField(image_size=freq, n_bins=5)
    elif image_method == 4:
        image_ = RecurrencePlot()
    image_X_fit = image_.fit_transform(np.transpose(np.array(x).reshape(-1, 1)))
    # image_Y_fit = image_.fit_transform(np.transpose(np.array(y).reshape(-1, 1)))
    tmpX = cv2.dct(image_X_fit[0])[:16, :16]
    tmpY = cv2.dct(tmpY)[:16, :16]
    # tmpY = dct2(image_Y_fit[0])[:16,:16]
    tmpX_m = tmpX - np.mean(tmpX)
    tmpY_m = tmpY - np.mean(tmpY)
    dotXY = tmpX_m * tmpY_m
    return sum(sum(dotXY > 0)) / dotXY.shape[0] / dotXY.shape[1]


##dHash
def cal_dHash_Hanming(x, tmpY, freq, image_method=1):
    # x = tmp_trainx0.iloc[:,0]
    # freq = pic_len
    if image_method == 1:
        image_ = GramianAngularField(image_size=freq, method='summation')
    elif image_method == 2:
        image_ = GramianAngularField(image_size=freq, method='difference')
    elif image_method == 3:
        image_ = MarkovTransitionField(image_size=freq, n_bins=5)
    elif image_method == 4:
        image_ = RecurrencePlot()
    image_X_fit = image_.fit_transform(np.transpose(np.array(x).reshape(-1, 1)))
    tmpX = image_X_fit[0][1:, :] - image_X_fit[0][:-1, :]
    tmpY = tmpY[1:, :] - tmpY[:-1, :]
    tmpX_m = tmpX - np.mean(tmpX)
    tmpY_m = tmpY - np.mean(tmpY)
    dotXY = tmpX_m * tmpY_m
    return sum(sum(dotXY > 0)) / dotXY.shape[0] / dotXY.shape[1]


def cal_HMD(tmp_trainx, tmp_trainy, tmp_testx, pic_len, image_method, X_num, Hash_method, X_start=0):
    pic_len = pic_len[0]
    if len(tmp_trainy) - pic_len <= 10:
        image_num = len(tmp_trainy) - pic_len
    else:
        image_num = 10
    image_method = image_method[0]
    X_num = X_num[0]
    Hash_method = Hash_method[0]
    # HMD = np.zeros((len(tmp_trainy) - pic_len,tmp_trainx.shape[1]))
    HMD = np.zeros((image_num, tmp_trainx.shape[1]))
    tmp_trainy = normalize_Y(tmp_trainy)
    tmp_testx = pd.DataFrame(tmp_testx, columns=tmp_trainx.columns)
    num = 0
    # for pp in range(0,len(tmp_trainy) - pic_len,image_step):
    for pp in np.linspace(0, len(tmp_trainy) - pic_len, num=image_num, dtype=int):
        print("当前运算第{}个汉明距离，一共{}个".format(str(num + 1), str(image_num)))
        if image_method == 1:
            image_ = GramianAngularField(image_size=pic_len, method='summation')
        elif image_method == 2:
            image_ = GramianAngularField(image_size=pic_len, method='difference')
        elif image_method == 3:
            image_ = MarkovTransitionField(image_size=pic_len, n_bins=5)
        elif image_method == 4:
            image_ = RecurrencePlot()
        tmp_trainx0 = tmp_trainx.iloc[pp:pic_len + pp]
        tmp_trainy0 = tmp_trainy[pp:pic_len + pp]
        image_Y_fit = image_.fit_transform(np.transpose(np.array(tmp_trainy0).reshape(-1, 1)))
        tmpY = image_Y_fit[0]
        if Hash_method == 'a':
            HMD[num, :] = tmp_trainx0.apply(cal_aHash_Hanming, args=(tmpY, pic_len, image_method)).values
        elif Hash_method == 'p':
            HMD[num, :] = tmp_trainx0.apply(cal_pHash_Hanming, args=(tmpY, pic_len, image_method)).values
        elif Hash_method == 'd':
            HMD[num, :] = tmp_trainx0.apply(cal_dHash_Hanming, args=(tmpY, pic_len, image_method)).values
        num += 1
    HMD = pd.DataFrame(HMD, columns=tmp_trainx.columns)
    trainx_choose = tmp_trainx[HMD.min(axis=0).sort_values(ascending=False)[X_start:X_start + X_num, ].index]
    testx_choose = tmp_testx[HMD.min(axis=0).sort_values(ascending=False)[X_start:X_start + X_num, ].index]
    return trainx_choose, testx_choose


def DTW_filter(strainx1, train_y1, test_x1, k, t):
    """
    DTW算法, 对strainx1的第一个维度进行dtw计算，
    :param strainx1: 训练x
    :param train_y1: 训练y
    :param test_x1: 测试x
    :param k: 得到k个特征
    :param t:多少期
    :return:
    """

    def dtw_func(y):
        distance, path = fastdtw(x, y, dist=euclidean)
        return distance

    x = np.array(train_y1)
    DTW = np.apply_along_axis(dtw_func, 0, strainx1.values)
    DTWm = pd.DataFrame(DTW.reshape(1, len(DTW)), columns=pd.DataFrame(strainx1).columns)
    DTWaddname = DTWm.T.sort_values(by=0)[int(t):int(k) + int(t)].index
    index = list(DTWaddname)
    print(index)
    # while index.shape[0] <10:
    #     print("k选的太小了！！！！" + str(i))
    #     DTWaddname=DTWm[DTWm<k-0.01*i].dropna(axis=1).columns
    #     index =list(DTWaddname)
    #     i = i + 1
    train_x2 = strainx1[index]
    test_x = pd.DataFrame(test_x1)
    test_x.columns = strainx1.columns
    test_x2 = test_x[index]
    return train_x2, test_x2


# @ray.remote
def dtw_calculate(x, y):
    distance, _ = fastdtw(x, y, dist=euclidean)
    return distance


def dtw_filter_parallel(strainx1, train_y1, test_x1, k, t):
    """
    DTW的并行化版本
    :param strainx1:
    :param train_y1:
    :param test_x1:
    :param k:
    :param t:
    :return:
    """
    x_array = strainx1.values
    feature_len = strainx1.shape[1]
    # 启动ray
    if 'Windows' in platform():
        ray.init('ray://192.168.1.208:31666')
    else:
        ray.init('ray://example-cluster-ray-head:10001')
    # 构造远程函数
    dtw_res = ray.get([dtw_calculate.remote(train_y1, x_array[:, i]) for i in range(feature_len)])
    ray.shutdown()
    dtw_df = pd.DataFrame(data=dtw_res, index=strainx1.columns)
    dtw_name = dtw_df.sort_values(by=0)[t: k + t].index
    indics = list(dtw_name)
    print(indics)


    train_x2 = strainx1[indics]
    test_x = pd.DataFrame(test_x1)
    test_x.columns = strainx1.columns
    test_x2 = test_x[indics]
    return train_x2, test_x2


# 选取相关系数
def column_matrix_corr_matrix(Ytrain_roll, Xtrain_roll, k, bigger=True):
    Ytrain_roll = np.array(Ytrain_roll)
    y_hat = Ytrain_roll - Ytrain_roll.mean()
    x_hat = Xtrain_roll - Xtrain_roll.mean()
    cov_x_y = np.divide(np.dot(y_hat.T, x_hat), Ytrain_roll.shape[0] - 1)
    print(cov_x_y.shape)
    tmp_corr = np.divide(cov_x_y[0], Ytrain_roll.std() * Xtrain_roll.std(axis=0))
    tmp_corr = tmp_corr.sort_values(ascending=False)
    print(len(tmp_corr))
    if bigger:
        xx = tmp_corr > k
    else:
        xx = tmp_corr < k

    print(set(list(xx.index[xx])).intersection(set(list(Xtrain_roll.columns))))
    return Xtrain_roll[list(set(list(xx.index[xx])).intersection(set(list(Xtrain_roll.columns))))]


def del_dupli(Xtrain_roll, k2=0.9, func=column_matrix_corr_matrix):
    Xtrain_roll2 = Xtrain_roll
    Xtrain_roll3 = []
    while 1:
        tmpy0 = pd.DataFrame(Xtrain_roll2.iloc[:, 0])
        Xtrain_roll3.append(tmpy0)
        Xtrain_roll2.drop([Xtrain_roll2.columns[0]], axis=1, inplace=True)
        Xtrain_roll2 = func(tmpy0, Xtrain_roll2, k2, bigger=False)
        if Xtrain_roll2.shape[1] == 1:
            Xtrain_roll3.append(Xtrain_roll2)
            break
        elif Xtrain_roll2.shape[1] == 0:
            break
        print('3-')
        print(Xtrain_roll2.shape)
    return pd.concat(Xtrain_roll3, axis=1)
