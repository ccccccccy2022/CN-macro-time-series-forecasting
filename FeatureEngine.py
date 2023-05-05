# 剔除含有异常值的指标
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso as lasso
from sklearn.model_selection import TimeSeriesSplit
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
from xgboost import XGBRegressor as XGBR
#        import catboost as cb

from scipy.stats import pearsonr
# from sklearn.feature_selection import SelectKBest

from dcor import distance_correlation
from dcor.independence import distance_covariance_test
from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import cross_val_score
from sklearn import datasets, linear_model
from time import strftime, localtime
#######################################   异常值处理   #######################################################
def x_outliers(result_df,sigma):
    r1 = result_df[result_df > np.mean(result_df) + sigma * np.std(result_df)].dropna(axis=1, how='all')
    r2 = result_df[result_df < np.mean(result_df) - sigma * np.std(result_df)].dropna(axis=1, how='all')
    result = pd.concat([result_df[r1.columns], result_df[r2.columns]], axis=1)
    result_df2 = result_df.drop(columns=result.columns)
    return result_df2


##特征处理K_mean
def featureengine(k_mean_flag,min_max_flag,normalization_flag,trainadd_x,strainadd_x,test_x,nnn,k,n_for):
    def k_mean(k, data):
        from sklearn.cluster import KMeans
        print('开始k-means聚类')
        model = KMeans(n_clusters=k, n_jobs=4, max_iter=500)  # 分为k类，并发数4
        model.fit(data.T)
        r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
        r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
        r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
        r.columns = list(data.index) + [u'类别数目']  # 重命名表头
        dataT = pd.DataFrame(data.T.values, index=data.columns)
        r3 = pd.concat([dataT, pd.Series(model.labels_, index=data.columns)], axis=1)  # 详细输出每个样本对应的类别
        r3.columns = list(data.index) + [u'聚类类别']  # 重命名表头
        # r3.to_excel('k_mean'+str(k)+'.xlsx')
        return r2.T

    def k_mean_x(k_mean_flag, data, k,nnn):
        # global nnn
        if k_mean_flag == 1:
            x = list(np.zeros((nnn, 1)))
            for j in range(nnn):
                x[j] = k_mean(k, data[j])
        else:
            x = data
        return x
    if k_mean_flag == 1:
        X = list(map(lambda x: pd.concat([trainadd_x[x], test_x[x]], axis=0), range(nnn)))
        if min_max_flag == 1:
            Xs = list(
                map(lambda x: (X[x] - np.amin(X[x], axis=0)) / (np.amax(X[x], axis=0) - np.amin(X[x], axis=0)), range(nnn)))
        if normalization_flag == 1:
            Xs = list(map(lambda x: (X[x] - np.mean(X[x], axis=0)) / np.std(X[x], axis=0), range(nnn)))
        K_mean_xs = k_mean_x(k_mean_flag, Xs, k, nnn)
        strainadd_kx = list(map(lambda x: K_mean_xs[x][:-n_for], range(nnn)))
        test_kx = list(map(lambda x: K_mean_xs[x][-n_for:], range(nnn)))
        return strainadd_kx,test_kx
    else:
        return strainadd_x,test_x


################################################  wrapper方法   #############################################
def rfe_select_lasso(step,trainadd_x,trainadd_y,nnn,scoring):
    coef=list(np.zeros((nnn, 1)))
    tss = TimeSeriesSplit(n_splits=4)
    lr = lasso()
    rfe = list(np.zeros((nnn, 1)))
    fit = list(np.zeros((nnn, 1)))
    labeln = list(np.zeros((nnn, 1)))
    rfer = list(np.zeros((nnn, 1)))
    srfer = list(np.zeros((nnn, 1)))
    for j in range(0, nnn, 1):
        rfe[j] = RFECV(lr, cv=tss.split(np.array(trainadd_x[j]), np.array(trainadd_y[j])), scoring=scoring['Dv'],
                       step=step,verbose=1,n_jobs=6)
        # lasso用标准化后的数据集筛选
        fit[j] = rfe[j].fit(trainadd_x[j],trainadd_y[j])
        labeln[j] = list(range(fit[j].support_.shape[0]))
        rfer[j] = pd.DataFrame({'label': labeln[j], 'T-F': fit[j].support_, 'rank': rfe[j].ranking_})
        srfer[j] = rfer[j].sort_values(axis=0, ascending=True, by='rank')
    return rfer, srfer,coef


def vssrfe(n_select,trainadd_x,trainadd_y,nnn):
    rfer = list(np.zeros((nnn, 1)))
    labeln=list(np.zeros((nnn, 1)))
    srfer=list(np.zeros((nnn, 1)))
    coef=list(np.zeros((nnn, 1)))
    n_steps=40
    for j in range(nnn):
        X=pd.DataFrame(trainadd_x[j])
        y2=trainadd_y[j]
        m, n_features = pd.DataFrame(X).shape
        X_tmp=X
        labeln[j] = list(range(X.shape[1]))
        X.columns=labeln[j]
        rfer[j]=pd.DataFrame({'rank':np.zeros((X_tmp.shape[1])),'label':labeln[j]})
        count = n_features
        cut=n_features
        i=0
        from sklearn.linear_model import LassoCV as LassoCV
        lassocv=LassoCV()
        lassocv.fit(np.array(X_tmp), np.array(y2))
        Alpha = lassocv.alpha_
        while count > n_select:
              count = np.int(count - n_steps)
              if cut / count==2 and n_steps>1:
                 cut=count
                 n_steps=n_steps/2
              clf = lasso(alpha=Alpha)
              clf.fit(np.array(X_tmp), np.array(y2))
              coef[j]=pd.DataFrame({'coef':np.array([abs(e) for e in (clf.coef_)]),'label':X_tmp.columns})
              coef_sorted=np.argsort(coef[j]['coef'])[::-1]
              coef_eliminated=coef[j]['label'][coef_sorted[:count]]
              X_tmp = X_tmp.loc[:,list(coef_eliminated)]
              X_tmp = X_tmp.dropna(axis=1,how='all')
              rfer[j]['rank'][list(coef_eliminated)]=i
              i=i+1
        srfer[j] = rfer[j].sort_values(axis=0, ascending=False, by='rank')
    return rfer,srfer,coef


def vssrfe_randomF(n_select,trainadd_x,trainadd_y,nnn):
    rfer = list(np.zeros((nnn, 1)))
    labeln=list(np.zeros((nnn, 1)))
    srfer=list(np.zeros((nnn, 1)))
    coef=list(np.zeros((nnn, 1)))
    n_steps=40
    for j in range(nnn):
        X=pd.DataFrame(trainadd_x[j])
        y2=trainadd_y[j]
        m, n_features = pd.DataFrame(X).shape
        X_tmp=X
        labeln[j] = list(range(X.shape[1]))
        X.columns=labeln[j]
        rfer[j]=pd.DataFrame({'rank':np.zeros((X_tmp.shape[1])),'label':labeln[j]})
        count = n_features
        cut=n_features
        i=0
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=50, max_depth=4)
        rf.fit(np.array(X_tmp), np.array(y2))
        while count > n_select:
              count = np.int(count - n_steps)
              if cut / count==2 and n_steps>1:
                 cut=count
                 n_steps=n_steps/2
              clf = rf
              clf.fit(np.array(X_tmp), np.array(y2))
              coef[j]=pd.DataFrame({'coef':np.array(clf.feature_importances_),'label':X_tmp.columns})
              coef_sorted=np.argsort(coef[j]['coef'])[::-1]
              coef_eliminated=coef[j]['label'][coef_sorted[:count]]
              X_tmp = X_tmp.loc[:,list(coef_eliminated)]
              X_tmp = X_tmp.dropna(axis=1,how='all')
              rfer[j]['rank'][list(coef_eliminated)]=i
              i=i+1
        srfer[j] = rfer[j].sort_values(axis=0, ascending=False, by='rank')
    return rfer,srfer,coef


def vssrfe_GBDT(n_select,trainadd_x,trainadd_y,nnn):
    rfer = list(np.zeros((nnn, 1)))
    labeln=list(np.zeros((nnn, 1)))
    srfer=list(np.zeros((nnn, 1)))
    coef=list(np.zeros((nnn, 1)))
    n_steps=40
    for j in range(nnn):
        X=pd.DataFrame(trainadd_x[j])
        y2=trainadd_y[j]
        m, n_features = pd.DataFrame(X).shape
        X_tmp=X
        labeln[j] = list(range(X.shape[1]))
        X.columns=labeln[j]
        rfer[j]=pd.DataFrame({'rank':np.zeros((X_tmp.shape[1])),'label':labeln[j]})
        count = n_features
        cut=n_features
        i=0
        from sklearn import ensemble
        gbdt = ensemble.GradientBoostingRegressor()
        gbdt.fit(np.array(X_tmp), np.array(y2))
        while count > n_select:
              count = np.int(count - n_steps)
              if cut / count==2 and n_steps>1:
                 cut=count
                 n_steps=n_steps/2
              clf = gbdt
              clf.fit(np.array(X_tmp), np.array(y2))
              coef[j]=pd.DataFrame({'coef':np.array(clf.feature_importances_),'label':X_tmp.columns})
              coef_sorted=np.argsort(coef[j]['coef'])[::-1]
              coef_eliminated=coef[j]['label'][coef_sorted[:count]]
              X_tmp = X_tmp.loc[:,list(coef_eliminated)]
              X_tmp = X_tmp.dropna(axis=1,how='all')
              rfer[j]['rank'][list(coef_eliminated)]=i
              i=i+1
        srfer[j] = rfer[j].sort_values(axis=0, ascending=False, by='rank')
    return rfer,srfer,coef


def recursion_normalization_func_v2(trainadd_x, test_x, n_select, rfer,
                                 srfer, nnn):
    train2_x = list(np.zeros((nnn, 1)))
    test2_x = list(np.zeros((nnn, 1)))
    for j in range(nnn):
            trainadd_x[j] = pd.DataFrame(np.array(trainadd_x[j].copy()), columns=rfer[j]['label'])
            test_x[j] = pd.DataFrame(np.array(test_x[j]), columns=rfer[j]['label'])
            train2_x[j] = pd.DataFrame(trainadd_x[j][trainadd_x[j].columns[srfer[j]['label'][0:n_select]]],
                                       dtype=np.float)
            test2_x[j] = pd.DataFrame(test_x[j][test_x[j].columns[srfer[j]['label'][0:n_select]]], dtype=np.float)
    return train2_x, test2_x


##特征处理RFE


def RFE(method,step,n_select,length,n_for,nnn,trainadd_x,trainadd_y,test_x):

    print("特征选择第一步开始!!!")
    if method == 1:
        rfer2, srfer2,coef= rfe_select_lasso(step, trainadd_x, trainadd_y,n_for)
    if method == 2:
        rfer2, srfer2, coef = vssrfe(n_select, trainadd_x, trainadd_y, n_for)
    if method==3:
        rfer2, srfer2, coef=vssrfe_randomF(n_select, trainadd_x, trainadd_y, n_for)
    print("特征选择第一步结束!!!")
    print("特征选择第二步开始!!!")
    train2_x, test2_x = recursion_normalization_func_v2(trainadd_x,test_x,n_select,rfer2, srfer2,n_for)
    print("特征选择第二步结束!!!")
    return rfer2,srfer2,coef,train2_x,test2_x

#############################     数据增强   ############################################################
##特征提取CNN_model
    #n2表示第二层循环的循环次数，n_for
def CNN_model(train_x,train_y,test_x,n2):
    def create_dataset2(train_y,test_x,n2):
        Input = [list() for x in range(n2)]
        Output = [list() for x in range(n2)]
        test_X = list(np.zeros((n2, 1)))
        for j in range(0, n2, 1):
            for i in range(batchsize):
                Input[j] = Input[j] + [train_y[j][i:-test_x[j].shape[0] - batchsize + i]]
                if i != batchsize:
                    Output[j] = Output[j] + [train_y[j][-test_x[j].shape[0] - batchsize + i:-batchsize + i]]
                else:
                    Output[j] = Output[j] + [train_y[j][-test_x[j].shape[0] - batchsize + i:]]
            test_X[j] = np.array(train_y[j][-(train_y[j].shape[0] - test_x[j].shape[0] - batchsize):]).reshape(1,train_y[j][-(train_y[j].shape[0] - test_x[j].shape[0] - batchsize):].shape[0],1)
            Input[j] = np.array(Input[j]).reshape(batchsize, train_y[j].shape[0] - test_x[j].shape[0] - batchsize, 1)
            Output[j] = np.array(Output[j]).reshape(batchsize, test_x[j].shape[0], 1)
        return Input,Output,test_X
    def y_cnn_model(epochs, v_split, dilations, filters,Input2, Output2, test_X):
        from tensorflow.keras.layers import Dense, Conv1D, Dropout, Lambda
        from tensorflow.keras import Input, Model
        # define an input history series and pass it through a stack of dilated causal convolutions.
        i = Input(shape=(None, 1))
        for dilation_rate in dilations:
            o = Conv1D(filters=filters, kernel_size=2, dilation_rate=dilation_rate, padding='causal')(i)
        o = Dense(128, activation='relu')(o)
        o = Dropout(.2)(o)
        o = Dense(1)(o)
        nfor=Output2.shape[1]
        def slice(x, nfor):
            return x[:, -nfor:, :]

        pred_seq_train = Lambda(slice, arguments={'nfor': nfor})(o)
        m = Model(i, pred_seq_train)
        m.compile(optimizer='adam', loss='mse')
        history = m.fit(Input2, Output2,
                        epochs=epochs,
                        validation_split=v_split)
        yhat = m.predict(test_X, verbose=0)
        yhat_train = m.predict(Input2, verbose=0)
        return history, yhat, yhat_train

    def yhat_train_connect(n2, test_x, batchsize,yhat_train, train_y):
        yhat_train2 = [list() for x in range(n2)]
        for i in range(n2):
            yhat_train2[i] = yhat_train[i][0].flatten()
            for j in range(1, batchsize, 1):
                yhat_train2[i] = np.hstack([yhat_train2[i], yhat_train[i][j][-1:].flatten()]).flatten()
            yhat_train2[i] = np.hstack([train_y[i][:(-yhat_train2[i].shape[0])], yhat_train2[i]])
        return yhat_train2

    def create_xgboost_trainadd2(n2,yhat_train_2,yhat_2,batchsize,test_x2,train_y2,train_x2):
        yhat_train2=yhat_train_connect(n2, test_x2, batchsize,yhat_train_2,train_y2)
        yhat22=list(map(lambda x:yhat_2[x].flatten(),range(n2)))
        train_x2222=list(map(lambda x:np.hstack([train_x2[x],yhat_train2[x].reshape(yhat_train2[x].shape[0],1)]),range(n2)))
        test_x2222=list(map(lambda x:np.hstack([test_x2[x],yhat22[x].reshape(yhat22[x].shape[0],1)]),range(n2)))
        return train_x2222,test_x2222,yhat_train2,yhat22

    epoch_num = 200
    verbose_set = 1
    epochs = 50
    v_split = 0.2
    dilations = [1, 2, 4, 8, 16, 32]
    filters = 6
    batchsize = 6

    Input2, Output2, Test_X2 = create_dataset2(train_y, test_x,n2)
    model = list(np.zeros((n2, 1)))
    history = list(np.zeros((n2, 1)))
    history_2 = list(np.zeros((n2, 1)))
    yhat = list(np.zeros((n2, 1)))
    yhat_2 = list(np.zeros((n2, 1)))
    yhat_train = list(np.zeros((n2, 1)))
    yhat_train_2 = list(np.zeros((n2, 1)))
    for j in range(0, n2, 1):
        history_2[j], yhat_2[j], yhat_train_2[j] = y_cnn_model(epochs, v_split, dilations, filters,Input2[j], Output2[j], Test_X2[j])
    train_x2h, test_x2h, yhat_train2, yhat22 = create_xgboost_trainadd2(n2,yhat_train_2,yhat_2,batchsize,test_x, train_y, train_x)
    return train_x2h,test_x2h


def gplearn_generationX(strainx,stestx,train_y,nnn):
    import gplearn as gpl
    from gplearn.genetic import SymbolicTransformer
    train_x2=list(np.zeros((nnn,1)))
    test_x2=list(np.zeros((nnn,1)))
    function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min']
    gp_featurestrain=list(np.zeros((nnn,1)))
    gp_featurestest=list(np.zeros((nnn,1)))
    for j in range(nnn):
        gp = SymbolicTransformer(generations=20, population_size=2000,
                                 hall_of_fame=100, n_components=100,
                                 function_set=function_set,
                                 parsimony_coefficient=0.0005,
                                 max_samples=0.9, verbose=1,
                                 random_state=0, n_jobs=6)
        gp.fit(strainx[j], train_y[j])
        gp_featurestrain[j] = pd.DataFrame(gp.transform(strainx[j]),index=strainx[j].index)
        gp_featurestrain[j]=(gp_featurestrain[j]-np.amin(gp_featurestrain[j]))/(np.amax(gp_featurestrain[j])-np.amin(gp_featurestrain[j]))
        gp_featurestest[j] = pd.DataFrame(gp.transform(stestx[j]),index=stestx[j].index)
        gp_featurestest[j]=(gp_featurestest[j]-np.amin(gp_featurestest[j]))/(np.amax(gp_featurestest[j])-np.amin(gp_featurestest[j]))
        train_x2[j]=pd.concat([gp_featurestrain[j],strainx[j]],axis=1)
        test_x2[j]=pd.concat([gp_featurestest[j],stestx[j]],axis=1)
        train_x2[j]=train_x2[j].fillna(0)
        test_x2[j]=test_x2[j].fillna(0)
    return train_x2,test_x2
####################################################### filter方法 ########################################
def survive_x(theshold, trainadd_y, strainadd_x, stestx, nnn):
    Target = []
    train_x2 = list(np.zeros((nnn, 1)))
    test_x2 = list(np.zeros((nnn, 1)))
    for j in range(nnn):
        target = []
        print("第" + str(j + 1) + "次筛选开始！")
        if stestx[j].shape[0] > 3:
            for_length = stestx[j].shape[0]
            targetpd = pd.concat(
                [pd.DataFrame(trainadd_y[j][(-for_length):], index=pd.DataFrame(strainadd_x[j])[(-for_length):].index),
                 pd.DataFrame(strainadd_x[j])[(-for_length):]], axis=1)
            targetpd.columns = ['y'] + list(pd.DataFrame(strainadd_x[j]).columns)
            targetcorr = targetpd.corr('spearman')
            targetcorr.columns = ['y'] + list(pd.DataFrame(strainadd_x[j]).columns)
            target = targetcorr['y'][abs(targetcorr['y']) > theshold].index
            print(target)
            import sys, time
            i = strainadd_x[j].shape[0] - 3 * for_length
            if target.shape[0] > 10:
                while i > -1:
                    targetpdi = pd.concat([pd.DataFrame(trainadd_y[j][i:-for_length]),pd.DataFrame(strainadd_x[j])[i:-for_length][target.drop(['y'])]], axis=1)
                    targetpdi.columns = list(target)
                    targetcorri = targetpdi.corr('spearman')
                    targetcorri.columns = list(target)
                    target = targetcorri['y'][abs(targetcorri['y']) > theshold].index
                    if target.shape[0] < 5:
                        target = targetcorri['y'].index
                        train_x2[j] = pd.DataFrame(strainadd_x[j][target.drop(['y'])])
                        test_x2[j] = pd.DataFrame(stestx[j][target.drop(['y'])])
                        break
                    i = i - for_length
                    sys.stdout.write('>')
                    sys.stdout.flush()
                    time.sleep(0.1)
            else:
                train_x2[j] = strainadd_x[j]
                test_x2[j] = stestx[j]
        else:
            print("3期以下相关性样本自动跳过")
            train_x2[j] = strainadd_x[j]
            test_x2[j] = stestx[j]

        Target = Target + list(target)

    return Target, train_x2, test_x2


def DTW(strainx1,stestx,train_y,nnn,k,feature_num):
    import numpy as np
    from scipy.spatial.distance import euclidean
    from fastdtw import fastdtw
    distance=[[0 for i in range(strainx1[0].shape[1])] for i in range(nnn)]
    path=[[0 for i in range(strainx1[0].shape[1])] for i in range(nnn)]
    train_x2=list(np.zeros((nnn,1)))
    test_x2=list(np.zeros((nnn,1)))
    for j in range(nnn):
        if (stestx[0].shape[0]) > 3:
            x = np.array(train_y[j][-stestx[j].shape[0]:])
            for i in range(strainx1[j].shape[1]):
                y = np.array(strainx1[j][strainx1[j].columns[i]][-stestx[j].shape[0]:])
                distance[j][i], path[j][i] = fastdtw(x, y, dist=euclidean)

            index=np.where(np.array(distance[j])<k)
            i=1
            while index.shape[0]<feature_num:
                print("k选的太小了！！！！"+str(i))
                index = np.where(np.array(distance[j]) < i*k)
                i=i+1
            train_x2[j]=strainx1[j][strainx1[j].columns[index]]
            test_x2[j]=stestx[j][stestx[j].columns[index]]
        else:
            print("长度小于3期，跳过DTW！！！")
            train_x2[j] = strainx1[j]
            test_x2[j] = stestx[j]
    return train_x2,test_x2


def synchrony(nnn,strainx,train_y,test_x,k):
    from scipy.signal import hilbert, butter, filtfilt
    import numpy as np
    import pandas as pd
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b,a= butter(order,[low,high],btype='bandpass')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    lowcut = 0.01
    highcut = 0.5
    fs = 30.
    order = 1

    train_x2=list(np.zeros((nnn,1)))
    test_x2=list(np.zeros((nnn,1)))
    for j in range(nnn):
        d1 = pd.DataFrame(train_y[j])
        N=strainx[j].shape[1]
        index=[]
        ii = 0
        while np.array(index).shape[0] == 0 and k - ii > 0:
            for i in range(strainx[j].shape[1]):
                d2 = pd.DataFrame(strainx[j][pd.DataFrame(strainx[j]).columns[i]])
                #y1 = butter_bandpass_filter(d1, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
                #y2 = butter_bandpass_filter(d2, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
                al1 = np.angle(hilbert(d1), deg=False)
                al2 = np.angle(hilbert(d2), deg=False)
                phase_synchrony = (1 - np.sin(np.abs(al1 - al2) / 2)).flatten()
                if np.mean(phase_synchrony[-np.int(1/2*train_y[j].shape[0]):]) > k-ii*0.1:
                    index=index+[pd.DataFrame(strainx[j]).columns[i]]
            ii=ii+1

        if np.array(index).shape[0]==0:
            print("提取相位同步的特征失败")
            train_x2[j] = strainx[j]
            test_x2[j] = test_x[j]
        else:
            train_x2[j]=pd.DataFrame(pd.DataFrame(strainx[j])[index])
            test_x2[j]=pd.DataFrame(pd.DataFrame(test_x[j],columns=strainx[j].columns)[index])
    return train_x2,test_x2

def filter(filter_name,strainx,train_y,test_x,k,t,nnn,n_for):
    import numpy as np
    import pandas as pd
    train_x2 = list(np.zeros((nnn, 1)))
    test_x2 = list(np.zeros((nnn, 1)))
    if filter_name=="CORT":
        def cort_func(d2):
            d22 = d2[:-1]
            d2_ = d2[1:]
            d2_d2 = d2_ - d22
            cort = np.abs(np.dot(d2_d2.T, d1_d1) / (np.sqrt(np.dot(d2_d2.T, d2_d2)) * np.sqrt(np.dot(d1_d1.T, d1_d1))))
            return cort
        for j in range(nnn):
            d1 = pd.DataFrame(train_y[j]).iloc[:-1,:]
            d1_=pd.DataFrame(pd.DataFrame(train_y[j]).iloc[1:,:].values,index=range(train_y[j].shape[0]-1))
            d1_d1=d1_-d1
            index = []
            ii = 0
            while np.array(index).shape[0] == 0 and k - ii*0.1 > 0:
                print(str(k-ii*0.1))
                cort=np.apply_along_axis(cort_func,0,np.array(strainx[j]))
                cortm=pd.DataFrame(cort.reshape(1,len(cort.flatten())),columns=pd.DataFrame(strainx[j]).columns)
                cortaddname=cortm[cortm>(k - ii * 0.1)].dropna(axis=1).columns
                index=index+list(cortaddname)
                ii = ii + 1
            if np.array(index).shape[0]==0:
                print("阈值提取特征失败")
                train_x2[j] = strainx[j]
                test_x2[j] = test_x[j]
            else:
                train_x2[j]=pd.DataFrame(pd.DataFrame(strainx[j])[index])
                if test_x[j].shape[0] != 0:
                   test_x2[j]=pd.DataFrame(pd.DataFrame(test_x[j],columns=strainx[j].columns)[index])
    elif filter_name == "CZY_filter":
        from scipy import random
        def column_matrix_corr_matrix(Ytrain_roll, Xtrain_roll, k, bigger=True):
            Ytrain_roll = np.array(Ytrain_roll)
            y_hat = np.add(Ytrain_roll, -Ytrain_roll.mean())
            x_hat = np.add(Xtrain_roll, -Xtrain_roll.mean())
            cov_x_y = np.divide(np.dot(y_hat.T, x_hat), Ytrain_roll.shape[0] - 1)
            tmp_corr = np.divide(cov_x_y[0], Ytrain_roll.std() * Xtrain_roll.std(axis=0))
            tmp_corr = tmp_corr.sort_values(ascending=False)
            if bigger:
                xx = abs(tmp_corr) > k
            else:
                xx = abs(tmp_corr) < k
            return xx.index[xx]
        for j in range(nnn):
            x_survive_tt= column_matrix_corr_matrix(train_y[j], strainx[j], k, bigger=True)
            tRollSpan = strainx[j].index
            for tt in tRollSpan[np.int(len(tRollSpan) / 2):]:
                y_tt = pd.DataFrame(train_y[j]).iloc[:tt]
                d_tt = pd.concat([y_tt, pd.DataFrame(strainx[j]).iloc[:tt]], axis=1)
                x_name_tt = column_matrix_corr_matrix(y_tt, d_tt, k, bigger=True)
                x_survive_tt = list(set(x_name_tt) | set(x_survive_tt))[1:]
            print(x_survive_tt)
            if len(x_survive_tt)!=0:
                train_x2[j] = pd.DataFrame(pd.DataFrame(strainx[j])[x_survive_tt])
                if test_x[j].shape[0] != 0:
                   test_x2[j] = pd.DataFrame(pd.DataFrame(test_x[j],columns=strainx[j].columns)[x_survive_tt])
            else:
                print('没有存活下来的指标')
                rnd_idx = random.permutation(strainx[j].shape[1])[:100]
                train_x2[j] = pd.DataFrame(pd.DataFrame(strainx[j].iloc[:,rnd_idx]))
                if test_x[j].shape[0] != 0:
                   test_x2[j] = pd.DataFrame(pd.DataFrame(test_x[j],columns=strainx[j].columns)[train_x2[j].columns])
    elif filter_name=="synchrony_filter":
        from scipy.signal import hilbert, butter, filtfilt
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='bandpass')
            return b, a

        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = filtfilt(b, a, data)
            return y

        lowcut = 0.01
        highcut = 0.5
        fs = 30.
        order = 1
        def synchrony_func(d2):
            al1 = np.angle(hilbert(d1), deg=False)
            al2 = np.angle(hilbert(d2), deg=False)
            phase_synchrony = (1 - np.sin(np.abs(al1 - al2) / 2)).flatten()
            return np.mean(phase_synchrony)
        for j in range(nnn):
            d1 = pd.DataFrame(train_y[j])
            N = strainx[j].shape[1]
            index = []
            ii = 0
            while np.array(index).shape[0] == 0 and k - ii*0.01 > 0:
                syn=np.apply_along_axis(synchrony_func,0,np.array(strainx[j]))
                synm=pd.DataFrame(syn.reshape(1,len(syn)),columns=pd.DataFrame(strainx[j]).columns)
                synaddname=synm[synm>(k - ii * 0.01)].dropna(axis=1).columns
                index=index+list(synaddname)
                ii = ii + 1

            if np.array(index).shape[0] == 0:
                print("提取相位同步的特征失败")
                train_x2[j] = strainx[j].copy()
                test_x2[j] = test_x[j].copy()
            else:
                train_x2[j] = pd.DataFrame(pd.DataFrame(strainx[j])[index])
                if test_x[j].shape[0] != 0:
                   test_x2[j] = pd.DataFrame(pd.DataFrame(test_x[j],columns=strainx[j].columns)[index])
    elif filter_name=="DTW_filter":
        from model_tools.Multi import DTW_filter
        from multiprocessing import Pool
        zip_args = list(zip(strainx, train_y, test_x,  [k for x in range(nnn)],[t for x in range(nnn)]))
        with Pool() as pool:
            pool = Pool(processes=6)
            res = pool.starmap(DTW_filter, zip_args)
            pool.close()
            pool.join()
        train_x2 = list(map(lambda x: res[x][0], range(nnn)))
        print(train_x2)
        test_x2 = list(map(lambda x: res[x][1], range(nnn)))
    elif filter_name=="MIC":
        def MIC_func(d2):
            mine = MINE(alpha=0.6, c=15)
            mine.compute_score(np.array(d1).flatten(), np.array(d2).flatten())
            MIC = np.around(mine.mic(), 1)
            return MIC

        for j in range(nnn):
            d1 = pd.DataFrame(train_y[j])
            N = strainx[j].shape[1]
            index = []
            ii = 0
            while np.array(index).shape[0] == 0 and k - ii * 0.01 > 0:
                MIC=np.apply_along_axis(MIC_func,0,np.array(strainx[j]))
                MICm=pd.DataFrame(MIC.reshape(1,len(MIC)),columns=pd.DataFrame(strainx[j]).columns)
                MICaddname=MICm[MICm>(k - ii * 0.01)].dropna(axis=1).columns
                index=index+list(MICaddname)
                ii = ii + 1

            if np.array(index).shape[0] == 0:
                print("MIC筛选特征失败")
                train_x2[j] = strainx[j].copy()
                test_x2[j] = pd.DataFrame(test_x[j]).copy()
            else:
                train_x2[j] = pd.DataFrame(pd.DataFrame(strainx[j])[index])
                if test_x[j].shape[0]!=0:
                    test_x2[j] = pd.DataFrame(pd.DataFrame(test_x[j],columns=strainx[j].columns)[index])
    elif filter_name=="HSIC":
        import numpy as np
        def hsic(x, y):
            Kx = np.expand_dims(x, 0) - np.expand_dims(x, 1)
            Kx = np.exp(-Kx.astype(float) ** 2)  # 计算核矩阵
            Ky = np.expand_dims(y, 0).astype(float) - np.expand_dims(y, 1).astype(float)
            Ky = np.exp(-Ky.astype(float) ** 2)  # 计算核矩阵
            Kxy = np.dot(Kx, Ky)
            n = Kxy.shape[0]
            h = np.trace(Kxy) / n ** 2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
            return h * n ** 2 / (n - 1) ** 2

        def HSIC_func(d2):
            HSIC = hsic(np.array(d1).flatten(), np.array(d2).flatten())
            return HSIC
        for j in range(nnn):
            d1 = train_y[j]
            index = []
            ii = 0
            while np.array(index).shape[0] == 0 and k - ii*0.00001 > 0:
                HSIC = np.apply_along_axis(HSIC_func, 0, np.array(strainx[j]))
                HSICm = pd.DataFrame(HSIC.reshape(1, len(HSIC)), columns=pd.DataFrame(strainx[j]).columns)
                HSICaddname = HSICm[HSICm > (k - ii * 0.00001)].dropna(axis=1).columns
                index = index + list(HSICaddname)
                ii = ii + 1

            if np.array(index).shape[0] == 0:
                print("HSIC筛选特征失败")
                train_x2[j] = strainx[j].copy()
                test_x2[j] = test_x[j].copy()
            else:
                train_x2[j] = pd.DataFrame(pd.DataFrame(strainx[j])[index])
                if test_x[j].shape[0] != 0:
                    test_x2[j] = pd.DataFrame(pd.DataFrame(test_x[j],columns=strainx[j].columns)[index])
    elif filter_name=="Pearson":
        for j in range(nnn):
            k2 = 0.8
            X_num = 80

            def findcorr(Ytrain_roll, Xtrain_roll, k1_1, X_num):
                while 1:
                    chooseXmatrix2 = column_matrix_corr_matrix(Ytrain_roll, Xtrain_roll, k1_1)
                    k1_1 -= 0.05
                    if chooseXmatrix2.shape[1] > X_num:
                        break
                return chooseXmatrix2.columns, k1_1

            chooseX1, k1_1 = findcorr(train_y[j], strainx[j], k, X_num)
            tmptestx = pd.DataFrame(test_x[j],columns=strainx[j].columns)
            chooseX2, k1_2 = findcorr(train_y[j][-n_for: ],tmptestx, k, X_num)
            XnameAll0 = chooseX1 & chooseX2

            while len(chooseX1) < 100:
                X_num += 100
                chooseX1, k1_1 = findcorr(train_y[j],strainx[j], k1_1, X_num)
                chooseX2, k1_2 = findcorr(train_y[j].iloc[-n_for:, ], test_x[j], k1_2, X_num)
                XnameAll0 = chooseX1 & chooseX2

            if len(XnameAll0)>1:
                chooseXmatrix3 = del_dupli(strainx[j][XnameAll0], k2)

            train_x2[j]=pd.DataFrame(pd.DataFrame(strainx[j])[chooseXmatrix3.columns])
            test_x2[j]=pd.DataFrame(pd.DataFrame(test_x[j],columns=strainx[j].columns)[chooseXmatrix3.columns])
    elif filter_name[0:7] == "HMImage":
        if 'step' in filter_name:
            print('image_step')
            pic_len = int(64 / 72 * strainx[0].shape[0])
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
                np.linspace(0, np.floor(nnn / feature_step) * feature_step, int(np.floor(nnn / feature_step)),
                            endpoint=False, dtype=int)))
            if len(feature_index) == 0:
                feature_index = [0]
            print('要计算的期数为{}'.format(feature_index))
            feature_index.sort()
            strainx_cal = []
            train_y_cal = []
            test_x_cal = []
            for ff in feature_index:
                strainx_cal.append(strainx[ff])
                train_y_cal.append(train_y[ff])
                test_x_cal.append(test_x[ff])
            nnn_cal = len(feature_index)
            train_x2_cal = list(map(lambda x: cal_HMD(strainx_cal[x], train_y_cal[x], test_x_cal[x], list([pic_len]),
                                                      list([image_method]), list([k]), list([Hash_method]))[0],
                                    range(nnn_cal)))
            print(train_x2)
            test_x2_cal = list(map(lambda x: cal_HMD(strainx_cal[x], train_y_cal[x], test_x_cal[x], list([pic_len]),
                                                     list([image_method]), list([k]), list([Hash_method]))[1],
                                   range(nnn_cal)))
            train_x2 = []
            test_x2 = []
            for j in range(nnn):
                if j in feature_index:
                    train_x2.append(train_x2_cal[feature_index.index(j)])
                    test_x2.append(test_x2_cal[feature_index.index(j)])
                    print("第{}个滚动期，计算特征工程".format(j + 1))
                else:
                    train_x2.append(strainx[j][train_x2[-1].columns])
                    tmptestx2 = pd.DataFrame(test_x[j], columns=strainx[j].columns)
                    test_x2.append(tmptestx2[test_x2[-1].columns])
                    print("第{}个滚动期，不计算特征工程".format(j + 1))
        else:
            print('image')
            pic_len = int(64 / 72 * strainx[0].shape[0])
            if pic_len % 2 != 0:
                pic_len += 1
            image_method = int(filter_name[8])
            Hash_method = filter_name[-1]
            from multiprocessing import Pool
            from model_tools.Multi import cal_HMD
            # zip_args = list(zip(strainx, train_y, test_x, [list([pic_len]) for x in range(nnn)],
            #                     [list([image_method]) for x in range(nnn)], [list([k]) for x in range(nnn)],
            #                     [list([Hash_method]) for x in range(nnn)]))
            # pool = Pool(processes=6)
            # res = pool.starmap(cal_HMD, zip_args)
            # pool.close()
            # pool.join()
            train_x2 = list(map(lambda x:
                                cal_HMD(strainx[x], train_y[x], test_x[x], list([pic_len]), list([image_method]),
                                        list([k]), list([Hash_method]))[0], range(nnn)))
            print(train_x2)
            test_x2 = list(map(lambda x:
                               cal_HMD(strainx[x], train_y[x], test_x[x], list([pic_len]), list([image_method]),
                                       list([k]), list([Hash_method]))[1], range(nnn)))
    elif filter_name == "CrossPearson":
        def cross_corr(x, y):
            y = np.array(y).flatten()
            x = np.array(x)
            Rxy = np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)
            return Rxy

        for j in range(nnn):
            XDcorr = np.apply_along_axis(cross_corr, 0, strainx[j], train_y[j])
            Xchoose = strainx[j].loc[:, abs(XDcorr) > k]
            ii = 1
            while len(Xchoose.columns) <= 200:
                Xchoose = strainx[j].loc[:, abs(XDcorr) > (k - ii * 0.02)]
                ii += 1
            print("cross_corr:" + str(k - (ii - 1) * 0.02))
            if len(Xchoose) > 1:
                Xchoose = del_dupli(Xchoose, 0.8)

            train_x2[j] = Xchoose
            test_x2[j] = pd.DataFrame(pd.DataFrame(test_x[j], columns=strainx[j].columns)[Xchoose.columns])
    elif filter_name == 'diedaifilterSA':
        def diedaifilterSA(strainx, trainy, test_x, k, nnn, n_for):
            def xunhuan_model(train_set, train_y2, test_x):
                from skfeature.function.information_theoretical_based.GA import Genetic_Algorithm
                from sklearn.ensemble import ExtraTreesRegressor  # 使用ExtraTrees 模型作为示范
                from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
                # 选择模拟退火中评价特征子集的的损失函数
                from sklearn.metrics import mean_squared_error  # 回归问题我们使用MSE
                import xgboost as xgb
                # clf = ExtraTreesRegressor(n_estimators=25)
                # best = {}
                # best['numCenters'] = 10
                # best['up'] = 0.01
                # best['beta'] = 0.00001
                # x_old = self.train_set
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print((test_x.shape))

                def experiment(params):
                    try:
                        trainrate = params['trainrate']
                        # print("**************************")
                        # x = np.matrix(train_set.iloc[:int(trainrate * train_set.shape[0]), :])
                        # x2 = np.matrix(train_set.iloc[int(trainrate * train_set.shape[0]):, :])
                        # y = np.matrix(train_y2[:int(trainrate * train_set.shape[0])]).T
                        x = (train_set.iloc[:int(trainrate * train_set.shape[0]), :])
                        x2 = (train_set.iloc[int(trainrate * train_set.shape[0]):, :])
                        y = (train_y2[:int(trainrate * train_set.shape[0])])
                        model = xgb.XGBRegressor(n_estimator=int(params['n_estimator']))
                        model.fit(x, y)
                        pred = model.predict(x2)
                        # print("&&&&&&&&&&&&&&&&&&&&&&&&")
                        # print(pred)
                    except:
                        print('Something happened')
                        print('-' * 10)
                        return {'loss': 999999, 'status': STATUS_OK}

                    predicted = pred
                    original = train_y2[int(trainrate * train_set.shape[0]):]
                    # print(original)
                    mse = np.mean(np.square(predicted - original))
                    # print('mse是%s'%mse)
                    if np.isnan(mse):
                        print('NaN happened')
                        print('-' * 10)
                        return {'loss': 999999, 'status': STATUS_OK}

                    # print(mse)
                    # print('-' * 10)
                    sys.stdout.flush()
                    return {'loss': mse, 'status': STATUS_OK}

                print('*************************')
                print(train_set.shape)
                max_depth1 = [3, 4, 5, 6, 7, 8, 9, 10]
                space4 = {'n_estimator': hp.uniform('n_estimator', 100, 400),
                          # 'max_depth': hp.choice ('max_depth', max_depth1),
                          # 'beta': hp.uniform('beta', 0.00001, 10),
                          'trainrate': hp.uniform('trainrate', 0.5, 0.9)
                          }
                trials = Trials()
                best = fmin(experiment, space4, algo=tpe.suggest, max_evals=5, trials=trials)
                print('best n_estimator]%s' % best['n_estimator'])
                clf = xgb.XGBRegressor(n_estimators=int(best['n_estimator']))
                selector = Genetic_Algorithm(loss_func=mean_squared_error, estimator=clf,
                                             n_gen=10, n_pop=20, algorithm='one-max')
                # 在训练集中训练
                # GA.py中有具体每个参数的含义，此处不赘述
                print('进行ga模型开始吃')
                print(train_set.shape)
                selector.fit(X_train=train_set.values, y_train=train_y2, cv=5)
                # print(len(self.df.columns[selector.best_sol]))
                # print((self.df.columns[selector.best_sol]))
                dd = pd.DataFrame(columns=['特征名'])
                print('selector.best_loss%s' % selector.best_loss)
                # print(train_set.columns[selector.best_sol])
                name = train_set.columns[selector.best_sol]
                print('selector%s' % selector._cross_val)
                # dd['特征'] = train_set[list(name)]
                dd['特征名'] = train_set.columns[selector.best_sol]
                dd.index = range(len(dd))
                feature_name = dd['特征名'].values
                print('特征名')
                print(feature_name)
                # feature_list.append(feature_name)
                end = train_set[feature_name]
                test_x1 = pd.DataFrame(test_x, columns=train_set.columns)
                test = test_x1[feature_name]
                # trainy1 = pd.DataFrame(train_y2,columns = ['y'])
                # end1 = pd.concat([end,trainy1],axis = 1)
                loss = selector.best_loss
                return end, test, loss

            def diedai(train_set, train_y, test_x):
                aa = train_set
                print(len(aa))
                i = 0
                loss = []
                loss.append(1.0)
                while True:
                    i = i + 1
                    ga, test, loss1 = xunhuan_model(train_set, train_y, test_x)
                    print(loss1[0])
                    loss.append(loss1[0])
                    print('进行ga选择前的特征数%s' % (aa.shape[1]))
                    print('进行ga选择后的特征数%s' % ga.shape[1])
                    print('迭代后的loss%s' % loss[i])
                    print('迭代前的loss%s' % loss[i - 1])
                    if (loss[i] <= loss[i - 1]) and (ga.shape[1] < aa.shape[1]):
                        train_set = ga
                        test_x = test
                        print(test_x.shape)
                        print('ga后的%s' % (train_set.shape[1]))
                        aa = ga
                    else:
                        break
                # res, model_name = ga.GA()
                # ga.to_excel(r'C:\zhuque\zhuque\新建文件夹 (4)\结果数据集).xlsx')
                print(i)
                print(ga)
                print(test.shape)
                return ga, test

            train_x2 = list(map(lambda x: diedai(strainx[x], trainy[x], test_x[x])[0], range(nnn)))
            test_x2 = list(map(lambda x: diedai(strainx[x], trainy[x], test_x[x])[1], range(nnn)))
            return train_x2, test_x2

        train_x2, test_x2 = diedaifilterSA(strainx, train_y, test_x, k, nnn, n_for)
    elif filter_name == 'DD_filter':
        def matric_filter(train_x, train_y, test_x, k, nnn):
            from pyts.image import GramianAngularField
            import SeasonalAdj
            import datetime
            import scipy.signal as signal
            def tc_x12(data):
                begyear = 2000
                begmonth = 1
                indexl = []
                for i in range(len(data)):
                    indexl.append(datetime.date(begyear, begmonth + 1, 1) - datetime.timedelta(days=1))
                    if begmonth + 1 != 12:
                        begmonth += 1
                    else:
                        begyear += 1
                        begmonth = 0
                data.index = np.array(indexl)

                x12 = SeasonalAdj.X12a()
                ret12 = x12.x12(pd.DataFrame(data), mode='add')  # mode 'add'
                SA = ret12['SA']
                SF = ret12['SF']
                TC = ret12['TC']
                return np.array(TC).flatten()

            def infoentropy(x1, label):
                x1 = pd.DataFrame(x1)
                diffx1 = x1.diff()
                t = 1.5
                L = np.array(
                    list(map(lambda x: diffx1.loc[diffx1.index[x]] * diffx1.loc[diffx1.index[x + 1]],
                             range(len(diffx1) - 1))))
                L[L < 0] = 1  # 前后差分反向的是拐点
                R1 = np.array(
                    list(map(lambda x: diffx1.loc[diffx1.index[x]] / diffx1.loc[diffx1.index[x + 1]],
                             range(len(diffx1) - 1))))
                con = np.vstack((L, R1))
                L[np.where((con[1, :] > t) | (con[1, :] < 1 / t))] = 1
                L[np.where((con[1, :] < 0) & (L != 1))] = 0
                L[np.where((L != 0) & (L != 1))] = 0
                output = pd.concat([x1, pd.DataFrame(L, index=x1.index[:-1], columns=['0-1事件'])], axis=1)
                output = output.fillna(0)
                # output.to_csv('output='+str(t)+'_extend='+str(extend_flag)+str(n_extend)+'.csv')

                # 计算信息熵
                window = 6
                # 滚动计算信息熵
                index = np.array(range(len(x1) - window))
                batch = list(map(lambda x: output.values[index[x]:index[x] + window, 1], range(len(index))))
                H = np.zeros((len(batch), 1))

                def p(x):
                    if x == 0:
                        return 0
                    else:
                        return 1

                for x in range(len(batch)):
                    H[x] = np.array(list(map(lambda j: -(1 - p(batch[x][j]) + batch[x].sum() / window) * \
                                                       np.log((1 - p(batch[x][j]) + batch[x].sum() / window)),
                                             range(window)))).sum()
                output2 = pd.concat(
                    [output, pd.DataFrame(H, index=output.index[window:window + len(H)], columns=['信息熵'])],
                    axis=1)
                # output2.to_csv('O(window='+str(window)+')_extend='+str(extend_flag)+str(n_extend)+'.csv')
                # 标记tc后的拐点
                x1tc = tc_x12(pd.DataFrame(x1))
                output2 = pd.concat([output2, pd.DataFrame(x1tc, columns=['tc'], index=output.index[0:len(x1tc)])],
                                    axis=1)

                # 构建tc拐点识别序列
                point = x1tc[signal.argrelextrema(x1tc, np.greater)]
                pointseries = np.zeros((len(x1tc), 1))
                pointseries[signal.argrelextrema(x1tc, np.less)] = -1
                pointseries[signal.argrelextrema(x1tc, np.greater)] = 1
                # x1：window的信息熵
                output2 = pd.concat(
                    [output2,
                     pd.DataFrame(pointseries, columns=['pointseries'], index=output2.index[0:len(pointseries)])],
                    axis=1)
                # output2.to_csv('O+tc(window='+str(window)+')_extend='+str(extend_flag)+str(n_extend)+'.csv')
                return output2[label]

            def mtx_similar1(arr1: np.ndarray, arr2: np.ndarray) -> float:
                '''
                计算矩阵相似度的一种方法。将矩阵展平成向量，计算向量的乘积除以模长。
                注意有展平操作。
                :param arr1:矩阵1
                :param arr2:矩阵2
                :return:实际是夹角的余弦值，ret = (cos+1)/2
                '''
                farr1 = arr1.ravel()
                farr2 = arr2.ravel()
                len1 = len(farr1)
                len2 = len(farr2)
                if len1 > len2:
                    farr1 = farr1[:len2]
                else:
                    farr2 = farr2[:len1]

                numer = np.sum(farr1 * farr2)
                denom = np.sqrt(np.sum(farr1 ** 2) * np.sum(farr2 ** 2))
                similar = numer / denom  # 这实际是夹角的余弦值
                return (similar + 1) / 2  # 姑且把余弦函数当线性

            dem = train_x[0].shape[1]
            train_x3 = list(np.zeros((nnn, 1)))
            test_x3 = list(np.zeros((nnn, 1)))
            test_xx = list(np.zeros((nnn, 1)))
            train_x_lists = [[] for _ in range(dem)]
            test_x_lists = [[] for _ in range(dem)]
            for i in range(dem):
                train_x_lists[i] = list(np.zeros((nnn, 1)))
                test_x_lists[i] = list(np.zeros((nnn, 1)))
            train_y_info = list(np.zeros((nnn, 1)))
            train_X_gasf = [[] for _ in range(dem)]

            for i in range(nnn):
                gasf = GramianAngularField(image_size=train_x[i].shape[0], method='summation')
                # 将train_y转化成分类数据
                # train_y_info[0] = infoentropy(train_y[i], 'pointseries')
                # 将train_x转化成分类数据
                Xchoose = []
                for j in range(dem):
                    # train_x_lists[j][i] = infoentropy(train_x[i].iloc[:, j].values, 'pointseries')
                    train_X_gasf[j] = list(gasf.fit_transform(train_x[i].iloc[:, j].values.reshape(1, -1)))
                    # 定义gram矩阵转化
                    train_Y_gasf = list(gasf.fit_transform(train_y[i].reshape(1, -1)))
                    a1 = mtx_similar1(train_X_gasf[j][0], train_Y_gasf[0])
                    print('矩阵相似度为：' + str(a1))
                    if a1 >= k:
                        Xchoose.append(j)
                train_x3[i] = pd.DataFrame(train_x[i].iloc[:, Xchoose])
                test_xx[i] = pd.DataFrame(test_x[i], columns=strainx[i].columns)
                test_x3[i] = pd.DataFrame(test_xx[i].iloc[:, Xchoose])
            return train_x3, test_x3

        train_x2, test_x2 = matric_filter(strainx, train_y, test_x, k, nnn)
    else:
        train_x2 = list(map(lambda x: strainx[x], range(nnn)))
        for x in range(nnn):
            test_x[x]=pd.DataFrame(test_x[x])
            pd.DataFrame(test_x[x]).columns=strainx[x].columns
        test_x2 = list(map(lambda x: test_x[x], range(nnn)))
    return train_x2,test_x2
    # elif filter_name[0:7] == 'History':
    #     from filter_select import getfiltername
    #     print(filter_name.split('-')[1])
    #     import os
    #     print(os.listdir(os.path.pardir))
    #     filterset_name = getfiltername(filter_name.split('-')[1])
    #     print(strainx[0].columns)
    #     print(list(set(filterset_name).intersection(set(strainx[0].columns))))
    #     train_x2 = list(
    #         map(lambda x: strainx[x][list(set(filterset_name).intersection(set(strainx[x].columns)))], range(nnn)))
    #     for x in range(nnn):
    #         test_x[x] = pd.DataFrame(test_x[x])
    #         pd.DataFrame(test_x[x]).columns = strainx[x].columns
    #     test_x2 = list(
    #         map(lambda x: test_x[x][list(set(filterset_name).intersection(set(strainx[x].columns)))], range(nnn)))
    # elif filter_name=="LLS_filter":
    #     class Feature():
    #         def __init__(self, df, X, Y, method, N):
    #             """
    #             :param df  数据集
    #             :param X   自变量
    #             :param Y   标签 label
    #             :param mrthod 特征选择算法名
    #             :param N: 选择前N个特征
    #             """
    #             self.df = df
    #             self.train_set = X
    #             self.train_y = Y
    #             self.method = method
    #             self.N = N
    #
    #         def RandomForest(self):
    #             """
    #              :param 算法类型
    #              :return:权重值
    #                      算法名
    #              """
    #             if self.method == "RandomForestRegressor":
    #                 clf = RandomForestRegressor(n_estimators=50, random_state=123)
    #                 clf.fit(self.train_set, self.train_y)
    #             elif self.method == "Adaboost":
    #                 clf = AdaBoostRegressor()
    #                 clf.fit(self.train_set, self.train_y)
    #             elif self.method == "XGBboost":
    #                 params = {'booster': 'gbtree',
    #                           'objective': 'reg: gamma',
    #                           'gamma': 0.1,
    #                           'max_depth': 5,
    #                           'lambda ': 3,
    #                           'subsample': 0.7,
    #                           'colsample_bytree': 0.7,
    #                           'min_child_weight': 3,
    #                           'silent': 1,
    #                           'eta': 0.1,
    #                           'seed': 1000,
    #                           'nthread': -1,
    #                           }
    #                 dtrain = xgb.DMatrix(self.train_set, self.train_y)
    #                 num_rounds = 300
    #                 plst = params.items()
    #                 # clf = xgb.train(plst, dtrain, num_rounds)
    #                 clf = XGBR(n_estimators=100)
    #                 clf.fit(self.train_set, self.train_y)
    #                 # clf.save_model('0001.model')
    #             elif self.method == "CatBoost":
    #                 clf = cb.CatBoostRegressor()
    #                 clf.fit(self.train_set, self.train_y)
    #                 # clf.save_model('%s.model'%self.method)
    #             res = clf.feature_importances_
    #             importances = res[:self.N]
    #             indices = np.argsort(importances)
    #             res1 = pd.DataFrame(columns=['特征名', '权重系数'])
    #             res1['特征名'] = np.array(self.df.columns)[indices]
    #             res1['权重系数'] = importances[indices]
    #             res1.sort_values(by='权重系数', ascending=False, inplace=True)
    #             res1.index = range(len(res1))
    #             return clf.feature_importances_, res1, self.method  # should only remove the 4th features
    #
    #         def plot_machine_learing(self, res):
    #             """
    #             :param res: 特征重要性
    #             :return: 该模型的画图
    #             """
    #             # plot the variable importance
    #             import matplotlib.pyplot as plt
    #             # %% matplotlibinline
    #             # clf.feature_importances_ = res
    #             importances = res[:self.N]
    #             indices = np.argsort(importances)
    #             # res1 = pd.DataFrame(columns=['特征名','权重系数'])
    #             # res1.to_excel(r'C:\Users\10910\Desktop\新建文件夹 (3)\特征工程.xlsx',sheet_name='%s'%self.method)
    #             # res1['特征名']= np.array(self.df.columns)[indices]
    #             #         # res1['权重系数'] = importances[indices]
    #             #         # print(res1)
    #             plt.figure(figsize=(25, 30), dpi=100)
    #             # 设置横纵坐标的名称以及对应字体格式
    #             font2 = {'family': 'Times New Roman',
    #                      'weight': 'normal',
    #                      'size': 30,
    #                      }
    #             plt.title('%s Feature Importances' % self.method, font2)
    #             plt.barh(range(len(indices)), importances[indices], color='seagreen', align='center')
    #             plt.yticks(range(len(indices)), np.array(self.df.columns)[indices], fontsize=20)
    #             plt.xticks(fontsize=25)
    #             plt.xlabel('Relative Importance', font2)
    #             save_path = '.\\res\\'
    #             png_date = strftime('%Y%m%d%H%M', localtime())
    #             png_name = save_path + ('%s' % self.method) + png_date + '.jpg'
    #             plt.savefig(png_name, dpi=500, bbox_inches='tight', pad_inches=0)
    #             plt.show()
    #             # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
    #             plt.clf()
    #             plt.close()
    #
    #         ################单变量特征选择
    #         def udf_pearsonr(self, X, y):
    #             """
    #             :param X: 自变量
    #             :param y: 因变量
    #             :return: person值
    #             """
    #             # 将会分别计算每一个变量与目标变量的关系
    #             result = np.array([pearsonr(x, y) for x in X.T])  # 包含(皮尔森相关系数, p值) 的列表
    #             return np.absolute(result[:, 0]), result[:, 1]
    #
    #             # SelectKBest 将会基于一个判别函数自动选择得分高的变量
    #
    #         def person(self):
    #             """
    #             计算Person相关系数
    #             :return:
    #             """
    #             # 这里的判别函数为皮尔森相关系数
    #             # self.train_set = self.train_set.astype(dtype)
    #             # self.train_y = self.train_y.astype(dtype)
    #             selector = SelectKBest(self.udf_pearsonr, k=self.N)  # k => 我们想要选择的变量数
    #             selector.fit(self.train_set, self.train_y)  # 在训练集上训练
    #             dd = pd.DataFrame(columns=['特征名', '相关系数', 'P_value'])
    #             res1 = []
    #             res2 = []
    #             for idx in range(self.train_set.shape[1]):
    #                 pea_score, p_value = pearsonr(self.train_set[:, idx], self.train_y)
    #                 res1.append(round(np.abs(pea_score), 2))
    #                 res2.append(round(p_value, 3))
    #                 print(f"第{idx + 1}个变量和目标的皮尔森相关系数的绝对值为{round(np.abs(pea_score), 2)}, p-值为{round(p_value, 3)}")
    #             dd['特征名'] = self.df.columns
    #             dd['相关系数'] = res1
    #             dd['P_value'] = res2
    #             # print(dd)
    #             dd.sort_values(by='相关系数', ascending=False, inplace=True)
    #             dd.index = range(len(dd))
    #             return dd[:self.N], self.method
    #             # 应选择第几个变量
    #
    #         def udf_dcorr(self, X, y):
    #             """
    #             计算距离值
    #             :param X:
    #             :param y:
    #             :return:
    #             """
    #             # 将会分别计算每一个变量与目标变量的关系
    #             result = np.array([[distance_correlation(x, y),
    #                                 distance_covariance_test(x, y)[0]] for x in X.T])  # 包含(距离相关系数, p值) 的列表
    #             return result[:, 0], result[:, 1]
    #
    #         def distance_person(self):
    #             """
    #             计算距离相关系数
    #             :return:
    #             """
    #             selector = SelectKBest(self.udf_dcorr, k=self.N)  # k => 我们想要选择的变量数
    #             selector.fit(self.train_set, self.train_y)  # 在训练集上训练
    #             dd = pd.DataFrame(columns=['特征名', '相关系数', 'P_value'])
    #             res1 = []
    #             res2 = []
    #             for idx in range(self.train_set.shape[1]):
    #                 d_score = distance_correlation(self.train_set[:, idx], self.train_y)
    #                 p_value = distance_covariance_test(self.train_set[:, idx], self.train_y)[0]
    #                 print(f"第{idx + 1}个变量和目标的距离相关系数为{round(d_score, 2)}, p-值为{round(p_value, 3)}")
    #                 res1.append(round(np.abs(d_score), 2))
    #                 res2.append(round(p_value, 3))
    #             dd['特征名'] = self.df.columns
    #             dd['相关系数'] = res1
    #             dd['P_value'] = res2
    #             dd.sort_values(by='相关系数', ascending=False, inplace=True)
    #             dd.index = range(len(dd))
    #             return dd[:self.N], self.method
    #
    #         def udf_MI(self, X, y):
    #             """
    #             计算MI值
    #             :param X:
    #             :param y:
    #             :return:
    #             """
    #             result = mutual_info_regression(X, y, n_neighbors=5)  # 用户可以输入想要的临近数
    #             return result
    #
    #         def MIC(self):
    #             """
    #             计算所有特征的MIC值
    #             :return:
    #             """
    #             selector = SelectKBest(self.udf_MI, k=self.N)  # k => 我们想要选择的变量数
    #             selector.fit(self.train_set, self.train_y)
    #             dd = pd.DataFrame(columns=['特征名', '相关系数'])
    #             res1 = []
    #             res2 = []
    #             for idx in range(self.train_set.shape[1]):
    #                 score = mutual_info_regression(self.train_set[:, idx].reshape(-1, 1), self.train_y, n_neighbors=5)
    #                 print(f"第{idx + 1}个变量与因变量的互信息为{round(score[0], 2)}")
    #                 res1.append(round(score[0], 2))
    #             dd['特征名'] = self.df.columns
    #             dd['相关系数'] = res1
    #             dd.sort_values(by='相关系数', ascending=False, inplace=True)
    #             dd.index = range(len(dd))
    #             return dd[:self.N], self.method
    #
    #         #################多元特征选择
    #         def MRMR(self):
    #             """
    #             :return:
    #             """
    #             from skfeature.function.information_theoretical_based import MRMR
    #             feature_index, J_CMI, MIvalue = MRMR.mrmr(self.train_set, self.train_y,
    #                                                       n_selected_features=self.N)  # 在训练集上训练
    #             # print(J_CMI)
    #             # print(MIvalue)
    #             # print(self.df.columns[feature_index])
    #             transformed_train = self.train_set[:, feature_index]  # 转换训练集
    #             dd = pd.DataFrame(columns=['特征名'])
    #             dd['特征名'] = self.df.columns[feature_index]
    #             # dd['对应的值'] = J_CMI
    #             dd.index = range(len(dd))
    #             # print(feature_index)
    #             # print(dd)
    #             return dd, self.method
    #
    #         def FCBF(self):
    #             """
    #             :return:
    #             """
    #             from skfeature.function.information_theoretical_based import FCBF
    #             # print(self.train_set.astype(float)[0])
    #             feature_index = \
    #                 FCBF.fcbf(self.train_set.astype(int), self.train_y.astype(int), n_selected_features=self.N)[
    #                     0]  # 在训练集上训练
    #             transformed_train = self.train_set[:, feature_index[0:self.N]]
    #             # print(feature_index)
    #
    #         def ReliefF(self):
    #             """
    #             :return:
    #             """
    #             from skfeature.function.similarity_based import reliefF
    #             score = reliefF.reliefF(self.train_set, self.train_y)  # 计算每一个变量的权重
    #             feature_index = reliefF.feature_ranking(score)[0:self.N]  # 依据权重选择变量
    #             # print(self.df.columns[feature_index])
    #             # print(score)
    #             transformed_train = self.train_set[:, feature_index[0:self.N]]  # 转换训练集
    #             dd = pd.DataFrame(columns=['特征名'])
    #             dd['特征名'] = self.df.columns[feature_index]
    #             # dd['对应的值'] = J_CMI
    #             dd.index = range(len(dd))
    #             # print(feature_index)
    #             # print(dd)
    #             return dd, self.method
    #
    #         def SPEC(self):
    #             """
    #             :return:
    #             """
    #             from skfeature.function.similarity_based import SPEC
    #             score = SPEC.spec(self.train_set)  # 计算每一个变量的得分
    #             # print(score)
    #             feature_index = SPEC.feature_ranking(score)[0:self.N]  # 依据变量得分选择变量
    #             # print(feature_index)
    #             # print(self.df.columns[feature_index])
    #             transformed_train = self.train_set[:, feature_index[0:self.N]]
    #             dd = pd.DataFrame(columns=['特征名'])
    #             dd['特征名'] = self.df.columns[feature_index]
    #             # dd['对应的值'] = J_CMI
    #             dd.index = range(len(dd))
    #             # print(feature_index)
    #             # print(dd)
    #             return dd, self.method
    #
    #         ############################################封装方法
    #         ############################确定性方法
    #         ###################SBS递归消除算法
    #         def RFECV(self):
    #             """
    #             :return:
    #             """
    #             from sklearn.feature_selection import RFECV
    #             from sklearn.ensemble import ExtraTreesRegressor  # 使用ExtraTrees 模型作为示范
    #             clf = ExtraTreesRegressor(n_estimators=25)
    #             selector = RFECV(estimator=clf, step=1, cv=5)  # 使用5折交叉验证
    #             # 每一步我们仅删除一个变量
    #             selector = selector.fit(self.train_set, self.train_y)
    #             # print(self.df.columns[selector.support_])
    #             dd = pd.DataFrame(columns=['特征名'])
    #             dd['特征名'] = self.df.columns[selector.support_]
    #             dd.index = range(len(dd))
    #             return dd, self.method
    #
    #         ########################随机方法
    #         #########     SA退火算法
    #         def SA(self):
    #             """
    #             :return:
    #             """
    #             from skfeature.function.information_theoretical_based.SA import Simulated_Annealing
    #             from sklearn.ensemble import ExtraTreesRegressor  # 使用ExtraTrees 模型作为示范
    #
    #             # 选择模拟退火中评价特征子集的的损失函数
    #             from sklearn.metrics import mean_squared_error  # 回归问题我们使用MSE
    #
    #             clf = ExtraTreesRegressor(n_estimators=25)
    #             selector = Simulated_Annealing(loss_func=mean_squared_error, estimator=clf,
    #                                            init_temp=0.2, min_temp=0.005, iteration=10, alpha=0.9)
    #             # 在训练集中训练
    #             # SA.py中有具体每个参数的含义，此处不赘述
    #
    #             selector.fit(X_train=self.train_set, y_train=self.train_y, cv=5)  # 使用5折交叉验证
    #             # (selector.best_sol)
    #             # print(len(self.df.columns[selector.best_sol]))
    #             dd = pd.DataFrame(columns=['特征名'])
    #             dd['特征名'] = self.df.columns[selector.best_sol]
    #             dd.index = range(len(dd))
    #             # print(selector.best_loss)
    #             return dd, self.method
    #
    #         #########遗传算法GA
    #         def GA(self):
    #             """
    #             :return:
    #             """
    #             from skfeature.function.information_theoretical_based.GA import Genetic_Algorithm
    #             from sklearn.ensemble import ExtraTreesRegressor  # 使用ExtraTrees 模型作为示范
    #
    #             # 选择模拟退火中评价特征子集的的损失函数
    #             from sklearn.metrics import mean_squared_error  # 回归问题我们使用MSE
    #
    #             clf = ExtraTreesRegressor(n_estimators=25)
    #             selector = Genetic_Algorithm(loss_func=mean_squared_error, estimator=clf,
    #                                          n_gen=10, n_pop=20, algorithm='NSGA2')
    #             # 在训练集中训练
    #             # GA.py中有具体每个参数的含义，此处不赘述
    #
    #             selector.fit(X_train=self.train_set, y_train=self.train_y, cv=5)
    #             # print(len(self.df.columns[selector.best_sol]))
    #             # print((self.df.columns[selector.best_sol]))
    #             dd = pd.DataFrame(columns=['特征名'])
    #             dd['特征名'] = self.df.columns[selector.best_sol]
    #             dd.index = range(len(dd))
    #             # print(selector.best_loss)
    #             return dd, self.method
    #
    #         ################################正则化算法
    #         def Lasso(self):
    #             """
    #             :return:
    #             """
    #             from sklearn.feature_selection import SelectFromModel
    #             from sklearn.linear_model import Lasso
    #             clf = Lasso(normalize=True, alpha=0.001)
    #             from sklearn.preprocessing import StandardScaler
    #             model = StandardScaler()
    #             model.fit(self.train_set)
    #             self.train_set = model.transform(self.train_set)
    #             # 在进行线性回归前，我们需要先对变量进行缩放操作，否则回归系数大小无法比较
    #             # alpha控制正则效果的大小，alpha越大，正则效果越强
    #             clf.fit(self.train_set, self.train_y)  # 在训练集上训练
    #             selector = SelectFromModel(clf, prefit=True, threshold=1e-5, max_features=self.N)
    #             # print(selector.get_support())
    #             # print(self.df.columns[selector.get_support()])
    #             dd = pd.DataFrame(columns=['特征名'])
    #             dd['特征名'] = self.df.columns[selector.get_support()]
    #             dd.index = range(len(dd))
    #             # print(selector.best_loss)
    #             return dd, self.method
    #
    #         def SVR(self):
    #             """
    #             :return:
    #             """
    #             from sklearn.feature_selection import SelectFromModel
    #             from sklearn import svm
    #             from sklearn.svm import LinearSVR
    #             from sklearn.preprocessing import StandardScaler
    #             model = StandardScaler()
    #             model.fit(self.train_set)
    #             standardized_train = model.transform(self.train_set)
    #             clf = LinearSVR(C=0.001, random_state=100)
    #             # C控制正则效果的大小，C越大，正则效果越弱
    #             clf.fit(standardized_train, self.train_y)
    #             selector = SelectFromModel(clf, prefit=True, threshold=1e-5, max_features=self.N)
    #             # print(selector.transform(self.train_set))
    #             # print(self.df.columns[selector.get_support()])
    #             dd = pd.DataFrame(columns=['特征名'])
    #             dd['特征名'] = self.df.columns[selector.get_support()]
    #             dd.index = range(len(dd))
    #             # print(selector.best_loss)
    #             return dd, self.method
    #
    #         def get_result(train_set, train_y, model, N):
    #             """
    #             :return:
    #             """
    #             df = train_set
    #             res = pd.DataFrame()
    #             if model == 'person':
    #                 ###################################1 Filter Methods 过滤法
    #                 #######################1.1 基于Univariate Filter Methods 单变量特征过滤
    #                 #####1.1.1 Pearson Correlation (regression problem) 皮尔森相关系数 (回归问题)
    #                 rf = Feature(train_set, train_set.values, train_y.values, 'person', N)
    #                 res, model_name = rf.person()
    #                 res['index'] = res.index
    #                 res['rank'] = N - res['index']
    #                 # model_list.append(model_name)
    #                 # res.to_excel(writer, sheet_name='%s' % model_name)
    #                 #####1.1.2 Distance Correlation (regression problem) 距离相关系数 (回归问题)
    #             elif model == 'Distance':
    #                 rf = Feature(train_set, train_set.values, train_y.values, 'Distance', N)
    #                 res, model_name = rf.distance_person()
    #                 res['index'] = res.index
    #                 res['rank'] = N - res['index']
    #                 # model_list.append(model_name)
    #                 # res.to_excel(writer, sheet_name='%s' % model_name)
    #                 # print(res)
    #             elif model == 'MIC':
    #                 ######1.1.3 Mutual Information (regression problem) 互信息 (回归问题)
    #                 rf = Feature(train_set, train_set.values, train_y.values, 'MIC', N)
    #                 res, model_name = rf.MIC()
    #                 res['index'] = res.index
    #                 res['rank'] = N - res['index']
    #                 # model_list.append(model_name)
    #                 # res.to_excel(writer, sheet_name='%s' % model_name)
    #                 # print(res)
    #             elif model == 'MRMR':
    #                 ########################1.2 Multivariate Filter Methods 多元特征过滤,越靠前的特征越重要
    #                 #######1.2.1 FCBF  算法有问题
    #                 # rf = Feature(train_set, train_set.values, train_y.values, 'FCBF', N)
    #                 # res = rf.FCBF()
    #                 # res.to_excel(writer, sheet_name='%s' % model_name)
    #                 # rf.plot_machine_learing(res)
    #                 ######1.2.2 MRMR
    #                 rf = Feature(train_set, train_set.values, train_y.values, 'MRMR', N)
    #                 res, model_name = rf.MRMR()
    #                 res['index'] = res.index
    #                 res['rank'] = N - res['index']
    #                 # model_list.append(model_name)
    #                 # res.to_excel(writer, sheet_name='%s' % model_name)
    #                 # #######1.2.3 ReliefF
    #             elif model == 'ReliefF':
    #                 rf = Feature(train_set, train_set.values, train_y.values, 'ReliefF', N)
    #                 res, model_name = rf.ReliefF()
    #                 res['index'] = res.index
    #                 res['rank'] = N - res['index']
    #                 # model_list.append(model_name)
    #                 # res.to_excel(writer, sheet_name='%s' % model_name)
    #             elif model == 'SPEC':
    #                 # # #######1.2.4 SPEC
    #                 rf = Feature(train_set, train_set.values, train_y.values, 'SPEC', N)
    #                 res, model_name = rf.SPEC()
    #                 res['index'] = res.index
    #                 res['rank'] = N - res['index']
    #                 # model_list.append(model_name)
    #                 # res.to_excel(writer, sheet_name='%s' % model_name)
    #                 ####################################2 封装方法
    #                 #########2.1确定性算法
    #                 #########2.1.1RFECV
    #             elif model == 'RFECV':
    #                 rf = Feature(train_set, train_set.values, train_y.values, 'RFECV', N)
    #                 res, model_name = rf.RFECV()
    #                 # model_list.append(model_name)
    #                 # res.to_excel(writer, sheet_name='%s' % model_name)
    #                 ########2.2Randomized Algorithms 随机方法
    #                 ########2.2.1 Simulated Annealing (SA) 基于模拟退火特征选择
    #             elif model == 'SA':
    #                 sa = Feature(train_set, train_set.values, train_y.values, 'SA', N)
    #                 res, model_name = sa.SA()
    #                 # model_list.append(model_name)
    #                 # res.to_excel(writer, sheet_name='%s' % model_name)
    #                 ########2.2.2Genetic Algorithm (GA) 基于基因算法特征选择
    #             elif model == 'GA':
    #                 ga = Feature(train_set, train_set.values, train_y.values, 'GA', N)
    #                 res, model_name = ga.GA()
    #                 # model_list.append(model_name)
    #                 # res.to_excel(writer, sheet_name='%s' % model_name)
    #                 #####################################3 嵌入式方法
    #                 #############################3.1基于正则化模型的方法
    #                 #########3.1.1 Lasso
    #             elif model == 'Lasso':
    #                 La = Feature(train_set, train_set.values, train_y.values, 'Lasso', N)
    #                 res, model_name = La.Lasso()
    #                 res['index'] = res.index
    #                 res['rank'] = N - res['index']
    #                 # model_list.append(model_name)
    #                 # res.to_excel(writer, sheet_name='%s' % model_name)
    #                 ########3.1.2 SVR
    #             elif model == 'SVR':
    #                 Log = Feature(train_set, train_set.values, train_y.values, 'SVR', N)
    #                 res, model_name = Log.SVR()
    #                 res['index'] = res.index
    #                 res['rank'] = N - res['index']
    #                 # model_list.append(model_name)
    #                 # res.to_excel(writer, sheet_name='%s' % model_name)
    #                 #############################3.2 基于树模型的特征选择
    #                 # RandomForestRegressor
    #             elif model == 'RandomForestRegressor':
    #                 print('*****************RandomForestRegressor开始')
    #                 rf = Feature(df, train_set, train_y, 'RandomForestRegressor', N)
    #                 res1, res, model_name = rf.RandomForest()
    #                 res['index'] = res.index
    #                 res['rank'] = N - res['index']
    #                 # model_list.append(model_name)
    #                 # rf.plot_machine_learing(res)
    #                 # res1.to_excel(writer, sheet_name='%s'%model_name)
    #             elif model == 'Adaboost':
    #                 print('*****************Adaboost开始')
    #                 # Adaboost
    #                 Ada = Feature(df, train_set, train_y, 'Adaboost', N)
    #                 res1, res, model_name = Ada.RandomForest()
    #                 res['index'] = res.index
    #                 res['rank'] = N - res['index']
    #                 # model_list.append(model_name)
    #                 # Ada.plot_machine_learing(res)
    #                 # res1.to_excel(writer, sheet_name='%s' % model_name)
    #             elif model == 'XGBboost':
    #                 print('*****************XGBoost开始')
    #                 # # XGBoost
    #                 XGB = Feature(df, train_set, train_y, 'XGBboost', N)
    #                 res1, res, model_name = XGB.RandomForest()
    #                 res['index'] = res.index
    #                 res['rank'] = N - res['index']
    #                 # model_list.append(model_name)
    #                 # XGB.plot_machine_learing(res)
    #                 # res1.to_excel(writer, sheet_name='%s' % model_name)
    #             elif model == 'CatBoost':
    #                 print('*****************CatBoost开始')
    #                 # CatBoost
    #                 Cat = Feature(df, train_set, train_y, 'CatBoost', N)
    #                 res1, res, model_name = Cat.RandomForest()
    #                 res['index'] = res.index
    #                 res['rank'] = N - res['index']
    #                 # model_list.append(model_name)
    #                 # Cat.plot_machine_learing(res)
    #                 # res1.to_excel(writer, sheet_name='%s' % model_name)
    #                 # print(model_list)
    #             # print(res_end)
    #             # writer.save()
    #             # writer.close()
    #             return res
    #     def lls_filter(filter_name, strainx, trainy, test_x, k, nnn, n_for):
    #         print('***********************%s开始*************************' % filter_name)
    #         print('***********************可能耗时较长，请耐心等待*************************')
    #         N = int(k)
    #         feature_list = []
    #         train_x2 = list(np.zeros((nnn, 1)))
    #         test_x2 = list(np.zeros((nnn, 1)))
    #         ########   排序部分
    #         for j in range(nnn):
    #             dd = pd.DataFrame(columns=['y1值', 'y值'])
    #             dd['y1值'] = trainy[j]
    #             dd['y值'] = trainy[j]
    #             train_y = dd.loc[:, 'y值']
    #             train_set = strainx[j]
    #             test_set = test_x[j]
    #             model = ['person', 'MIC', 'MRMR', 'ReliefF', 'SPEC', 'Lasso', 'SVR', 'RandomForestRegressor',
    #                      'Adaboost', 'XGBboost']#, 'CatBoost']
    #             res_end = pd.DataFrame()
    #             for model_in in model:
    #                 res = get_result(train_set, train_y, model_in, N)
    #                 res_end = res_end.append(res)
    #
    #             df_end = res_end[['rank', '特征名']].groupby(['特征名']).sum().reset_index()
    #             df_end.sort_values(by='rank', ascending=False, inplace=True)
    #             df_end.index = range(len(df_end))
    #             #########非排序部分
    #             #model_other = ['RFECV', 'SA', 'GA']
    #             # model_other = ['GA']
    #             # res_end1 = pd.DataFrame()
    #             # for model_out in model_other:
    #             #     res = get_result(train_set, train_y, model_out, N)
    #             #     res_end1 = res_end1.append(res)
    #             # res_end1 = res_end1.groupby(['特征名']).first().reset_index()
    #             # c = res_end1['特征名'].values
    #             # df_finall_end = df_end[df_end['特征名'].isin((c))]
    #             df_finall_end=df_end
    #             feature_name = (df_finall_end)['特征名'].values
    #             print('***********************************************************选择的特征数量个数为%s' % len(feature_name))
    #             # feature_list.append(feature_name)
    #             train_x2[j] = train_set[feature_name[:N]]
    #             test_x2[j] = pd.DataFrame(test_set,columns=train_set.columns)[feature_name[:N]]
    #         return train_x2, test_x2
    #     train_x2, test_x2 = lls_filter(filter_name, strainx, train_y, test_x, k, nnn, n_for)
    # elif filter_name[0:15]=='LLSfilterVoting':
    #     def lls_filter(filter_name, strainx, trainy, test_x, k, nnn, n_for):
    #         print('***********************%s开始*************************' % filter_name)
    #         print('***********************可能耗时较长，请耐心等待*************************')
    #         model = filter_name.split('_')[1:]
    #         N = int(k)
    #         feature_list = []
    #         train_x2 = list(np.zeros((nnn, 1)))
    #         test_x2 = list(np.zeros((nnn, 1)))
    #         ########   排序部分
    #         for j in range(nnn):
    #             dd = pd.DataFrame(columns=['y1值', 'y值'])
    #             dd['y1值'] = trainy[j]
    #             dd['y值'] = trainy[j]
    #             train_y = dd.loc[:, 'y值']
    #             train_set = strainx[j]
    #             test_set = test_x[j]
    #             # model = ['person', 'MIC', 'MRMR', 'ReliefF', 'SPEC', 'Lasso', 'SVR', 'RandomForestRegressor',
    #             #          'Adaboost', 'XGBboost']#, 'CatBoost']
    #             res_end = pd.DataFrame()
    #             for model_in in model:
    #                 res = get_result(train_set, train_y, model_in, N)
    #                 res_end = res_end.append(res)
    #
    #             df_end = res_end[['rank', '特征名']].groupby(['特征名']).sum().reset_index()
    #             df_end.sort_values(by='rank', ascending=False, inplace=True)
    #             df_end.index = range(len(df_end))
    #             #########非排序部分
    #             #model_other = ['RFECV', 'SA', 'GA']
    #             # model_other = ['GA']
    #             # res_end1 = pd.DataFrame()
    #             # for model_out in model_other:
    #             #     res = get_result(train_set, train_y, model_out, N)
    #             #     res_end1 = res_end1.append(res)
    #             # res_end1 = res_end1.groupby(['特征名']).first().reset_index()
    #             # c = res_end1['特征名'].values
    #             # df_finall_end = df_end[df_end['特征名'].isin((c))]
    #             df_finall_end=df_end
    #             feature_name = (df_finall_end)['特征名'].values
    #             print('***********************************************************选择的特征数量个数为%s' % len(feature_name))
    #             # feature_list.append(feature_name)
    #             train_x2[j] = train_set[feature_name[:N]]
    #             test_x2[j] = pd.DataFrame(test_set,columns=train_set.columns)[feature_name[:N]]
    #         return train_x2, test_x2
    #     train_x2, test_x2 = lls_filter(filter_name, strainx, train_y, test_x, k, nnn, n_for)
    # elif filter_name=="LLS_filter_PearsonXGBoost":
    #     def lls_filter(filter_name, strainx, trainy, test_x, k, nnn, n_for):
    #         print('***********************%s开始*************************' % filter_name)
    #         print('***********************可能耗时较长，请耐心等待*************************')
    #         N = int(k)
    #         feature_list = []
    #         train_x2 = list(np.zeros((nnn, 1)))
    #         test_x2 = list(np.zeros((nnn, 1)))
    #         ########   排序部分
    #         for j in range(nnn):
    #             dd = pd.DataFrame(columns=['y1值', 'y值'])
    #             dd['y1值'] = trainy[j]
    #             dd['y值'] = trainy[j]
    #             train_y = dd.loc[:, 'y值']
    #             train_set = strainx[j]
    #             test_set = test_x[j]
    #             model = ['person', 'XGBboost']
    #             res_end = pd.DataFrame()
    #             for model_in in model:
    #                 res = get_result(train_set, train_y, model_in, N)
    #                 res_end = res_end.append(res)
    #
    #             df_end = res_end[['rank', '特征名']].groupby(['特征名']).sum().reset_index()
    #             df_end.sort_values(by='rank', ascending=False, inplace=True)
    #             df_end.index = range(len(df_end))
    #             #########非排序部分
    #             #model_other = ['RFECV', 'SA', 'GA']
    #             # model_other = ['GA']
    #             # res_end1 = pd.DataFrame()
    #             # for model_out in model_other:
    #             #     res = get_result(train_set, train_y, model_out, N)
    #             #     res_end1 = res_end1.append(res)
    #             # res_end1 = res_end1.groupby(['特征名']).first().reset_index()
    #             # c = res_end1['特征名'].values
    #             # df_finall_end = df_end[df_end['特征名'].isin((c))]
    #             df_finall_end=df_end
    #             feature_name = (df_finall_end)['特征名'].values
    #             print('***********************************************************选择的特征数量个数为%s' % len(feature_name))
    #             # feature_list.append(feature_name)
    #             train_x2[j] = train_set[feature_name[:N]]
    #             test_x2[j] = pd.DataFrame(test_set,columns=train_set.columns)[feature_name[:N]]
    #         return train_x2, test_x2
    #     train_x2, test_x2 = lls_filter(filter_name, strainx, train_y, test_x, k, nnn, n_for)
    # elif filter_name == "HMfilter":
    #
    # def lls_filter(filter_name, strainx, trainy, test_x, k, nnn, n_for):
    #     print('***********************%s开始*************************' % filter_name)
    #     print('***********************可能耗时较长，请耐心等待*************************')
    #     N = int(k)
    #     feature_list = []
    #     train_x2 = list(np.zeros((nnn, 1)))
    #     test_x2 = list(np.zeros((nnn, 1)))
    #     ########   排序部分
    #     for j in range(nnn):
    #         dd = pd.DataFrame(columns=['y1值', 'y值'])
    #         dd['y1值'] = trainy[j]
    #         dd['y值'] = trainy[j]
    #         train_y = dd.loc[:, 'y值']
    #         train_set = strainx[j]
    #         test_set = test_x[j]
    #         model = ['person', 'XGBboost']
    #         res_end = pd.DataFrame()
    #         for model_in in model:
    #             res = get_result(train_set, train_y, model_in, N)
    #             res_end = res_end.append(res)
    #
    #         df_end = res_end[['rank', '特征名']].groupby(['特征名']).sum().reset_index()
    #         df_end.sort_values(by='rank', ascending=False, inplace=True)
    #         df_end.index = range(len(df_end))
    #         #########非排序部分
    #         # model_other = ['RFECV', 'SA', 'GA']
    #         # model_other = ['GA']
    #         # res_end1 = pd.DataFrame()
    #         # for model_out in model_other:
    #         #     res = get_result(train_set, train_y, model_out, N)
    #         #     res_end1 = res_end1.append(res)
    #         # res_end1 = res_end1.groupby(['特征名']).first().reset_index()
    #         # c = res_end1['特征名'].values
    #         # df_finall_end = df_end[df_end['特征名'].isin((c))]
    #         df_finall_end = df_end
    #         feature_name = (df_finall_end)['特征名'].values
    #         print('***********************************************************选择的特征数量个数为%s' % len(feature_name))
    #         # feature_list.append(feature_name)
    #         train_x2[j] = train_set[feature_name[:N]]
    #         test_x2[j] = pd.DataFrame(test_set, columns=train_set.columns)[feature_name[:N]]
    #     return train_x2, test_x2
    #
    #         def HM_filter(train_x, train_y, test_x, k, nnn, n_for):
    #             from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
    #             n = int(k)
    #             pic_len = n_for
    #
    #             def HanMing(tmpX1, tmpY1):
    #                 tmpX = tmpX1.reshape(-1, 1)
    #                 tmpY = tmpY1.reshape(-1, 1)
    #                 tmpX[tmpX >= np.mean(tmpX)] = 1
    #                 tmpX[tmpX < np.mean(tmpX)] = 0
    #                 hashX = tmpX.reshape(1, -1)[0]
    #                 tmpY[tmpY >= np.mean(tmpY)] = 1
    #                 tmpY[tmpY < np.mean(tmpY)] = 0
    #                 hashY = tmpY.reshape(1, -1)[0]
    #                 return sum(hashX == hashY) / hashX.shape[0]
    #
    #             def cal_aHash_Hanming2(x, freq, image_method=1):
    #                 # x = tmp_trainx.iloc[:,0]
    #                 # y = tmp_trainy
    #                 # freq = 6
    #                 if image_method == 1:
    #                     image_ = GramianAngularField(image_size=freq, method='summation')
    #                 elif image_method == 2:
    #                     image_ = GramianAngularField(image_size=freq, method='difference')
    #                 elif image_method == 3:
    #                     image_ = MarkovTransitionField(image_size=freq, n_bins=5)
    #                 elif image_method == 4:
    #                     image_ = RecurrencePlot()
    #                 image_X_fit = image_.fit_transform(np.transpose(np.array(x).reshape(-1, 1)))
    #                 # image_Y_fit = image_.fit_transform(np.transpose(np.array(y).reshape(-1,1)))
    #                 return image_X_fit
    #
    #             print('1-Start gaf_figure!')
    #             train_x1 = np.zeros(((nnn, train_x[0].shape[1], len(train_x[0]) - pic_len, 1, pic_len, pic_len)))
    #             train_y1 = np.zeros(((nnn, len(train_x[0]) - pic_len, 1, pic_len, pic_len)))
    #             for i in range(nnn):
    #                 train_y1[i] = list(
    #                     map(lambda x: cal_aHash_Hanming2(train_y[i][x:x + pic_len], pic_len, 1),
    #                         range(len(train_x[i]) - pic_len)))
    #                 for k in range(train_x[0].shape[1]):
    #                     train_x1[i][k] = list(
    #                         map(lambda x: cal_aHash_Hanming2(train_x[i].iloc[x:x + pic_len, k], pic_len, 1),
    #                             range(len(train_x[i]) - pic_len)))
    #
    #             def HM_index_select(train_X1, X_gasf, n, pic_len, dimention, train_x, test_x):
    #                 HM = []
    #                 for i in range(dimention):
    #                     for k in range(train_X1.shape[1]):
    #                         HM.append(
    #                             HanMing(train_X1.reshape(-1, 1, pic_len, pic_len)[k * i + i].reshape(pic_len, pic_len),
    #                                     X_gasf[k].reshape(pic_len, pic_len)))
    #                 HMmatrix = np.array(HM).reshape(dimention, len(train_x) - pic_len)
    #                 HMvalue = HMmatrix.mean(axis=1)
    #                 HMV = pd.DataFrame(HMvalue).sort_values(by=[0], ascending=False)
    #                 train_xchoose = train_x.iloc[:, HMV.index[0:n]]
    #                 test_x = pd.DataFrame(test_x, columns=train_x.columns)
    #                 test_xchoose = test_x.iloc[:, HMV.index[0:n]]
    #                 return train_xchoose, test_xchoose
    #
    #             print('2-Start HM_selecting!')
    #             train_x2 = list(map(lambda x:
    #                                 HM_index_select(train_x1[x], train_y1[x], n, pic_len, train_x[0].shape[1],
    #                                                 train_x[x],
    #                                                 test_x[x])[0], range(nnn)))
    #             test_x2 = list(map(lambda x:
    #                                HM_index_select(train_x1[x], train_y1[x], n, pic_len, train_x[0].shape[1],
    #                                                train_x[x],
    #                                                test_x[x])[1], range(nnn)))
    #             return train_x2, test_x2
    #
    #         # strainx1,test_x1=lls_filter(filter_name, strainx, train_y, test_x, 1000, nnn, n_for)
    #         train_x2, test_x2 = HM_filter(strainx, train_y, test_x, k, nnn, n_for)


def ewtsplit(x,level):
    import ewtpy
    tmpTrain = np.array(x)
    xx, mfb, boundaries = ewtpy.EWT1D(tmpTrain, N=level)
    xx1 = xx[:, 0:-1].sum(axis=1)
    xx2 = xx[:, -1]
    xx3 = np.subtract(tmpTrain, xx.sum(axis=1))
    return xx1,xx2,xx3

def splitX(train_x0, test_x0,level):
    # train_x0 = strainadd_x[0]
    # test_x0 = test_x[0]
    EWTx = pd.DataFrame()
    Xall = pd.concat([train_x0, pd.DataFrame(test_x0)], axis=0)
    print('EWT拆分X')
    xxx = np.apply_along_axis(ewtsplit, 0, np.array(Xall),level[0])
    for ee in range(xxx.shape[0]):
        EWTx = pd.concat([EWTx,pd.DataFrame(xxx[ee], columns=Xall.columns + '_EWT_' + str(ee + 1))],axis=1)
    train_x1 = EWTx.iloc[0:train_x0.shape[0], :]
    test_x1 = EWTx.iloc[-test_x0.shape[0]:, :]
    return train_x1, test_x1

def ewtsplit2(x,level):
    import ewtpy
    tmpTrain = np.array(x)
    xx, mfb, boundaries = ewtpy.EWT1D(tmpTrain, N=level)
    # xx1 = xx[:, 0:-1].sum(axis=1)
    # xx2 = xx[:, -1]
    xx1 = xx[:, 0:2].sum(axis=1)
    xx2 = xx[:, 2:].sum(axis=1)
    xx3 = np.subtract(tmpTrain, xx.sum(axis=1))
    return xx1,xx2,xx3

def splitX2(train_x0, test_x0,level):
    # train_x0 = strainadd_x[0]
    # test_x0 = test_x[0]
    EWTx = pd.DataFrame()
    Xall = pd.concat([train_x0, pd.DataFrame(test_x0)], axis=0)
    print('EWT拆分X')
    xxx = np.apply_along_axis(ewtsplit2, 0, np.array(Xall),level[0])
    for ee in range(xxx.shape[0]):
        EWTx = pd.concat([EWTx,pd.DataFrame(xxx[ee], columns=Xall.columns + '_EWT_' + str(ee + 1))],axis=1)
    train_x1 = EWTx.iloc[0:train_x0.shape[0], :]
    test_x1 = EWTx.iloc[-test_x0.shape[0]:, :]
    return train_x1, test_x1

def ewtsplit3(x,level):
    import ewtpy
    tmpTrain = np.array(x)
    xx, mfb, boundaries = ewtpy.EWT1D(tmpTrain, N=level)
    # xx1 = xx[:, 0:-1].sum(axis=1)
    # xx2 = xx[:, -1]
    xx1 = xx[:, 0:3].sum(axis=1)
    xx2 = xx[:, 3:].sum(axis=1)
    xx3 = np.subtract(tmpTrain, xx.sum(axis=1))
    return xx1,xx2,xx3

def splitX3(train_x0, test_x0,level):
    # train_x0 = strainadd_x[0]
    # test_x0 = test_x[0]
    EWTx = pd.DataFrame()
    Xall = pd.concat([train_x0, pd.DataFrame(test_x0)], axis=0)
    print('EWT拆分X')
    xxx = np.apply_along_axis(ewtsplit3, 0, np.array(Xall),level[0])
    for ee in range(xxx.shape[0]):
        EWTx = pd.concat([EWTx,pd.DataFrame(xxx[ee], columns=Xall.columns + '_EWT_' + str(ee + 1))],axis=1)
    train_x1 = EWTx.iloc[0:train_x0.shape[0], :]
    test_x1 = EWTx.iloc[-test_x0.shape[0]:, :]
    return train_x1, test_x1

def ewtsplit4(x,level):
    import ewtpy
    tmpTrain = np.array(x)
    xx, mfb, boundaries = ewtpy.EWT1D(tmpTrain, N=level)
    xx1 = xx.sum(axis=1)
    xx3 = np.subtract(tmpTrain, xx.sum(axis=1))
    return xx1,xx3

def splitX4(train_x0, test_x0,level):
    # train_x0 = strainadd_x[0]
    # test_x0 = test_x[0]
    EWTx = pd.DataFrame()
    Xall = pd.concat([train_x0, pd.DataFrame(test_x0)], axis=0)
    print('EWT拆分X_7')
    xxx = np.apply_along_axis(ewtsplit4, 0, np.array(Xall),level[0])
    for ee in range(xxx.shape[0]):
        EWTx = pd.concat([EWTx,pd.DataFrame(xxx[ee], columns=Xall.columns + '_EWT_' + str(ee + 1))],axis=1)
    train_x1 = EWTx.iloc[0:train_x0.shape[0], :]
    test_x1 = EWTx.iloc[-test_x0.shape[0]:, :]
    return train_x1, test_x1

##选取相关系数
def column_matrix_corr_matrix(Ytrain_roll, Xtrain_roll, k, bigger=True):
    Ytrain_roll = np.array(Ytrain_roll)
    y_hat = Ytrain_roll-Ytrain_roll.mean()
    x_hat = Xtrain_roll-Xtrain_roll.mean()
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

if __name__ =='__main__':
    import pickle
    filter_name='DTW_filter'
