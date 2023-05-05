# -*- coding = utf-8 -*-
# @time:2022/5/30 13:46
# Author:Tjd_T
# @File:get_result.py
# @Software:PyCharm
import os
import pickle
import numpy as np
import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import optuna

# -------------------------------拼合部分代码开始----------------------------------
def add_parameter(dict_args, arg_dict, input_folder, muti_data_save) -> dict:
    """
    此函数用来往参数字典添加新的参数，保持新旧文件兼容性
    @param dict_args:
    @param input_folder:
    @param n_for:
    @return:
    """
    data_save = arg_dict['data_save']
    for key in arg_dict.keys():
        if key == 'trans_former_save':
            dict_args['space'][key] = arg_dict[key]
        else:
            dict_args[key] = arg_dict[key]
    dict_args['space']['informer_switch'] = False  # 需删除
    # dict_args['space']['epochs']=1 #需删除
    dict_args['data_save'] = data_save + '/' + muti_data_save
    n_for = int(dict_args['n_for'])
    try:
        if dict_args['space']['output_model']:  # 添加是否保存模型
            for current_roll in input_folder.split('/'):  # 开始检查当前滚动期是否是最后n_for期
                if 'roll_' in current_roll:
                    roll_id = int(current_roll.split('_')[1])
                    if roll_id > dict_args['total_done'] - n_for:
                        dict_args['space']['output_model'] = True
                    else:
                        dict_args['space']['output_model'] = False
    except KeyError as er:
        dict_args['space']['output_model'] = False
    try:
        if dict_args['space']['model_seed']:
            pass  # 如果有这个参数，逻辑走这，不做改动
    except KeyError as er:
        dict_args['space']['model_seed'] = 10
    try:
        if dict_args['space']['cnn_layer']:
            pass  # 如果有这个参数，逻辑走这，不做改动
    except KeyError as er:
        dict_args['space']['cnn_layer'] = False
    return dict_args


def val_standard(method, x, x2):
    """
    对y进行标准化的函数
    @param method:
    @param x:
    @param x2:
    @return:
    """
    y_scaler = None
    if method == 'z-score':
        y_scaler = sklearn.preprocessing.StandardScaler().fit(x.reshape(-1, 1))
        ans = y_scaler.transform(x2.reshape(-1, 1)).flatten()
    elif method == '0-1':
        y_scaler = sklearn.preprocessing.MinMaxScaler().fit(x.reshape(-1, 1))
        ans = y_scaler.transform(x2.reshape(-1, 1)).flatten()
    elif method == 'MaxAbsScaler':
        y_scaler = sklearn.preprocessing.MaxAbsScaler().fit(x.reshape(-1, 1))
        ans = y_scaler.transform(x2.reshape(-1, 1)).flatten()
    else:
        ans = x2
    return ans


def date_to_sep(a):
    x = [a.year, a.month, a.day]
    x_ans = pd.DataFrame(x).T
    x_ans.columns = ['year', 'month', 'day']
    return x_ans


def get_time(dict_args):
    """
    将时间编码为年月日，将空缺的时间补齐
    :param dict_args:
    :return:
    """
    n_for = dict_args['n_for']
    timeembedding_start = dict_args['n_in'] + dict_args['n_ori_start']
    timeembedding_end = timeembedding_start + dict_args['length'] + dict_args['n_start'] - dict_args['n_ori_start'] - 1
    timeindex = dict_args['y'].loc[timeembedding_start: timeembedding_end, 'time']
    timeindex_extend = None
    # 判断是周度还是月度
    if (dict_args['y'].loc[timeembedding_start + 1, 'time'] - dict_args['y'].loc[timeembedding_start, 'time']).days > 7:
        timeindex_extend = pd.date_range(timeindex.iloc[-1].strftime('%Y-%m-%d'),
                                         # error2 不是(dict_args['y'])的最后一个，而是这个训练集的最后一个时间点
                                         periods=n_for + 1, freq='M',
                                         closed='right')  # error3 数据长度不对，period的n_for+1了，要后边截取【1：】
    else:
        timeindex_extend = pd.date_range(timeindex.iloc[-1].strftime('%Y-%m-%d'),
                                         periods=n_for + 1, freq='W-SUN', closed='right')
    # 进行拆分
    time_y_pd = list(map(
        lambda x: pd.DataFrame(date_to_sep(timeindex.iloc[x]).values, index=[x],
                               columns=date_to_sep(timeindex.iloc[x]).columns),
        range(len(timeindex))))
    time_extend_pd = list(map(
        lambda x: pd.DataFrame(date_to_sep(timeindex_extend[x]).values, index=[x],
                               columns=date_to_sep(timeindex.iloc[x]).columns),
        range(len(timeindex_extend))))

    return time_y_pd, time_extend_pd


def concate_mutil_data(input_folder, mutil_data, arg_dict):
    # 盛放x的变量，find是为了做特征，feed是进模型
    strainx_feed_list, strainx_find_list, stestx_feed_list = [], [], []
    # 盛放y的变量
    train_ystd_list, train_ystd_find_list, train_ynstd_feed_list, test_ynstd_feed_list = [], [], [], []
    train_ylag_feed_list, test_ylag_feed_list = [], []
    # 盛放时间编码的变量
    train_time_code_list, test_time_code_list = [], []
    for mutil_data_save in mutil_data:
        with open(input_folder + 'parameter_{}'.format(mutil_data_save[6:]), 'rb') as fd:
            dict_args = pickle.load(fd)
        n_for = int(dict_args['n_for'])
        length = dict_args['length']
        # 添加参数
        dict_args = add_parameter(dict_args, arg_dict, input_folder, mutil_data_save)
        # 划分训练集、测试集
        from split_train_test import get_train_test
        train_y, strainx, test_y, stestx, train_x = get_train_test(dict_args)
        # 训练集x
        strainx_find_list.append(strainx[0].iloc[:length, :])
        strainx_feed_list.append(strainx[0])
        # 测试集x
        stestx_feed_list.append(stestx[0])
        # 未标准化的y保存
        train_ynstd_feed_list.append(pd.DataFrame(train_y[0]))
        test_ynstd_feed_list.append(pd.DataFrame(test_y[0]))
        # 训练集y标准化
        train_ystd_list.append(
            pd.DataFrame(val_standard(dict_args['space']['y_normalize'], train_y[0][:dict_args['length']], train_y[0])))
        train_ystd_find_list.append(train_ystd_list[-1].iloc[:length, :])
        # 滞后的y
        train_y_lag = train_ystd_list[-1].shift(n_for).fillna(0)
        train_ylag_feed_list.append(train_y_lag)
        test_y_lag = pd.DataFrame(train_ystd_list[-1][-n_for:])
        test_ylag_feed_list.append(test_y_lag)
        # 处理时间编码
        train_time_pd, test_time_pd = get_time(dict_args)  # 得到扩展的时间
        train_time_code_list.append(pd.concat(train_time_pd, axis=0))
        test_time_code_list.append(pd.concat(test_time_pd, axis=0))
    # 拼合x
    strainx_find_con_pd = pd.concat([strainx_find_list[0] for x in range(len(strainx_find_list))], axis=0, join='inner', ignore_index=True)
    strainx_feed_con_pd = pd.concat([strainx_feed_list[0] for x in range(len(strainx_feed_list))], axis=0, join='inner', ignore_index=True)
    stestx_feed_con_pd = pd.concat([stestx_feed_list[0] for x in range(len(stestx_feed_list))], axis=0, join='inner', ignore_index=True)
    # 拼合y
    train_ystd_feed_con_pd = pd.concat(train_ystd_list, axis=0, ignore_index=True)
    train_ystd_find_con_pd = pd.concat(train_ystd_find_list, axis=0, ignore_index=True)
    train_ynstd_feed_con_pd = pd.concat(train_ynstd_feed_list, axis=1)

    test_ynstd_feed_con_pd = pd.concat(test_ynstd_feed_list, axis=0, ignore_index=True)

    # 拼合时间
    train_time_code_con_pd = pd.concat(train_time_code_list, axis=0, ignore_index=True)
    test_time_code_con_pd = pd.concat(test_time_code_list, axis=0, ignore_index=True)

    #  train add lag
    train_ylag_con_feed_list = [pd.concat(train_ylag_feed_list, axis=1)] * len(mutil_data)
    train_ylag_con_feed_list = pd.concat(train_ylag_con_feed_list, axis=0, ignore_index=True)
    train_ylag_con_feed_list.columns = dict_args['Y_name']
    #  testTYC add lag
    test_ylag_con_feed_list = [pd.concat(test_ylag_feed_list, axis=1)] * len(mutil_data)
    test_ylag_con_feed_list = pd.concat(test_ylag_con_feed_list, axis=0, ignore_index=True)
    test_ylag_con_feed_list.columns = dict_args['Y_name']

    try:
        from feature_filteringnoray2 import Filter
    # with open('train_x.pkl','wb') as pk_file:
    #     pickle.dump([strainx_find_con_pd],pk_file)
    # with open('train_y.pkl', 'wb') as pk_file:
    #     pickle.dump([train_ystd_find_con_pd.values.flatten()], pk_file)
    # with open('test_x.pkl', 'wb') as pk_file:
    #     pickle.dump([stestx_feed_con_pd], pk_file)
    # with open('dict_args.pkl', 'wb') as pk_file:
    #     pickle.dump(dict_args, pk_file)
        filter_obj = Filter([strainx_find_con_pd],
                            [train_ystd_find_con_pd.values.flatten()],
                            [stestx_feed_con_pd], dict_args['kfilter'], dict_args['kfilter'],framelist=dict_args['framelist'])
        train_x4, test_x4 = filter_obj.run(dict_args['filter_name'], len(dict_args['Y_name']))
    except:
        from feature_filteringnoray import Filter
        filter_obj = Filter([strainx_find_con_pd],
                            [train_ystd_find_con_pd.values.flatten()],
                            [stestx_feed_con_pd], dict_args['kfilter'], dict_args['kfilter'])
        train_x4, test_x4 = filter_obj.run(dict_args['filter_name'], len(dict_args['Y_name']))
    # # # #TODO 20220715
    # # # # import pickle
    # with open('train_x4.pkl','wb') as pk_file:
    #     pickle.dump(train_x4,pk_file)
    # with open('test_x4.pkl','wb') as pk_file:
    #     pickle.dump(test_x4,pk_file)
    # with open('train_x4.pkl','rb') as pk_file:
    #     train_x4 = pickle.load(pk_file)
    # with open('test_x4.pkl','rb') as pk_file:
    #     test_x4 = pickle.load(pk_file)

    # 修改20220110
    try:
        strainx_muti_feed = pd.concat(
            [train_ylag_con_feed_list, strainx_feed_con_pd[train_x4[0].columns], train_time_code_con_pd], axis=1)
    except:
        sfcp = [pd.DataFrame(strainx_feed_con_pd[x.split('^^')[1]])  if pd.DataFrame(strainx_feed_con_pd[x.split('^^')[1]]).shape[1]==1 else pd.DataFrame(strainx_feed_con_pd[x.split('^^')[1]].iloc[:,0]) for x in train_x4[0].columns]
        sfcp2=pd.concat(sfcp,axis=1)
        sfcp2.columns=[x for x in train_x4[0].columns]
        strainx_muti_feed = pd.concat(
            [train_ylag_con_feed_list, sfcp2, train_time_code_con_pd], axis=1)
    stestx_muti_feed = pd.concat([test_ylag_con_feed_list, test_x4[0], test_time_code_con_pd], axis=1)
    return strainx_muti_feed, stestx_muti_feed, train_ystd_feed_con_pd, train_ynstd_feed_con_pd, \
           test_ynstd_feed_con_pd, dict_args


# -------------------------------拼合部分代码结束----------------------------------

# -------------------------------添加类别部分代码开始----------------------------------
def category_x(x, label_encoder):
    x_cate = list()
    for i in range(len(x)):
        x_mean = np.mean(x[i], axis=1)
        #TODO 20220715
        if x_mean.max()==x_mean.min():
            x_mean_minmax = (x_mean - x_mean.min()) / (x_mean.max() - x_mean.min()+1)
        else:
            x_mean_minmax = (x_mean - x_mean.min()) / (x_mean.max() - x_mean.min())
        x_mean_minmax = np.round(x_mean_minmax * 100, 0)
        x_cate.append(label_encoder.transform(x_mean_minmax))
    return x_cate


def category_y(y, label_encoder):
    y_cate = list()
    for i in range(len(y)):
        y_minmax = (y[i] - y[i].min()) / (y[i].max() - y[i].min())
        y_minmax = np.round(100 * y_minmax, 0)
        y_cate.append(label_encoder.transform(y_minmax))
    return y_cate


def add_category(train_x, train_y, test_x, test_y, args):
    label_encoder = LabelEncoder()
    label_encoder.fit([x for x in range(101)])
    if len(test_y[0])==1:
        train_ans, test_ans, count_list = train_x, test_x, []
        args['categray_counts_list'] = count_list
        args['categray_num'] = 0
        args['feature_name'] = train_x[0].columns.tolist()
        args['cnn_nums'] = train_ans[0].shape[1] - 3 - 0
    else:
        # 对y分类编码
        train_y_cate = category_y(train_y, label_encoder)
        # 若testy是空的，或者长度不够
        if len(test_y[0]) < args['y_num'] * args['n_for']:
            tem_test_y = []
            for i in range(len(train_y)):
                tem = train_y[i][-int(args['y_num'] * args['n_for']):]
                tem_test_y.append(tem)
            test_y = tem_test_y
        # 编码
        test_y_cate = category_y(test_y, label_encoder)
        train_x_cate = category_x(train_x, label_encoder)
        test_x_cate = category_x(test_x, label_encoder)
        # 拼合
        train_ans, test_ans, count_list = [], [], []
        for i in range(len(train_y)):
            train_cate = np.vstack([train_y_cate[i], train_x_cate[i]]).T
            train_cate = pd.DataFrame(train_cate, index=train_x[i].index)
            train_ans.append(pd.concat([train_x[i], train_cate], axis=1))
            test_y_len = len(test_y[i])
            test_cate = np.vstack([test_y_cate[i], test_x_cate[i][-test_y_len:]]).T
            test_cate = pd.DataFrame(test_cate, index=test_x[i].index[:len(test_y[i])])
            test_ans.append(pd.concat([test_x[i], test_cate], axis=1))
            count_list.append(max(101, len(np.unique(train_x_cate[i]))))
            count_list.append(max(101, len(np.unique(train_y_cate[i]))))
        # 至此，数据准备部分完成，为后续计算添加参数
        args['categray_counts_list'] = count_list
        args['categray_num'] = train_cate.shape[-1]
        args['feature_name'] = train_x[0].columns.tolist()
        args['cnn_nums'] = train_ans[0].shape[1] - 3 - train_cate.shape[-1]
    return train_ans, test_ans, args


# -------------------------------添加类别部分代码结束----------------------------------


def prepare_data(input_folder, arg_dict,sett=None):
    data_save = arg_dict['data_save']
    mutil_data = [x for x in os.listdir(data_save) if x[0:4] == 'data']
    mutil_data.sort(key=lambda x: int(x.split('_')[1]))  # 解决Y_name读取 不一致的问题
    if os.path.exists('/'.join(input_folder.split('/')[:-2])+'save_model/model/data_and_parameter.pkl'):
        with open(r'/'.join(input_folder.split('/')[:-2]) + 'save_model/model/data_and_parameter.pkl','rb') as pk_file:
            ans_list=pickle.load(pk_file)
        if sett is None:
            pass
        else:
            namelist=[x for x in ans_list[2][0].columns[:-3] if sett in x]
            ans_list[2][0][namelist]=0
            ans_list[4][0][namelist]=0
    else:
        # 得到拼合的数据
        strainx, stestx, train_ystd, train_ynstd, test_ynstd, args = concate_mutil_data(input_folder, mutil_data,
                                                                                        arg_dict)

        # 添加分类信息
        k = strainx.shape[1]
        args['space']['n_for'] = args['n_for']
        strainx_add_category, stestx_add_category, args_dict = add_category([strainx], [train_ystd.values.flatten()],
                                                                        [stestx],
                                                                        [test_ynstd.values.flatten()],
                                                                        args['space'])
        ans_list = [k, args_dict, strainx_add_category, train_ystd, stestx_add_category, test_ynstd, train_ynstd]
    return ans_list


def get_best_params(input_folder, arg_dict):
    data = prepare_data(input_folder, arg_dict)
    data0_columns = data[2][0].columns.to_list()[:-5]
    data0_columns = sorted(data0_columns)
    category = data[2][0].iloc[:, -5:]
    if data[2][0][data0_columns].shape[1]==data0_columns:
        data[2][0] = pd.concat([data[2][0][data0_columns], category], axis=1)
    else:
        pass

    def objective(trial):
        data[1]['lr_schedule'] = dict()
        data[1]['lr_schedule']['name'] = None
        data[1]['optim'] = dict()
        data[1]['optim']['name'] = trial.suggest_categorical("optimizer", ["Adam", "Adadelta", "Adagrad"])
        data[1]['lr']= trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        data[1]['max_seq_len'] = trial.suggest_int("max_seq_len", 10, 40, step=5)
        data[1]['batch_size'] = trial.suggest_int("batch_size", 5, 100, step=16)
        t_size=trial.suggest_float("t_size", 0.7, 0.8, log=False)
        from model_tools.model_train_frame import ModelTrain
        train_exe = ModelTrain(data[0], data[1])
        try:
            ans, loss = train_exe.train_method(data, train_size=t_size)
            return loss
        except:
            raise optuna.exceptions.TrialPruned()


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    best_param=study.best_params
    print("Best loss: {}".format(study.best_value))
    if not os.path.exists(arg_dict['data_save'] + '/best_param/'):
        os.mkdir(arg_dict['data_save'] + '/best_param/')
    with open(arg_dict['data_save']+'/best_param/'+arg_dict['save_model'].split('/')[-4]+'_best_param.pkl', 'wb') as pk_file:
        pickle.dump(best_param,pk_file)
    return best_param,study

def get_result(input_folder, arg_dict,best_param='do_one',setting=None):
    if setting is None:
        data = prepare_data(input_folder, arg_dict)#得在predict_data里边把值变了，变谁外边给参数
    else:
        data = prepare_data(input_folder, arg_dict,sett=setting)#得在predict_data里边把值变了，变谁外边给参数
    # 临时添加
    data[1]['optim'] = dict()
    data[1]['lr_schedule'] = dict()
    data[1]['lr_schedule']['name'] = None
    # data[1]['epochs'] = 100
    # data[1]['model_seed'] = 50
    data[1]['seed_flag'] = True
    # data[1]['lr']=0.01
    data0_columns = data[2][0].columns.to_list()[:-5]
    # data0_columns = sorted(data0_columns)
    category = data[2][0].iloc[:, -5:]
    # data[2][0] = pd.concat([data[2][0][data0_columns], category], axis=1)
    from model_tools.model_train_frame import ModelTrain
    if best_param is 'open':
        best_params,s=get_best_params(input_folder,arg_dict)
        data[1]['optim']['name'] = best_params['optimizer']
        data[1]['lr'] = best_params['lr']
        data[1]['batch_size'] =best_params['batch_size']
        t_size=best_params['t_size']
        data[1]['max_seq_len']=best_params['max_seq_len']
    elif best_param is 'do_one':
        if os.path.exists(arg_dict['data_save'] + '/best_param/'):
            with open(arg_dict['data_save']+'/best_param/'+os.listdir(arg_dict['data_save'] + '/best_param/')[0],'rb') as pk_file:
                best_params=pickle.load(pk_file)
        else:
            best_params, s = get_best_params(input_folder, arg_dict)
        data[1]['optim']['name'] = best_params['optimizer']
        data[1]['lr'] = best_params['lr']
        data[1]['batch_size'] =best_params['batch_size']
        data[1]['max_seq_len']=best_params['max_seq_len']
        t_size=0.8#best_params['t_size']
    else:
        data[1]['optim']['name'] = 'Ad'
        t_size = 0.8
    train_exe = ModelTrain(data[0], data[1])
    result,_ = train_exe.train_method(data, t_size)

    return result


if __name__ == '__mian__':
    pass
