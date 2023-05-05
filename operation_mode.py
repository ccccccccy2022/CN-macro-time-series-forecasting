# -*- coding = utf-8 -*-
# @time:2021/11/23 14:44
# Author:Tjd_T
# @File:operation_mode.py
# @Software:PyCharm
import pickle
from get_result import get_result
import os


def multi_mode(need_arg):
    # 准备参数
    folder = need_arg['folder']
    data_save = need_arg['data_save']
    total_done = need_arg['total_done']
    # 继续准备
    print("Using output folder {}".format(folder))
    input_folder = folder + '/input/'
    output_folder = folder + '/output/'
    model_save = folder + '/save_model/'
    if not os.path.exists(model_save):
        os.mkdir(model_save)
    #  开始跑
    dict_args = dict()
    dict_args['save_model'] = model_save
    dict_args['data_save'] = data_save
    dict_args['trans_former_save'] = model_save
    dict_args['total_done'] = total_done

    res = get_result(input_folder, dict_args)

    with open(os.path.join(output_folder, folder.split('/')[-2] + '_res.pkl'), 'wb') as FD:
        pickle.dump(res[0], FD)


def along_mode(need_arg):
    # 准备参数
    folder = need_arg['folder']
    data_save = need_arg['data_save']
    total_done = need_arg['total_done']
    # 继续准备
    print("Using output folder {}".format(folder))
    input_folder = folder + '/input/'
    output_folder = folder + '/output/'
    model_save = folder + '/save_model/'
    if not os.path.exists(model_save):
        os.mkdir(model_save)
    # 开始跑
    with open(input_folder + 'parameter.pkl', 'rb') as fd:
        dict_args = pickle.load(fd)
    dict_args['total_done'] = total_done
    dict_args['save_model'] = model_save
    dict_args['data_save'] = data_save
    dict_args['space']['trans_former_save'] = model_save
    dict_args['space']['n_for'] = dict_args['n_for']
    dict_args['data_save'] = data_save + '/dataa.pkl'

    from alone_model_train_zh import get_alone_task
    result = get_alone_task(input_folder, dict_args)

    with open(os.path.join(output_folder, folder.split('/')[-2] + '_res.pkl'), 'wb') as FD:
        pickle.dump(result, FD)


def test_mode(need_arg):
    # 准备参数
    folder = need_arg['folder']
    data_save = need_arg['data_save']
    total_done = need_arg['total_done']
    # 继续准备
    print("Using output folder {}".format(folder))
    input_folder = folder + '/input/'
    output_folder = folder + '/output/'
    model_save = folder + '/save_model/'
    if not os.path.exists(model_save):
        os.mkdir(model_save)
    #  开始跑
    dict_args = dict()
    dict_args['save_model'] = model_save
    dict_args['data_save'] = data_save
    dict_args['trans_former_save'] = model_save
    dict_args['total_done'] = total_done
####
    res = get_result(input_folder, dict_args, best_param='do_one')

    # TODO 20220715 res[0]-->res
    with open(os.path.join(output_folder, folder.split('/')[-2] + '_res.pkl'), 'wb') as FD:
        pickle.dump(res, FD)
    with open(input_folder + os.listdir(input_folder)[0], 'rb') as fd:
        dict_argss = pickle.load(fd)
    framelist = dict_argss['framelist']
    for f,frame0 in enumerate(framelist):
        res2 = get_result(input_folder, dict_args,best_param='do_one',setting=frame0)
        with open(os.path.join(output_folder, folder.split('/')[-2]+'_'+str(frame0).replace(':',"").replace('(',"").replace(')',"").replace('*',"").replace('/',"") + '_res.pkl'), 'wb') as FD:
            pickle.dump(res2, FD)
if __name__ == '__main__':
    pass
