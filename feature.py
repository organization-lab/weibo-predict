# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# get features for weibo data
# -*- coding: utf-8 -*-

import scipy.io as sio
import re
import scipy as sp
import numpy as np
import json

def categorize_weibo(filein):
    """分类输出微博信息

    按三个数量分类源文件, 暂不做分词
    """
    t = re.compile('\t')
    time_sep = re.compile('-')

    #init io
    output_forward = []
    output_comment = []
    output_like = []
    for i in range(0, 7):
        output_forward.append(open(('forward' + str(i) + '.txt'), 'w', encoding='utf-8'))
        output_comment.append(open(('comment' + str(i) + '.txt'), 'w', encoding='utf-8'))
        output_like.append(open(('like' + str(i) + '.txt'), 'w', encoding='utf-8'))

    for line in filein: # write number of users as demand in num_user   
        (uid, mid, time, forward_count,
        comment_count, like_count, content) = json.loads(line)

        forward_class = classify(forward_count)
        comment_class = classify(comment_count)
        like_class = classify(like_count)

        output_forward[forward_class].write(line)
        output_comment[comment_class].write(line)
        output_like[like_class].write(line)       

def classify(number):
    """分成七类, 防止单个类别样本量太少, 不够总结 feature

    """
    if number == 0:
        return 0
    elif number < 6: 
        return 1
    elif number < 11:
        return 2
    elif number < 21:
        return 3
    elif number < 51:
        return 4
    elif number < 101:
        return 5
    else:
        return 6

def categorize(count):
    if count == 0 :
        return 0
    elif count < 5:
        return 1
    elif count < 11:
        return 2
    elif count <21:
        return 3
    elif count < 51:
        return 4
    elif count < 101:
        return 5
    else: 
        return 6

def y_and_weight(filein, filein_name):
    """输出 y 和 weight
    """
    t = re.compile('\t')
    y_like = []
    y_comment = []
    y_forward = []
    weight = []

    #逐行分词并添加到X,y
    linenum = 0
    for line in filein: # write number of users as demand in num_user   
        (uid, mid, day, forward_count,
        comment_count, like_count, content) = json.loads(line)
        
        forward_count, comment_count, like_count = int(forward_count), int(comment_count), int(like_count)
        
        y_forward.append(forward_count)
        y_comment.append(comment_count)
        y_like.append(like_count)

        weight_i = forward_count + comment_count + like_count + 1
        if weight_i > 100:
            weight_i = 100
        weight.append(weight_i)

    # io to file
    sio.savemat(filein_name[:-4] + 'y_like.mat', {'y':y_like})
    sio.savemat(filein_name[:-4] + 'y_comment.mat', {'y':y_comment})
    sio.savemat(filein_name[:-4] + 'y_forward.mat', {'y':y_forward})
    sio.savemat(filein_name[:-4] + 'weight.mat', {'weight':weight})

def uid_average(filein, filein_name):
    """输出 uid average
    """
    t = re.compile('\t')

    predict_data = open('uid_average.txt')
    predict_data = predict_data.readlines()

    dataset = {}

    for line in predict_data:
        uid, num_post, f,c,l = t.split(line)
        f = (float(f))
        c = (float(c))
        l = (float(l))
        dataset[uid] = [f, c, l] # 暂存平均值

    uid_features = []

    #逐行
    linenum = 0
    for line in filein: # write number of users as demand in num_user   
        (uid, mid, day, forward_count,
        comment_count, like_count, content) = json.loads(line)
        uid_features.append(dataset[uid])
        
        ''' small check
        linenum += 1
        if linenum == 10:
            print(uid_features)
            break
        '''

    # io to file
    sio.savemat(filein_name[:-4] + 'uid_ave.mat', {'X':uid_features})

def matrix_shape(matrices):
    for i in matrices:
        print((sio.loadmat(i)['X']).shape)

def combineX():
    matrix_list = ['12uid_ave.mat']

    combined_list = []
    combined_list.append(sio.loadmat('0908-12y100.mat')['y'])

    for i in matrix_list:
        combined_list.append(sio.loadmat(i)['X'])

    for i in combined_list:
        print(type(i))
        print(i.shape)
    
    combined_list = np.concatenate(combined_list, axis=1)
    #combined_list = sp.sparse.hstack(combined_list)
    print(combined_list.shape)
    sio.savemat('0908-10feature-100', {'X':combined_list})


if __name__ == '__main__':
    filein = open('weibo_train_data_cut.txt', encoding='utf-8')
    #categorize_weibo(filein)
    uid_average(filein, 'weibo_train_data_cut.txt')
    #y_and_weight(filein, 'weibo_train_data_cut.txt')

    #matrix_shape(['log_reg_modelX001-7-10.mat'])
    #matrix_shape(['7-10y001.mat'])
    #matrix_shape(['7-10uid_ave.mat'])
    #uid_average(filein, '12.txt')
    #combineX()
