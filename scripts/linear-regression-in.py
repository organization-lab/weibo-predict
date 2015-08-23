# -*- coding: utf-8 -*-
# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# misc function for i/o of basic data

import re
import math
import json

from sys import argv

#script, file1, file2 = argv

file1 = 'weibo_predict_data.txt'
file2 = 'predict_lr.txt'

filein = open(file1, encoding='utf-8')
#fileout = open('100user-list.txt', 'w')
fileout = open(file2, 'w')

#predict = open('predict/predict_1000_average.txt') 
#real = open('predict/predict_1000_real.txt') 

predict_model = open('lr_model.txt')
#predict_data = open('uid_average.txt')

def predict_average(predict_file, predict_data, fileout):
    """用 predict_data 的数据预测 predict_file

    i: predict_file, 暂时用 000 data
    p: 把 000 换成 average
    """

    t = re.compile('\t')
    predict_data = predict_data.readlines()

    dataset = {}

    for line in predict_data:
        uid, num_post, f,c,l = t.split(line)
        f = round(float(f))
        c = round(float(c))
        l = round(float(l))
        dataset[uid] = '{},{},{}'.format(f, c, l)
    
    i = 0
    for line in predict_file:
        uid, mid = t.split(line)[0], t.split(line)[1]

        if uid in dataset:
            fileout.write(uid + '\t' + mid + '\t' + dataset[uid] + '\n')
        else:
            fileout.write(uid + '\t' + mid + '\t' + '0,0,0' + '\n')

def get_data_and_write(filein, fileout):
    """用 readline 逐行读入数据并 unpack

    可用适当参数输出
    """
    t = re.compile('\t')
    for line in filein: # write number of users as demand in num_user   
        (uid, mid, time, forward_count,
        comment_count, like_count, content) = t.split(line)

        fileout.write(uid + '\t' + mid + '\t' + str(len(content)) + '\t')
        fileout.write(forward_count + '\t' + 
                      comment_count + '\t' + 
                      like_count + '\n')

def get_user_lists(filein, fileout, num_user = 50000):
    """读取 n 个用户的全部微博数据

    需要 uid 排序
    可用适当参数输出
    """
    t = re.compile('\w+')

    uid0 = ''
    num_post = 0
    llength = []
    f = []
    c = []
    l = []
    i = 0
    j = 0

    while i < num_user: # write number of users as demand in num_user 
        line = filein.readline()
        j += 1
        if line:
            uid, mid, length, forward_count, comment_count, like_count = re.findall(t,line)
        else:
            fileout.write(json.dumps([uid0, num_post, llength, f, c, l]))
            fileout.write('\n')
            print(j)
            print('num of users:', i)
            return
        if uid != uid0: # new uid
            if uid0:
                fileout.write(json.dumps([uid0, num_post, llength, f, c, l]))
                fileout.write('\n')
            else: #first user
                i -= 1
            i += 1
            if i % 1000 == 0:
                print(i)
            uid0 = uid
            num_post = 0
            llength = []
            f = []
            c = []
            l = []
            num_post += 1
            llength.append([int(length)])
            f.append(int(forward_count))
            c.append(int(comment_count))
            l.append(int(like_count))

            if i >= num_user:
                print(j)
                break
        else:
            num_post += 1
            llength.append([int(length)])
            f.append(int(forward_count))
            c.append(int(comment_count))
            l.append(int(like_count))
    print('num of users:', i)

def linear_reg(filein, fileout):
    """用 readline 逐行读入 json 数据并 linear regression

    可用适当参数输出
    """
    from sklearn import linear_model

    for line in filein: # write number of users as demand in num_user   
        (uid, num_post, llength, lf, lc, ll) = json.loads(line)

        linear_reg = []
        linear_reg.append(uid)
        linear_reg.append(num_post)
        #print(llength,lf,lc,ll)
        X = llength
        for y in lf, lc, ll:
            clf = linear_model.LinearRegression()
            clf.fit(X, y)

            #fileout.write(clf.coef_, clf.intercept_)
            r2 = clf.score(X,y)
            #print('r2', r2)
            linear_reg.append([clf.intercept_, clf.coef_[0], r2])
        fileout.write(json.dumps(linear_reg))
        fileout.write('\n')

def linear_reg_predict(filein, predict_model, fileout):
    """用 predict_model 的数据预测 predict_file

    i: predict_file
    p: 把 000 换成 average
    """
    t = re.compile('\t')

    dataset = {}

    for line in predict_model:
        uid, num_post, lf,lc,ll = json.loads(line)

        dataset[uid] = [lf,lc,ll]
    print(len(dataset))
    i = 0
    for line in filein:
        uid, mid, time, content = t.split(line)
        length = len(content)
        if uid in dataset:
            #print(dataset[uid])#
            x = int(length)
            f = round(dataset[uid][0][0] + dataset[uid][0][1] * x)
            c = round(dataset[uid][1][0] + dataset[uid][1][1] * x)
            l = round(dataset[uid][2][0] + dataset[uid][2][1] * x)
            if f < 0:
                f = 0
            if c < 0:
                c = 0
            if l < 0:
                l = 0
            linear_string = str(f) + ',' + str(c) + ',' + str(l) 
            fileout.write(uid + '\t' + mid + '\t' + linear_string + '\n')
        else:
            fileout.write(uid + '\t' + mid + '\t' + '0,0,0' + '\n')

#linear_reg(filein,fileout)
linear_reg_predict(filein, predict_model, fileout)
#get_data_and_write(filein, fileout)
#get_user_lists(filein, fileout)
#linear_reg(open('100user-list.txt'),fileout2)