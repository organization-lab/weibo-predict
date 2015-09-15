# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# cut for weibo data
# -*- coding: utf-8 -*-

import re
import jieba
import json
import math
from operator import itemgetter
from sys import argv

script, filein_name = argv

#filein = open(filein_name, encoding='utf-8')

def get_data_and_write(filein, fileout):
    """用 readline 逐行读入数据并 unpack

    可用适当参数输出
    """
    t = re.compile('\t')
    time_sep = re.compile('-')

    temp = []
    for line in filein: # write number of users as demand in num_user   
        (uid, mid, time, forward_count,
        comment_count, like_count, content) = t.split(line)
        #yyyy, mm, dd = time_sep.split(time)
        #print(yyyy, mm, dd)
        forward_count = int(forward_count)
        comment_count = int(comment_count)
        like_count = int(like_count)
        cut_list = jieba.lcut(content)
        #print(cut_list)
        if forward_count == 0 and comment_count == 0 and like_count == 0:
            fileout000.write('000 '+ ' '.join(cut_list))
        else:             
            if forward_count != 0:
                fileout100.write('100 '+ ' '.join(cut_list))
            if comment_count != 0:
                fileout010.write('010 '+ ' '.join(cut_list))
            if like_count != 0:
                fileout001.write('001 '+ ' '.join(cut_list))
#get_data_and_write(filein, fileout)

def init_dict(filein):
    """从已有的分词文件建立dict

    目前删去了只有一次的词。
    """
    d = {}
    t = re.compile(' ')
    for line in filein:
        words = t.split(line)
        for word in words:
            if word in d:
                d[word] += 1
            else:
                d[word] = 1

    d1 = {}
    d2 = 0
    d3 = 0
    d4 = 0
    d5 = 0
    d10 = 0
    for key in d:
        if d[key] >= 2:
            d1[key] = d[key]
            d2+=1  
    #print('2,3,4,5,10', d2,d3,d4,d5,d10)
    d1 = sorted(d1.items(), key=lambda d1:d1[1], reverse=True)
    fileout = open(filein_name + 'dict.txt', 'w', encoding='utf-8')
    fileout.write(json.dumps(d1))

def load_dict(filein):
    list_dict = json.loads(filein.readline())
    #print(list_dict)
    for key in list_dict[:10]:
        print(key[0], key[1])
    out_list = list_dict[:10000]
    fileout = open(filein_name + '10000.txt', 'w', encoding='utf-8')
    fileout.write(json.dumps(out_list))

def cal_features(filein):
    """分词并通过已有的字典生成 X,y(real)文件

    字典: f0, f1 长度, 设定为相同长度
    """
    with open(filein_name, encoding='utf-8') as f:
        num_lines = len(f.readlines())
    print(num_lines)

    # 载入字典
    f000 = open('7-10-000-dict10000.txt', encoding='utf-8')
    f001 = open('7-10-001-dict10000.txt', encoding='utf-8')
    f010 = open('7-10-010-dict10000.txt', encoding='utf-8')
    f100 = open('7-10-100-dict10000.txt', encoding='utf-8')

    features000 = {}
    features001 = {}
    features010 = {}
    features100 = {}

    f000 = json.loads(f000.readline())
    f001 = json.loads(f001.readline())
    f010 = json.loads(f010.readline())
    f100 = json.loads(f100.readline())
    print(len(f000),len(f001),len(f010),len(f100))

    for i in range(0, len(f000)):
        features000[f000[i][0]] = i
        features001[f001[i][0]] = i
        features010[f010[i][0]] = i
        features100[f100[i][0]] = i    

    dict_length = len(features000)

    #初始化存储数据结构
    t = re.compile('\t')
    time_sep = re.compile('-')
    #fileoutX = open('X.txt','w')
    temp = []

    from scipy import sparse
    X000 = sparse.dok_matrix((num_lines, dict_length)) #使用稀疏矩阵, 减少空间占用
    X001 = sparse.dok_matrix((num_lines, dict_length))
    X010 = sparse.dok_matrix((num_lines, dict_length)) #使用稀疏矩阵, 减少空间占用
    X100 = sparse.dok_matrix((num_lines, dict_length))    
    y000 = []
    y001 = []
    y010 = []
    y100 = []

    #逐行分词并添加到X,y
    linenum = 0
    for line in filein: # write number of users as demand in num_user   
        #(uid, mid, day, forward_count,
        #comment_count, like_count, content) = t.split(line)
        (uid, mid, day, content) = t.split(line)

        cut_list = jieba.lcut(content) #分词

        cut_list_removed = [] # remove duplicates
        for i in cut_list:
            if i not in cut_list_removed:
                cut_list_removed.append(i)

        for word in cut_list_removed:
            if word in features000:
                X000[linenum, features000[word]] = 1
            if word in features001:
                X001[linenum, features001[word]] = 1
            if word in features010:
                X010[linenum, features010[word]] = 1
            if word in features100:
                X100[linenum, features100[word]] = 1            
        # y 值, 只考虑 0/1 
        '''
        if int(forward_count) != 0:
            forward_count = 1
        else:
            forward_count = 0
        y100.append([forward_count])

        if int(comment_count) != 0:
            comment_count = 1
        else:
            comment_count = 0
        y010.append([comment_count])

        if int(like_count) != 0:
            like_count = 1
        else:
            like_count = 0
        y001.append([like_count])
        '''

        linenum += 1
        if linenum % 5000 == 0:
            print(linenum)

    # io to file
    import scipy.io as sio
    sio.savemat(filein_name[:-4] + 'X000.mat', {'X000':X000})
    sio.savemat(filein_name[:-4] + 'X001.mat', {'X001':X001})
    sio.savemat(filein_name[:-4] + 'X010.mat', {'X010':X010})
    sio.savemat(filein_name[:-4] + 'X100.mat', {'X100':X100}) 

    '''
    sio.savemat(filein_name[:-4] + 'y001.mat', {'y001':y001})
    sio.savemat(filein_name[:-4] + 'y010.mat', {'y010':y010})
    sio.savemat(filein_name[:-4] + 'y100.mat', {'y100':y100})
    '''

#log_reg 和 loader 用于逻辑回归, 目前是多变量版本
def log_reg(X,y):
    '''logistic regression

    o: 模型存储为 .pkl 文件 @ python 3.4.3
    '''
    import numpy as np
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    #weight = {0:1, 1:3, 2:7, 3:14, 4:30, 5:70, 6:101}
    # Set regularization parameter

    # turn down tolerance for short training time
    clf = LogisticRegression(class_weight=weight, max_iter=5)
    clf.fit(X, y)

    print("score with L1 penalty: %.4f" % clf.score(X, y))

    from sklearn.externals import joblib
    joblib.dump(clf, '0903_uid_ave_100.pkl') 

def loader():
    import scipy.io as sio
    import scipy.sparse as sp
    import numpy as np

    X = sio.loadmat('uid_dict_X100-7-10.mat')['X']
    y = sio.loadmat('7-10y100.mat')['y']
    print(X.shape, y.shape)
    print(type(X), type(y))
    log_reg(X,y.ravel())

def predict_loader():
    import scipy.io as sio
    X = sio.loadmat('copy/12.txt-X100-model.mat')['X100']
    y = sio.loadmat('copy/12.txt-y100-model.mat')['X100']
    predict(filein_name)
    #log_reg(X,y)
    #print(X,y)
    #predict_cut('0823-100clf_l1_LR.pkl',X,y)

def predict_compare(filein_name):
    """预测

    """
    # get models
    from sklearn.externals import joblib
    LR010 = joblib.load('0903_uid_ave_010.pkl') 
    LR001 = joblib.load('0903_uid_ave_001.pkl') 
    LR100 = joblib.load('0903_uid_ave_100.pkl') 

    import scipy.io as sio

    X = sio.loadmat('uid_dict_X001-12.mat')['X']
    y_predict = LR001.predict(X)
    y_real = sio.loadmat('12y001.mat')['y'].ravel()
    
    true, false = 0, 0
    i = 0
    #print(y_predict.size)
    for i in range(0, y_predict.size):
        #print(y_predict[i], y_real[i])
        if y_predict[i] == y_real[i]:
            true += 1
        else:
            false += 1
    print('001', true, false)


    X = sio.loadmat('uid_dict_X010-12.mat')['X']
    y_predict = LR010.predict(X)
    y_real = sio.loadmat('12y010.mat')['y'].ravel()
    true, false = 0, 0
    i = 0
    #print(y_predict.size)
    for i in range(0, y_predict.size):
        #print(y_predict[i], y_real[i])
        if y_predict[i] == y_real[i]:
            true += 1
        else:
            false += 1
    print('010', true, false)

    X = sio.loadmat('uid_dict_X100-12.mat')['X']
    y_predict = LR100.predict(X)
    y_real = sio.loadmat('12y100.mat')['y'].ravel()
    true, false = 0, 0
    i = 0
    #print(y_predict.size)
    for i in range(0, y_predict.size):
        #print(y_predict[i], y_real[i])
        if y_predict[i] == y_real[i]:
            true += 1
        else:
            false += 1
    print('100', true, false)

def predict(filein_name):
    """预测

    """
    # get models
    from sklearn.externals import joblib
    LR010 = joblib.load('0903_uid_ave_010.pkl') 
    LR001 = joblib.load('0903_uid_ave_001.pkl') 
    LR100 = joblib.load('0903_uid_ave_100.pkl') 

    import scipy.io as sio

    X = sio.loadmat('uid_dict_X001-12.mat')['X']
    y_predict_prob = LR001.predict_proba(X)
    f = open(filein_name[:-4] + '-001prob.txt', 'w')
    f.write(json.dumps(y_predict_prob.tolist()))

    X = sio.loadmat('uid_dict_X010-12.mat')['X']
    y_predict_prob = LR010.predict_proba(X)
    f = open(filein_name[:-4] + '-010prob.txt', 'w')
    f.write(json.dumps(y_predict_prob.tolist()))

    X = sio.loadmat('uid_dict_X100-12.mat')['X']
    y_predict_prob = LR100.predict_proba(X)
    f = open(filein_name[:-4] + '-100prob.txt', 'w')
    f.write(json.dumps(y_predict_prob.tolist()))

def predict_cut(model, X, y_real):
    from sklearn.externals import joblib
    saved_LR = joblib.load(model) 
    #y_predict = saved_LR.predict(X)
    y_predict_prob = saved_LR.predict_proba(X)

    print(y_predict_prob)
    

    f = open(filein_name + 'prob.txt', 'w')
    f.write(json.dumps(y_predict_prob.tolist()))

    #print(y_predict, y_real)
    #print(y_predict_prob)
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0
    for i in range(0,len(y_predict)):
        predict = y_predict[i]
        real = y_real[i][0]
        if real:
            if predict != real:
                false_neg += 1
            else:
                true_pos += 1
        else:
            if predict != real:
                false_pos += 1
            else:
                true_neg += 1

    print('m:{}, true_pos:{}, false_pos:{}, false_neg:{}, true_neg:{}'.format(i+1, true_pos,false_pos,false_neg, true_neg))

def cut_replace(filein):
    """cut_replace

    """
    with open(filein_name, encoding='utf-8') as f:
        num_lines = len(f.readlines())
    print(num_lines)

    #初始化存储数据结构
    t = re.compile('\t')
    time_sep = re.compile('-')
    temp = []

    fileout = open(filein_name + 'cut.txt', 'w', encoding='utf-8')
    #逐行分词并添加到X,y
    linenum = 0
    for line in filein: # write number of users as demand in num_user   
        (uid, mid, time, forward_count,
        comment_count, like_count, content) = t.split(line)

        cut_list = jieba.lcut(content) #分词
        content = ' '.join(cut_list)
        fileout.write('\t'.join([uid, mid, time, forward_count,
                            comment_count, like_count, content]))

def predict_logistic_reg(fileout=0):
    """用 predict_data 的数据预测 predict_file

    i: predict_file, 暂时用 000 data
    p: 把 000 换成 average
    """
    predict_file = open('12.txt')
    t = re.compile('\t')
    predict_data = open('uid_average.txt')
    fileout = open('12-log.txt', 'w')
    predict_data = predict_data.readlines()

    dataset = {}

    for line in predict_data:
        uid, num_post, f,c,l = t.split(line)
        f = (float(f))
        c = (float(c))
        l = (float(l))
        dataset[uid] = (f, c, l) # 暂存平均值
    
    f001 = open('copy/12-001prob.txt')
    f010 = open('copy/12-010prob.txt')
    f100 = open('copy/12-100prob.txt')
    log_reg_001 = json.loads(f001.readline())
    log_reg_010 = json.loads(f010.readline())
    log_reg_100 = json.loads(f100.readline())
    
    print(log_reg_001[0][0])
    print(len(log_reg_100))
    
    i = 0
    for line in predict_file:

        uid, mid = t.split(line)[0], t.split(line)[1]
        f_lr = log_reg_100[i][0]
        c_lr = log_reg_010[i][0]
        l_lr = log_reg_001[i][0]

        if i == 0:
            print(f_lr,c_lr,l_lr)

        if uid in dataset:
            ave = dataset[uid]
        else:
            ave = (0,0,0)

        if f_lr > 0.5:
            f_lr = 0
        else:
            f_lr = round((math.fabs(math.log(f_lr))+1) * (ave[0]+1))

        if c_lr > 0.5:
            c_lr = 0
        else:
            c_lr = round((math.fabs(math.log(c_lr))+1) * (ave[1]+1))

        if l_lr > 0.5:
            l_lr = 0
        else:
            l_lr = round((math.fabs(math.log(l_lr))+1) * (ave[2]+1))

        fileout.write(uid + '\t' + mid + '\t' + str(f_lr) + ','
                      + str(c_lr) + ',' + str(l_lr) +'\n') 
        if i == 0:
            print(f_lr,c_lr,l_lr)

        i+=1

def log_reg_data_append(fileout=0):
    """

    """
    predict_file = open('7-10.txt')
    t = re.compile('\t')
    predict_data = open('uid_average.txt')
    fileout = open('7-10-log-reg.txt', 'w')
    predict_data = predict_data.readlines()

    dataset = {}

    for line in predict_data:
        uid, num_post, f,c,l = t.split(line)
        f = (float(f))
        c = (float(c))
        l = (float(l))
        dataset[uid] = (f, c, l) # 暂存平均值
    
    f001 = open('7-10-001-prob.txt')
    f010 = open('7-10-010-prob.txt')
    f100 = open('7-10-100-prob.txt')
    log_reg_001 = json.loads(f001.readline())
    log_reg_010 = json.loads(f010.readline())
    log_reg_100 = json.loads(f100.readline())
    
    print(log_reg_001[0][0])
    print(len(log_reg_100))
    
    i = 0
    for line in predict_file:
        (uid, mid, time, forward_count,
        comment_count, like_count, content) = t.split(line)
        f_lr = log_reg_100[i][1]
        c_lr = log_reg_010[i][1]
        l_lr = log_reg_001[i][1]

        fileout.write(uid + '\t' + mid + '\t' + time + '\t' +
                      forward_count + '\t' + comment_count + '\t' + 
                      like_count + '\t' + str(f_lr) +'\t' + str(c_lr) +
                      '\t' + str(l_lr) + '\n') 
        if i == 0:
            print(f_lr,c_lr,l_lr)

        i+=1

def log_reg_data_append_average(fileout=0):
    """

    """
    predict_file = open('7-10-log-reg.txt')
    t = re.compile('\t')
    predict_data = open('uid_average.txt')
    fileout = open('7-10-log-reg-ave.txt', 'w')
    predict_data = predict_data.readlines()

    dataset = {}

    for line in predict_data:
        uid, num_post, f,c,l = t.split(line)
        f = (float(f))
        c = (float(c))
        l = (float(l))
        dataset[uid] = (f, c, l) # 暂存平均值
    
    i = 0
    for line in predict_file:
        uid = t.split(line)[0]

        fileout.write(line[:-1] + '\t'+ str(dataset[uid][0])+ '\t' +str(dataset[uid][1])+ '\t'+ str(dataset[uid][2])+'\n') 
        if i == 0:
            print(line[:-1] + '\t'+ str(f)+ '\t' +str(c)+ '\t'+ str(l)+'\n')

        i+=1

def linear_reg_uid():
    """整理成 uid 字典

    由于 uid average 是六个月的数据, logistic regression是四个月的数据, 严格来说有过度拟合的可能.
    暂时不管.
    """
    predict_file = open('7-10-log-reg-ave.txt')
    t = re.compile('\t')
    fileout = open('0826uid-log-reg-data.txt', 'w')

    dataset = {}
    #数据结构: dataset[uid] = [[X],[y]]
    predict_data = open('uid_average.txt')
    predict_data = predict_data.readlines()

    for line in predict_data:
        uid, num_post, f,c,l = t.split(line)
        dataset[uid] = [[],[]] 

    i = 0
    for line in predict_file:
        (uid, mid, time, 
            fr, cr, lr, # real
            fp, cp, lp, # probability
            fa, ca, la) = t.split(line) #average
        fr = float(fr)
        cr = float(cr)
        lr = float(lr)
        fp = float(fp)
        cp = float(cp)
        lp = float(lp)
        fa = float(fa)
        ca = float(ca)
        la = float(la[:-1])

        dataset[uid][0].append([fp, cp, lp, fa, ca, la]) #note 03 forward, 14 comment 25 like
        dataset[uid][1].append([fr, cr, lr])

    import pprint
    pprint.pprint(dataset['63cd513528bba60bfde4178467df6987'])

    fileout.write(json.dumps(dataset))

def linear_reg_poly2():
    f = open('0826uid-log-reg-data.txt')
    fileout = open('0831-linear-reg-2-model.txt', 'w')
    dataset = json.loads(f.readline())

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model

    models = {}

    for i in dataset:
        uid, s = i, dataset[i]

        Xf = []
        Xc = []
        Xl = []
        yf = []
        yc = []
        yl = []
    
        for j in range(0, len(s[0])):
            #print(s[0][j], s[1][j])
            fp,cp,lp, fa,ca,la = s[0][j]
            fr,cr,lr = s[1][j]

            # cal weight
            weight = fr + cr + lr 
            if weight > 100:
                weight = int(100 + 1)
            elif weight < 1: 
                weight = int(1)
            else:
                weight = int(weight + 1)
            weight = round(weight ** 0.5)

            Xf += [[fp]] * weight
            yf += [fr] * weight
            Xc += [[cp]] * weight
            yc += [cr] * weight
            Xl += [[lp]] * weight
            yl += [lr] * weight

        if Xf == []:
            continue

        poly = PolynomialFeatures(degree=2)
        Xf2 = poly.fit_transform(Xf)
        clf_f = linear_model.LinearRegression()
        clf_f.fit(Xf2, yf)

        poly = PolynomialFeatures(degree=2)
        Xc2 = poly.fit_transform(Xc)
        clf_c = linear_model.LinearRegression()
        clf_c.fit(Xc2, yc)

        poly = PolynomialFeatures(degree=2)
        Xl2 = poly.fit_transform(Xl)
        clf_l = linear_model.LinearRegression()
        clf_l.fit(Xl2, yl)


        models[i] = [[clf_f.intercept_, clf_f.coef_.tolist()],
                     [clf_c.intercept_, clf_c.coef_.tolist()],
                     [clf_l.intercept_, clf_l.coef_.tolist()]]
        #print(models)
        #break
        
    fileout.write(json.dumps(models))# 二次拟合, 已弃
    fileout.close()

def linear_reg():
    f = open('0826uid-log-reg-data.txt')
    fileout = open('0827-linear-reg-model.txt', 'w')
    dataset = json.loads(f.readline())

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model

    models = {}

    for i in dataset:
        uid, s = i, dataset[i]

        Xf = []
        Xc = []
        Xl = []
        yf = []
        yc = []
        yl = []

        for j in range(0, len(s[0])):
            #print(s[0][j], s[1][j])
            fp,cp,lp, fa,ca,la = s[0][j]
            fr,cr,lr = s[1][j]

            # cal weight
            weight = fr + cr + lr 
            if weight > 100:
                weight = int(1)
            elif weight < 1: 
                weight = int(10)
            else:
                weight = int(1)
            #weight = round(weight ** 0.5) # modify weight

            Xf += [[fp]] * weight
            yf += [fr] * weight
            Xc += [[cp]] * weight
            yc += [cr] * weight
            Xl += [[lp]] * weight
            yl += [lr] * weight

        if Xf == []:
            continue

        clf_f = linear_model.LinearRegression()
        clf_f.fit(Xf, yf)

        clf_c = linear_model.LinearRegression()
        clf_c.fit(Xc, yc)

        clf_l = linear_model.LinearRegression()
        clf_l.fit(Xl, yl)

        models[i] = [[clf_f.intercept_, clf_f.coef_.tolist()],
                     [clf_c.intercept_, clf_c.coef_.tolist()],
                     [clf_l.intercept_, clf_l.coef_.tolist()]]
        #print(models)
        #break
        
    fileout.write(json.dumps(models))
    fileout.close()

def final_predict_poly2():# 二次拟合, 已弃
    """使用 线性拟合的概率拟合 f,c,l

    i: uid average; model; data to predict(probability)
    o: predict f,c,l
    """
    '''
    # 读取 uid average, model
    filein = open('weibo_predict_data.txt', encoding='utf-8')
    prob_f = open('predict-0824-100prob.txt')
    prob_c = open('predict-0824-010prob.txt')
    prob_l = open('predict-0824-001prob.txt')
    models = open('0827-linear-reg-model.txt')
    predict_data = open('uid_average.txt')
    fileout = open('0827-predict.txt','w')    
    '''
    filein = open('copy/12.txt', encoding='utf-8')
    prob_f = open('copy/12-100prob.txt')
    prob_c = open('copy/12-010prob.txt')
    prob_l = open('copy/12-001prob.txt')
    models = open('0831-linear-reg-2-model.txt')
    predict_data = open('uid_average.txt')
    fileout = open('0831-predict-12-2.txt','w')
    
    models = json.loads(models.readline())
    t = re.compile('\t')
    log_reg_f = json.loads(prob_f.readline())
    log_reg_c = json.loads(prob_c.readline())
    log_reg_l = json.loads(prob_l.readline())

    predict_data = predict_data.readlines()
    average = {}
    for line in predict_data:
        uid, num_post, f,c,l = t.split(line)
        f = (float(f))
        c = (float(c))
        l = (float(l))
        average[uid] = (f, c, l) # 暂存平均值

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model
    import numpy as np

    i = 0
    for line in filein:
        uid, mid = t.split(line)[0], t.split(line)[1]
        f_lr = log_reg_f[i][1]
        c_lr = log_reg_c[i][1]
        l_lr = log_reg_l[i][1]

        if uid in models:
            #读取 线性拟合 models
            f_intercept, f_coef = models[uid][0]
            c_intercept, c_coef = models[uid][1]
            l_intercept, l_coef = models[uid][2]
            #使用 概率 和 average 进行预测
            poly = PolynomialFeatures(degree=2)
            Xf = poly.fit_transform([f_lr])
            Xc = poly.fit_transform([c_lr])
            Xl = poly.fit_transform([l_lr])

            f_predict = (f_intercept + np.dot(Xf,f_coef))[0]
            c_predict = (c_intercept + np.dot(Xc,c_coef))[0]
            l_predict = (l_intercept + np.dot(Xl,l_coef))[0]
            #整理输出为非负整数
            if f_predict < 0:
                f_predict = 0
            elif f_predict > 10000:
                f_predict = 10000
            else:
                f_predict = round(float(f_predict))

            if c_predict < 0:
                c_predict = 0
            elif c_predict > 10000:
                c_predict = 10000
            else:
                c_predict = round(float(c_predict))
            if l_predict < 0:
                l_predict = 0
            elif l_predict > 10000:
                l_predict = 10000          
            else:
                l_predict = round(float(l_predict))
            #print(f_predict, c_predict,l_predict)
            #print(f_predict)
        else: #可能出现未知用户, 避免读词典导致报错
            f_predict = round(float(f_lr))
            c_predict = round(float(c_lr))
            l_predict = round(float(l_lr))

        #输出
        fileout.write(uid + '\t' + mid + '\t' + 
                      str(f_predict) + ',' + str(c_predict) + ',' +
                      str(l_predict) + '\n')    
        
        i += 1
        
      
    #test output
    '''
    import numpy as np
    print(models)
    for i in Xf2:
        print(np.dot(i,clf_f.coef_)+clf_f.intercept_)'''

def final_predict():
    """使用 线性拟合的概率拟合 f,c,l

    i: uid average; model; data to predict(probability)
    o: predict f,c,l
    """
    # 读取 uid average, model
    '''
    filein = open('weibo_predict_data.txt', encoding='utf-8')
    prob_f = open('predict-0824-100prob.txt')
    prob_c = open('predict-0824-010prob.txt')
    prob_l = open('predict-0824-001prob.txt')
    models = open('0827-linear-reg-model.txt')
    predict_data = open('uid_average.txt')
    fileout = open('0827-predict.txt','w')    
    

    '''
    filein = open('copy/12.txt', encoding='utf-8')
    prob_f = open('copy/12-100prob.txt')
    prob_c = open('copy/12-010prob.txt')
    prob_l = open('copy/12-001prob.txt')
    models = open('0827-linear-reg-model.txt')
    predict_data = open('uid_average.txt')
    fileout = open('0827-predict-12.txt','w')
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model
    import numpy as np

    models = json.loads(models.readline())
    t = re.compile('\t')
    log_reg_f = json.loads(prob_f.readline())
    log_reg_c = json.loads(prob_c.readline())
    log_reg_l = json.loads(prob_l.readline())

    predict_data = predict_data.readlines()
    average = {}
    for line in predict_data:
        uid, num_post, f,c,l = t.split(line)
        f = (float(f))
        c = (float(c))
        l = (float(l))
        average[uid] = (f, c, l) # 暂存平均值

    i = 0
    for line in filein:
        uid, mid = t.split(line)[0], t.split(line)[1]
        f_lr = log_reg_f[i][1]
        c_lr = log_reg_c[i][1]
        l_lr = log_reg_l[i][1]

        if uid in models:
            #读取 线性拟合 models
            f_intercept, f_coef = models[uid][0]
            c_intercept, c_coef = models[uid][1]
            l_intercept, l_coef = models[uid][2]

            Xf = [f_lr]
            Xc = [c_lr]
            Xl = [l_lr]

            #使用 概率 和 average 进行预测

            f_predict = (f_intercept + np.dot(Xf,f_coef))
            c_predict = (c_intercept + np.dot(Xc,c_coef))
            l_predict = (l_intercept + np.dot(Xl,l_coef))

            #整理输出为非负整数
            if f_predict < 0:
                f_predict = 0
            elif f_predict > 10000:
                f_predict = 10000
            else:
                f_predict = round(float(f_predict))

            if c_predict < 0:
                c_predict = 0
            elif c_predict > 10000:
                c_predict = 10000
            else:
                c_predict = round(float(c_predict))
            if l_predict < 0:
                l_predict = 0
            elif l_predict > 10000:
                l_predict = 10000
            else:
                l_predict = round(float(l_predict))

        else: #可能出现未知用户, 避免读词典导致报错
            f_predict = round(float(f_lr))
            c_predict = round(float(c_lr))
            l_predict = round(float(l_lr))
        #输出
        fileout.write(uid + '\t' + mid + '\t' + 
                      str(f_predict) + ',' + str(c_predict) + ',' +
                      str(l_predict) + '\n')    
        i += 1
        #break

def predict_multi():
    """使用 线性拟合的概率拟合 f,c,l

    i: uid average; model; data to predict(probability)
    o: predict f,c,l
    """
    # 读取 uid average, model
    '''
    filein = open('weibo_predict_data.txt', encoding='utf-8')
    prob_f = open('predict-0824-100prob.txt')
    prob_c = open('predict-0824-010prob.txt')
    prob_l = open('predict-0824-001prob.txt')
    models = open('0827-linear-reg-model.txt')
    predict_data = open('uid_average.txt')
    fileout = open('0827-predict.txt','w')    

    '''
    filein = open('12.txt', encoding='utf-8')
    prob_f = open('12-100prob.txt')
    prob_c = open('12-010prob.txt')
    prob_l = open('12-001prob.txt')
    predict_data = open('uid_average.txt')
    fileout = open('0903-predict-12.txt','w')
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model
    import numpy as np

    t = re.compile('\t')
    log_reg_f = json.loads(prob_f.readline())
    log_reg_c = json.loads(prob_c.readline())
    log_reg_l = json.loads(prob_l.readline())

    predict_data = predict_data.readlines()
    average = {}
    for line in predict_data:
        uid, num_post, f,c,l = t.split(line)
        f = (float(f))
        c = (float(c))
        l = (float(l))
        average[uid] = (f, c, l) # 暂存平均值

    i = 0
    for line in filein:
        uid, mid = t.split(line)[0], t.split(line)[1]
        f_lr = log_reg_f[i]
        c_lr = log_reg_c[i]
        l_lr = log_reg_l[i]

        #print(f_lr,c_lr,l_lr)
        #print(f_lr.index(max(f_lr)),c_lr.index(max(c_lr)),l_lr.index(max(l_lr)))

        #print(predict_count(f_lr.index(max(f_lr))))
        #print(predict_count(c_lr.index(max(c_lr))))
        #print(predict_count(l_lr.index(max(l_lr))))
        '''
        if uid in models:
            #读取 线性拟合 models
            f_intercept, f_coef = models[uid][0]
            c_intercept, c_coef = models[uid][1]
            l_intercept, l_coef = models[uid][2]

            Xf = [f_lr]
            Xc = [c_lr]
            Xl = [l_lr]

            #使用 概率 和 average 进行预测

            f_predict = (f_intercept + np.dot(Xf,f_coef))
            c_predict = (c_intercept + np.dot(Xc,c_coef))
            l_predict = (l_intercept + np.dot(Xl,l_coef))

            #整理输出为非负整数
            if f_predict < 0:
                f_predict = 0
            elif f_predict > 10000:
                f_predict = 10000
            else:
                f_predict = round(float(f_predict))

            if c_predict < 0:
                c_predict = 0
            elif c_predict > 10000:
                c_predict = 10000
            else:
                c_predict = round(float(c_predict))
            if l_predict < 0:
                l_predict = 0
            elif l_predict > 10000:
                l_predict = 10000
            else:
                l_predict = round(float(l_predict))

        else: #可能出现未知用户, 避免读词典导致报错
        '''
        f_predict = predict_count(f_lr)
        c_predict = predict_count(c_lr)
        l_predict = predict_count(l_lr)
        
        #输出
        fileout.write(uid + '\t' + mid + '\t' + 
                      str(f_predict) + ',' + str(c_predict) + ',' +
                      str(l_predict) + '\n')    
        
        i += 1
        #break

def predict_count(proba_list):
    d = {0:0, 1:3, 2:7, 3:15, 4:30, 5:70, 6: 100}

    max_proba = max(proba_list)
    category = proba_list.index(max(proba_list))


    if category == 0:
        predict = 0
    elif category == 6:
        predict = round(100 * (1 + max_proba ** 2) )
    else:
        if proba_list[category - 1] > proba_list[category + 1]:
            second_max = proba_list[category - 1]
            predict = round(d[category - 1] + (max_proba - second_max) * (d[category] - d[category - 1]))
        else:
            second_max = proba_list[category + 1]
            predict = round(d[category] + (max_proba - second_max) * (d[category + 1] - d[category]))
    return predict

def main():
    import time
    t0 = time.time()
    #ftemp = open(filein)
    #init_dict(filein)
    #load_dict(filein)
    '''
    s = ['copy/8.txt-1.txt','copy/8.txt-2.txt',
         'copy/9.txt-0.txt','copy/9.txt-1.txt','copy/9.txt-2.txt',
         'copy/10.txt-0.txt','copy/10.txt-1.txt','copy/10.txt-2.txt',
         'copy/11.txt-0.txt','copy/11.txt-1.txt','copy/11.txt-2.txt',
         'copy/12.txt-0.txt','copy/12.txt-1.txt','copy/12.txt-2.txt',]
    global filein_name
    for filename in s:
        filein_name = filename
        print(filename)
        cal_features(open(filein_name, encoding='utf-8')) #计算feature
    '''
    #cal_features(filein)
    #loader() 
    #predict_loader() #预测
    #predict(filein_name)
    #predict_logistic_reg()
    #log_reg_data_append_average()
    #linear_reg_uid()
    #linear_reg()
    #linear_reg_poly2()
    #final_predict_poly2()
    #predict_multi()
    predict_compare('hello')
    t1 = time.time()
    print('Finished: runtime {}'.format(t1 - t0))
    #cut_replace(filein)

if __name__ == '__main__':
    main()