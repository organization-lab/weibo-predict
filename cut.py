# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# cut for weibo data
# -*- coding: utf-8 -*-

import re
import jieba
import json
from operator import itemgetter
from sys import argv

script, filein_name = argv

filein = open(filein_name, encoding='utf-8')
#fileout = open(fileout_name, 'w', encoding='utf-8')
'''
fileout000 = open('test000.txt', 'w', encoding='utf-8')
fileout100 = open('test100.txt', 'w', encoding='utf-8')
fileout010 = open('test010.txt', 'w', encoding='utf-8')
fileout001 = open('test001.txt', 'w', encoding='utf-8')
'''
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
        (uid, mid, day, forward_count,
        comment_count, like_count, content) = t.split(line)

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

        linenum += 1
        if linenum % 1000 == 0:
            print(linenum)

    # io to file
    import scipy.io as sio
    sio.savemat(filein_name[:-4] + 'X000.mat', {'X000':X000})
    sio.savemat(filein_name[:-4] + 'X001.mat', {'X001':X001})
    sio.savemat(filein_name[:-4] + 'X010.mat', {'X010':X010})
    sio.savemat(filein_name[:-4] + 'X100.mat', {'X100':X100})    
    sio.savemat(filein_name[:-4] + 'y001.mat', {'y001':y001})
    sio.savemat(filein_name[:-4] + 'y010.mat', {'y010':y010})
    sio.savemat(filein_name[:-4] + 'y100.mat', {'y100':y100})

def log_reg(X,y):
    '''logistic regression

    o: 模型存储为 filein_name + 'clf_l1_LR.pkl'
    '''
    import numpy as np
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Set regularization parameter
    for C in [100]:
        # turn down tolerance for short training time
        clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
        #clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
        clf_l1_LR.fit(X, y)
        #clf_l2_LR.fit(X, y)

        coef_l1_LR = clf_l1_LR.coef_.ravel()
        #coef_l2_LR = clf_l2_LR.coef_.ravel()

        # coef_l1_LR contains zeros due to the
        # L1 sparsity inducing norm

        sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
        #sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100

        print("C=%.2f" % C)
        print("Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR)
        print("score with L1 penalty: %.4f" % clf_l1_LR.score(X, y))
        #print("Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR)
        #print("score with L2 penalty: %.4f" % clf_l2_LR.score(X, y))

        y1= clf_l1_LR.predict_proba(X)
        print(y1)
        #y2= clf_l2_LR.predict_proba(X)
        #print(y2)

    from sklearn.externals import joblib
    joblib.dump(clf_l1_LR, filein_name + 'clf_l1_LR.pkl') 

def loader():
    import scipy.io as sio
    X000 = sio.loadmat(filein_name[:-4] + 'X000.mat')['X000']
    X001 = sio.loadmat(filein_name[:-4] + 'X001.mat')['X001']
    X010 = sio.loadmat(filein_name[:-4] + 'X010.mat')['X010']
    X100 = sio.loadmat(filein_name[:-4] + 'X100.mat')['X100']

    y001 = sio.loadmat(filein_name[:-4] + 'y001.mat')['y001']
    y010 = sio.loadmat(filein_name[:-4] + 'y010.mat')['y010']
    y100 = sio.loadmat(filein_name[:-4] + 'y100.mat')['y100']

    '''print(X000[:10])
    print(X001[:10])
    print(X010[:10])
    print(X100[:10])'''
    print()
    print(y001[-10:])
    print(y010[-10:])
    print(y100[-10:])

    import scipy.sparse as sp
    import numpy as np



    #log_reg(X,y)
    #print(X,y)
    #predict_cut('clf_l1_LR.pkl',X,y)

def predict_loader():
    import scipy.io as sio
    X = sio.loadmat(filein_name[:-4] + 'X.mat')['X']
    y = sio.loadmat(filein_name[:-4] + 'y.mat')['y']
    #log_reg(X,y)
    #print(X,y)
    predict_cut('first100user.txtclf_l1_LR.pkl',X,y)

def predict_cut(model, X, y_real):
    from sklearn.externals import joblib
    saved_LR = joblib.load(model) 
    y_predict = saved_LR.predict(X)
    y_predict_prob = saved_LR.predict_proba(X)
    print(y_predict_prob)
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

def main():
    import time
    t0 = time.time()
    #ftemp = open(filein)
    #init_dict(filein)
    #load_dict(filein)
    cal_features(filein) #计算feature
    #loader() 

    t1 = time.time()
    print('Finished: runtime {}'.format(t1-t0))
    #predict_loader() #预测
    #cut_replace(filein)

if __name__ == '__main__':
    main()