# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# cut for weibo data
# -*- coding: utf-8 -*-

import re
import jieba
import json
import math

def get_data_and_write(filein):
    """用 readline 逐行读入数据并 unpack

    可用适当参数输出
    """
    t = re.compile('\t')
    time_sep = re.compile('-')

    #fileout02 = open('02_norm.txt', 'w', encoding='utf-8')
    #fileout03 = open('03-07.txt', 'w', encoding='utf-8')
    fileout_cut = open('predict_cut.txt', 'w', encoding='utf-8')

    i = 0
    for line in filein: # write number of users as demand in num_user   
        (mid, uid, time, forward_count,
            comment_count, like_count, content) = t.split(line) # for normal file
        #(uid, mid, time, content) = t.split(line) # for test file
        '''
        yyyy, mm, dd = time_sep.split(time)
        forward_count = int(forward_count)
        comment_count = int(comment_count)
        like_count = int(like_count)'''

        cut_list = jieba.lcut(content)
        length = len(content)
        #print(cut_list)
        #print(uid, mid, time, forward_count, comment_count, like_count)
        #output = [mid, uid, time, forward_count,
        #    comment_count, like_count, length, cut_list]

        output = [mid, uid, time, length, cut_list]
        fileout_cut.write(json.dumps(output) + '\n')
        #if mm == '02':
        #    fileout02.write(line)
        i += 1

        if i % 50000 == 0:
            print(i)
    #fileout.write(json.dumps(uid_dict))
    print(i)
    #print(len(uid_dict))
    #print(uid_dict)


def predict_uid_ave():
    filein = open('weibo_predict_data_new.txt')
    uid_dict = json.loads(open('weibo_train_uid.txt', encoding='utf-8').readline())

    fileout_ave = open('predict_ave.txt', 'w', encoding='utf-8')
    t = re.compile('\t')

    for key in uid_dict:
        num_post, forward_count, comment_count, like_count = uid_dict[key]
        forward_ave = round(forward_count / num_post)
        comment_ave = round(comment_count / num_post)
        like_ave = round(like_count / num_post)
        uid_dict[key] = str(forward_ave)+ ',' +str(comment_ave)+',' +str(like_ave)

    i = 0
    for line in filein: # write number of users as demand in num_user
        (uid, mid, time, content) = t.split(line)
        if uid in uid_dict:
            fileout_ave.write(uid + '\t' + mid + '\t' + uid_dict[uid] + '\n')
        else:
            fileout_ave.write(uid + '\t' + mid + '\t' + '0,0,0' + '\n')


def get_data_and_write_predict(filein, fileout):
    """用 readline 逐行读入数据并 unpack

    可用适当参数输出
    """
    t = re.compile('\t')

    uid_dict = {}
    i = 0
    for line in filein: # write number of users as demand in num_user
        (mid, uid, time, content) = t.split(line)
        #yyyy, mm, dd = time_sep.split(time)
        #print(yyyy, mm, dd)
        #cut_list = jieba.lcut(content)
        #print(cut_list)
        #print(uid, mid, time, forward_count, comment_count, like_count)
        if uid in uid_dict: # update element
            uid_dict[uid] += 1
        else: #init dict element
            uid_dict[uid] = 1
        i += 1

        #if i == 500000:
        #    break
    fileout.write(json.dumps(uid_dict))
    print(i)
    print(len(uid_dict))
    #print(uid_dict)

def uid_compare():
    train = open('weibo_train_uid.txt', encoding='utf-8')
    uid_train = json.loads(train.readline())

    predict = open('weibo_predict_uid.txt', encoding= 'utf-8')
    uid_predict = json.loads(predict.readline())

    predict_in_train = 0
    post_in_train = 0
    post_in_predict = 0 
    predict_not_in_train = 0 
    for uid in uid_predict:
        if uid in uid_train:
            predict_in_train += 1
            num_post_train, num_forward, num_comment, num_like = uid_train[uid]
            post_in_train += num_post_train
            num_post_predict = uid_predict[uid]
            post_in_predict += num_post_predict
        else:
            predict_not_in_train += 1
    print(predict_in_train, predict_not_in_train)
    print(post_in_train, post_in_predict)

def uid_sort(filein_name):
    """从已有的分词文件建立dict

    目前删去了只有一次的词。 对于 class 0 不妨用 >5
    """
    filein = open(filein_name, encoding='utf-8')
    fileout = open(filein_name[:-4]+'uid_sorted.txt', 'w', encoding='utf-8')

    d = {} #uid:index
    #post_list = []

    i = 0
    for line in filein:
        (mid, uid, time, forward_count,
            comment_count, like_count, length, content) = json.loads(line)
        content = ' '.join(content)
        string = '\t'.join([mid, uid, time, str(forward_count),
            str(comment_count), str(like_count), str(length), content])
        if uid in d:
            d[uid].append(string)
        else:
            d[uid] = [string]
        i += 1
        #if i == 100:
        #    break

    for uid in d:
        for post in d[uid]:
            fileout.write(post)

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

def main():
    import time
    t0 = time.time()
    
    #filein = open('train_cut.txt', encoding='utf-8')
    #fileout = open('weibo_train_uid.txt', 'w', encoding='utf-8')
    uid_sort('train_cut.txt')
    #predict_uid_ave()

    #filein = open('weibo_predict_data.txt', encoding='utf-8')
    #fileout = open('weibo_predict_uid.txt', 'w', encoding='utf-8')
    #get_data_and_write_predict(filein, fileout)

    #uid_compare()

    t1 = time.time()
    print('Finished: runtime {}'.format(t1 - t0))
    #cut_replace(filein)

if __name__ == '__main__':
    main()