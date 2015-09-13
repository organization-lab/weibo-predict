# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# cut for weibo data
# -*- coding: utf-8 -*-

import re
import jieba
import json
import math

#filein = open(filein_name, encoding='utf-8')

def cut_to_lists(filein, fileout):
    """ 分词并输出列表到文件

    分词后已去重
    """
    t = re.compile('\t')
    time_sep = re.compile('-')

    temp = []

    for line in filein: # write number of users as demand in num_user   
        (uid, mid, time, forward_count,
        comment_count, like_count, content) = t.split(line)

        forward_count = int(forward_count)
        comment_count = int(comment_count)
        like_count = int(like_count)

        cut_list = jieba.lcut(content)
        content = ' '.join(cut_list)

        ''' remove duplicates
        cut_list_no_dup = [] # remove duplicates
        for i in cut_list:
            if i not in cut_list_no_dup:
                cut_list_no_dup.append(i)
        '''
        fileout.write(json.dumps([uid,mid,time,
            forward_count,comment_count,like_count,
            content]) + '\n')

def cut_loader():
    filein = open('weibo_train_data_cut.txt', encoding='utf-8')
    #fileout = open('weibo_train_data_cut.txt', 'w', encoding='utf-8')
    categorize_weibo(filein)

    '''
    for i in range(0,7):
        for j in ['forward','comment','like']:
            filein = open(j + str(i) +'.txt', encoding='utf-8')
            fileout = open(j + str(i) +'-dict.txt', 'w', encoding='utf-8')
            cut_to_lists(filein, fileout)
    '''

def categorize_weibo(filein):
    """分类输出微博信息

    按三个数量分类源文件
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

def init_dict(filein, fileout):
    """从已有的分词文件建立dict

    目前删去了只有一次的词。
    """
    d = {}
    t = re.compile(' ')
    for line in filein:
        (uid, mid, time, forward_count,
            comment_count, like_count, content) = json.loads(line)
        words = t.split(content)

        words_remove_dup = [] # remove duplicates
        for i in words:
            if i not in words_remove_dup:
                words_remove_dup.append(i)

        for word in words_remove_dup:
            if word in d:
                d[word] += 1
            else:
                d[word] = 1
    d1 = {}
    for key in d:
        if d[key] >= 2:
            d1[key] = d[key]

    d1 = sorted(d1.items(), key=lambda d1:d1[1], reverse=True)

    fileout.write(json.dumps(d1))

def load_dict(filein):
    list_dict = json.loads(filein.readline())
    #print(list_dict)
    for key in list_dict[:10]:
        print(key[0], key[1])
    out_list = list_dict[:10000]
    fileout = open(filein_name + '10000.txt', 'w', encoding='utf-8')
    fileout.write(json.dumps(out_list))

def cal_freq():
    """ 计算字典词频

    i: dict
    o: 词频 按照 forward comment like 排序 共 21 个元素
    """
    freq_list = [] # 是一个列表嵌套词典
    words_list = {}
    types = ['forward','comment','like']

    for j in types:
        for i in range(0,7):
            freq_list.append([])

    for j in types:
        for i in range(0,7):
            print(len(words_list))
            filein = open(j + str(i) +'-dict.txt', encoding='utf-8')
            list_dict = json.loads(filein.readline()) # list_dict 即为词典列表
            num_post = list_dict[0][1] # \n 的数量, 与微博数量一致
            print(len(list_dict))
            for word in list_dict:
                word[1] = word[1] / num_post # 计算频率
                if word[1] > 0:
                    if word[0] not in words_list:
                        words_list[word[0]] = 0
            list_dict = dict(list_dict)
            freq_list[types.index(j) * 7 + i] = list_dict
            
            '''
            fileout = open(j + str(i) +'-freq.txt', 'w', encoding='utf-8')
            fileout.write(json.dumps(list_dict))
            break'''
    #print(words_list[:10])
    print(len(words_list))
    dict_freq = {}
    init_freq = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for word in words_list:
        dict_freq[word] = init_freq[:]
        for i in range(0, len(freq_list)):
            if word in freq_list[i]:
                dict_freq[word][i] = freq_list[i][word]

    #print(dict_freq)
    fileout = open('dict-freq.txt', 'w', encoding='utf-8')
    fileout.write(json.dumps(dict_freq))

def get_features():
    """ 从词频字典提取 feature

    i: 词频字典
    o: feature list, txt in json
    """
    import numpy as np

    f = open('dict-freq.txt', encoding='utf-8')
    dict_freq = json.loads(f.readline())
    print(len(dict_freq))
    

    types = ['forward','comment','like']

    list_features_f = [{},{},{},{},{},{},{}]
    list_features_c = [{},{},{},{},{},{},{}]
    list_features_l = [{},{},{},{},{},{},{}]

    for key in dict_freq:
        freq_forward = dict_freq[key][0:6]
        index, num = freq_forward.index(max(freq_forward)), np.mean(freq_forward) * np.std(freq_forward)
        list_features_f[index][key] = num
        '''
        freq_comment = dict_freq[key][7:13]
        index, num = freq_comment.index(max(freq_comment)), np.mean(freq_comment) * np.std(freq_comment)
        list_features_c[index][key] = num

        freq_like = dict_freq[key][14:20]
        index, num = freq_like.index(max(freq_like)), np.mean(freq_like) * np.std(freq_forward)
        list_features_l[index][key] = num
        '''
    fileout_f = open('20150913features-forward-2000.txt', 'w', encoding='utf-8')
    fileout_c = open('20150913features-comment-2000.txt', 'w', encoding='utf-8')
    fileout_l = open('20150913features-like-2000.txt', 'w', encoding='utf-8')
    write_features(fileout_f, list_features_f)
    write_features(fileout_c, list_features_c)
    write_features(fileout_l, list_features_l)

def write_features(fileout, list_features):
    """ 输出 features 为 json 格式列表文件

    """
    sorted_features = []
    for i in range(0, len(list_features)):
        features = list_features[i]
        features = sorted(features.items(), key=lambda features:features[1], reverse=True)
        list_features[i] = features

    for i in range(0,1000): # 重排顺序版本
        for j in range(0, len(list_features) - 1):
            sorted_features.append(list_features[j][i])
    print(len(sorted_features))
    fileout.write(json.dumps(sorted_features))

    ''' # 不重排顺序版本
    for features in list_features:
        features = sorted(features.items(), key=lambda features:features[1], reverse=True)
        sorted_features += features[:1000]
    print(len(sorted_features))
    fileout.write(json.dumps(sorted_features))'''
    
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

    get_features()

    t1 = time.time()
    print('Finished: runtime {}'.format(t1 - t0))
    #cut_replace(filein)

if __name__ == '__main__':
    main()