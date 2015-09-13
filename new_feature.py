# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# cut for weibo data
# -*- coding: utf-8 -*-

import re
import jieba
import json
import math
import scipy.io as sio

#filein = open(filein_name, encoding='utf-8')

def sep_file(filein):
    """用 readline 逐行读入数据并 unpack

    可用适当参数输出
    """
    t = re.compile('\t')
    time_sep = re.compile('-')

    fileout = open('07-cut.txt', 'w', encoding='utf-8')
    fileout2 = open('08-12-cut.txt', 'w', encoding='utf-8')

    temp = []

    #i = 0
    for line in filein: # write number of users as demand in num_user   
        (uid, mid, time, forward_count,
        comment_count, like_count, content) = json.loads(line)
        yyyy, mm, dd = time_sep.split(time)
        #print(yyyy, mm, dd)
        if mm == '07':
            fileout.write(line)
        else:
            fileout2.write(line)
        #i+=1
        #if i == 100:
        #    break

def cut_to_lists(filein, filein_name):
    """ 分词并输出列表到文件

    分词后已去重
    """
    t = re.compile('\t')
    time_sep = re.compile('-')

    temp = []
    fileout = open(filein_name[:-4] + 'cut.txt', 'w', encoding='utf-8')

    for line in filein: # write number of users as demand in num_user   
        '''(uid, mid, time, forward_count,
        comment_count, like_count, content) = t.split(line)

        forward_count = int(forward_count)
        comment_count = int(comment_count)
        like_count = int(like_count)'''

        (uid, mid, time, content) = t.split(line) # for predict data

        cut_list = jieba.lcut(content)
        content = ' '.join(cut_list)

        ''' remove duplicates
        cut_list_no_dup = [] # remove duplicates
        for i in cut_list:
            if i not in cut_list_no_dup:
                cut_list_no_dup.append(i)
        '''
        fileout.write(json.dumps([uid, mid, time, content]) + '\n')

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
    sio.savemat(filein_name[:-4] + '_y_like.mat', {'y':y_like})
    sio.savemat(filein_name[:-4] + '_y_comment.mat', {'y':y_comment})
    sio.savemat(filein_name[:-4] + '_y_forward.mat', {'y':y_forward})
    sio.savemat(filein_name[:-4] + '_weight.mat', {'weight':weight})

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
        #(uid, mid, day,  content) = json.loads(line)

        (uid, mid, time, forward_count,
        comment_count, like_count, content) = json.loads(line)
        
        if uid in dataset: # 处理不在已有数据内的用户, 即其均值未知
            uid_features.append(dataset[uid])
        else: 
            uid_features.append([0,0,0])

    # io to file
    sio.savemat(filein_name[:-4] + '_uid_ave.mat', {'X':uid_features})

def post_length(filein, filein_name):
    """输出 微博长度

    """
    t = re.compile(' ')

    length = []

    #逐行
    linenum = 0
    for line in filein: # write number of users as demand in num_user   
        (uid, mid, day, forward_count,
        comment_count, like_count, content) = json.loads(line)
        #(uid, mid, day, content) = t.split(line)
        content = t.split(content)
        content = ''.join(content)
        print(content, len(content))
        break
        length.append([len(content)])
        
    # io to file
    sio.savemat(filein_name[:-4] +'_X_length.mat', {'X':length})

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
    
def cal_features(filein, filein_name):
    """ 通过 post & features 生成 X

    i: 分词过的文件
    o: X_dict
    """
  
    with open(filein_name, encoding='utf-8') as f:
        num_lines = len(f.readlines())
    print(num_lines)

    # 载入字典
    dict_f = open('20150913features-forward-1000.txt', encoding='utf-8')
    dict_c = open('20150913features-comment-1000.txt', encoding='utf-8')
    dict_l = open('20150913features-like-1000.txt', encoding='utf-8')

    features_f = {}
    features_c = {}
    features_l = {}

    dict_f = json.loads(dict_f.readline())
    dict_c = json.loads(dict_c.readline())
    dict_l = json.loads(dict_l.readline())

    print(len(dict_f))

    for i in range(0, len(dict_f)):
        features_f[dict_f[i][0]] = i 
        features_c[dict_c[i][0]] = i 
        features_l[dict_l[i][0]] = i 

    dict_length = len(features_f)
    print(dict_length)

    #初始化存储数据结构
    t = re.compile('\t')

    temp = []

    from scipy import sparse
    Xf = sparse.dok_matrix((num_lines, dict_length)) #使用稀疏矩阵, 减少空间占用 
    Xc = sparse.dok_matrix((num_lines, dict_length))
    Xl = sparse.dok_matrix((num_lines, dict_length))

    linenum = 0

    for line in filein: # write number of users as demand in num_user   
        (uid, mid, day, forward_count,
        comment_count, like_count, content) = json.loads(line)
        #(uid, mid, day, content) = json.loads(line)

        cut_list_removed = [] # remove duplicates
        for i in content:
            if i not in cut_list_removed:
                cut_list_removed.append(i)

        for word in cut_list_removed:
            if word in features_f:
                Xf[linenum, features_f[word]] = 1         
            if word in features_c:
                Xc[linenum, features_c[word]] = 1  
            if word in features_l:
                Xl[linenum, features_l[word]] = 1

        linenum += 1
        if linenum % 20000 == 0:
            print(linenum)
            #break#

    # io to file
    import scipy.io as sio
    sio.savemat(filein_name[:-4] + '_Xf.mat', {'X':Xf})
    sio.savemat(filein_name[:-4] + '_Xc.mat', {'X':Xc})
    sio.savemat(filein_name[:-4] + '_Xl.mat', {'X':Xl})

def loaders(filein_name):
    """ 把 loaders 整理到一起
    """

    #filein = open(filein_name, encoding='utf-8')
    #y_and_weight(filein, filein_name)

    #filein = open(filein_name, encoding='utf-8')
    #uid_average(filein, filein_name)

    filein = open(filein_name, encoding='utf-8')
    post_length(filein, filein_name)

def main():
    import time
    t0 = time.time()
    loaders('08-12-cut.txt')
    #cal_features(filein, 'weibo_predict_data_cut.txt')
    #uid_average(filein, '12-cut.txt')


    t1 = time.time()
    print('Finished: runtime {}'.format(t1 - t0))
    #cut_replace(filein)

if __name__ == '__main__':
    main()