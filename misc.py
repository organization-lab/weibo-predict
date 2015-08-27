# -*- coding: utf-8 -*-
# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# misc function for i/o of basic data

import re
import math
from sys import argv

#file1 = 'first100user.txt'
#file1 = 'weibo_train_data.txt'

script, filein_name = argv
#filein = open(filein_name, encoding='utf-8')

#fileout = open('predict_1000_average.txt', 'w')

#predict = open('predict/predict_1000_average.txt') 
#real = open('predict/predict_1000_real.txt') 

#predict_file = open('first1000user.txt')
#predict_data = open('uid_average.txt')

def sep(filein):
    f = filein.readlines()
    print(len(f[:100000]), len(f[100000:200000]), len(f[200000:]))

    fileout = open(filein_name + '-0.txt', 'w', encoding='utf-8')
    fileout.writelines(f[:100000])
    fileout = open(filein_name + '-1.txt', 'w', encoding='utf-8')
    fileout.writelines(f[100000:200000])
    fileout = open(filein_name + '-2.txt', 'w', encoding='utf-8')
    fileout.writelines(f[200000:])

def predict_000(filein, fileout):
    """保留前两列内容, 写下000

    前两列内容必须为 uid, mid
    """
    t = re.compile('\t')

    for line in filein:
        data = t.split(line)
        del data[2:]
        data.append('0,0,0')
        fileout.write('\t'.join(data))
        fileout.write('\n')

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
    time_sep = re.compile('-')

    temp = []
    for line in filein: # write number of users as demand in num_user   
        (uid, mid, time, forward_count,
        comment_count, like_count, content) = t.split(line)
        yyyy, mm, dd = time_sep.split(time)
        #print(yyyy, mm, dd)
        forward_count = int(forward_count)
        comment_count = int(comment_count)
        like_count = int(like_count)
        if forward_count == 0 and comment_count == 0 and like_count == 0:
            fileout000.write(line)
        else: 
            fileout111.write(line)
            if forward_count != 0:
                fileout100.write(line)
            if comment_count != 0:
                fileout010.write(line)
            if like_count != 0:
                fileout001.write(line)
            if forward_count and comment_count and like_count:
                fileout222.write(line)
            '''
            fileout.write(uid + '\t' + mid + '\t' + str(len(content)) + '\t')
            fileout.write(forward_count + ',' + 
                          comment_count + ',' + 
                          like_count + '\n')'''
        '''fileout.write('\t'.join([uid, mid, time, 
            forward_count, comment_count, like_count, 
            str(len(content)), content]))'''

def precision(predict, real):
    """实现计算精度的算法

    http://tianchi.aliyun.com/competition/information.htm?spm=0.0.0.0.31CeDM&raceId=5
    """

    find = re.compile('\d+,\d+,\d+')
    split = re.compile(',')

    precision_up = 0
    precision_down = 0

    for pred_i in predict:
        real_i = real.readline()
        fp, cp, lp = split.split(re.search(find, pred_i).group())
        fp, cp, lp = int(fp), int(cp), int(lp)
        # forward_predict, comment_predict, like_predict
        fr, cr, lr = split.split(re.search(find, real_i).group())
        fr, cr, lr = int(fr), int(cr), int(lr)

        # forward_real, comment_real, like_real
        dev_f = math.fabs(fp - fr) / (fr + 5)
        dev_c = math.fabs(cp - cr) / (cr + 3)
        dev_l = math.fabs(lp - lr) / (lr + 3)
        precision_i = 1 - 0.5 * dev_f - 0.25 * dev_c - 0.25 * dev_l
        #print(dev_f, dev_c, dev_l)
        count_i = fr + cr + lr
        if count_i > 100: # counti为第i篇博文的总的转发、评论、赞之和,当counti>100时，取值为100
            count_i = 100

        if precision_i - 0.8 > 0:
            precision_up += count_i + 1
        precision_down += count_i + 1
    print(precision_up, precision_down, precision_up / precision_down)

def get_data_user(filein, num_user):
    """读取 n 个用户的全部微博数据

    需要 uid 排序
    可用适当参数输出
    """
    t = re.compile('\t')

    uid0 = ''
    num_post = 0
    sum_f = 0
    sum_c = 0
    sum_l = 0
    i = 0
    j = 0
    fileout = open(filein_name[:-4] + '-' + str(num_user)+ '.txt', 'w', encoding='utf-8')
    while i < num_user: # write number of users as demand in num_user 
        line = filein.readline()
        j += 1
        if line:
            uid, mid, time, forward_count, comment_count, like_count, content = t.split(line)
            fileout.write(line)
        else:
            #fileout.write(line)
            #fileout.write('\n')
            print(j)
            return

        if uid != uid0: # new uid
            if uid0:
                pass
                #fileout.write(line)
                #fileout.write('\n')
            i += 1
            if i % 1000 == 0:
                print(i)
            uid0 = uid
            

            if i >= num_user:
                print(j)
                break
        else:
            num_post += 1
            
        #fileout.write('\t'.join([uid, mid, time, forward_count, comment_count, like_count, str(len(content)), content]))

if __name__ == '__main__':
    #predict_average(predict_file, predict_data, fileout)
    #print(len(filein.readlines()))
    #get_data_and_write(filein, fileout)
    #sep(filein)
    #get_data_user(filein, 1000)
    #predict_000(filein, fileout)
    #get_data(filein, fileout)
    #precision(predict, real)
    f1 = open('7-10-log_reg.txt', encoding='utf-8')
    for i in range(0,10):
        print(f1.readline())
    