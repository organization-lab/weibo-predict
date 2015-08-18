# -*- coding: utf-8 -*-
# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# misc function for i/o of basic data

import re

filein = open('first1000user.txt')
#output = open('predict.txt', 'w')
#fileout = open('predict_1000_000.txt', 'w')

predict = open('predict_1000_000.txt') 

real = open('predict_1000_real.txt') 

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

def get_data_and_write(filein, fileout):
    """用 readline 逐行读入数据并 unpack

    可用适当参数输出
    """
    t = re.compile('\t')
    for line in filein: # write number of users as demand in num_user   
        (uid, mid, time, forward_count,
         comment_count, like_count, num_content, content) = t.split(line)
        fileout.write(uid + '\t' + mid + '\t')
        fileout.write(forward_count + ',' + 
                      comment_count + ',' + 
                      like_count + '\n')

        '''fileout.write('\t'.join([uid, mid, time, 
            forward_count, comment_count, like_count, 
            str(len(content)), content]))'''

def precision(predict, real):
    """

    http://tianchi.aliyun.com/competition/information.htm?spm=0.0.0.0.31CeDM&raceId=5
    """
    import math

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

def get_data_user(filein, fileout, num_user):
    """读取 n 个用户的全部微博数据

    需要 uid 排序
    可用适当参数输出
    """
    t = re.compile('\t')

    uid0 = ''
    i = 0

    while i < num_user: # write number of users as demand in num_user 
        line = train_data.readline()
        uid, mid, time, forward_count, comment_count, like_count, content = t.split(line)

        if uid != uid0:
            i += 1
            uid0 = uid
            if i >= num_user:
                break

        fileout.write('\t'.join([uid, mid, time, forward_count, comment_count, like_count, str(len(content)), content]))


if __name__ == '__main__':
    #predict_000(filein, fileout)
    #get_data(filein, fileout)
    precision(predict, real)