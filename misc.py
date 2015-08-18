# -*- coding: utf-8 -*-
# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# misc function for i/o of basic data

import re
import math

#filein = open('predict_000.txt')
#output = open('predict.txt', 'w')
#fileout = open('predict_1000_average.txt', 'w')

predict = open('predict/predict_1000_average.txt') 
real = open('predict/predict_1000_real.txt') 

#predict_file = open('first1000user.txt')
#predict_data = open('uid_average.txt')

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

def get_data_user(filein, fileout, num_user):
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

    while i < num_user: # write number of users as demand in num_user 
        line = filein.readline()
        j += 1
        if line:
            uid, mid, time, forward_count, comment_count, like_count, content = t.split(line)
        else:
            fileout.write('\t'.join([uid0, str(num_post), str(sum_f/num_post),
                                         str(sum_c/num_post), str(sum_l/num_post)]))
            fileout.write('\n')
            print(j)
            return

        if uid != uid0: # new uid
            if uid0:
                fileout.write('\t'.join([uid0, str(num_post), str(sum_f/num_post),
                                         str(sum_c/num_post), str(sum_l/num_post)]))
                fileout.write('\n')
            i += 1
            if i % 1000 == 0:
                print(i)
            uid0 = uid
            num_post = 0
            sum_f = 0
            sum_c = 0
            sum_l = 0
            num_post += 1
            sum_f += int(forward_count)
            sum_c += int(comment_count)
            sum_l += int(like_count)

            if i >= num_user:
                print(j)
                break
        else:
            num_post += 1
            sum_f += int(forward_count)
            sum_c += int(comment_count)
            sum_l += int(like_count)

        #fileout.write('\t'.join([uid, mid, time, forward_count, comment_count, like_count, str(len(content)), content]))


if __name__ == '__main__':
    #predict_average(predict_file, predict_data, fileout)
    #print(len(filein.readlines()))
    #get_data_and_write(filein,fileout)
    #get_data_user(filein, fileout, 500000)
    #predict_000(filein, fileout)
    #get_data(filein, fileout)
    precision(predict, real)