# -*- coding: utf-8 -*-
# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# precision function

import re
import math

from sys import argv

script, predict, real = argv

def precision(predict, real, output=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    """实现计算精度的算法

    predict: 提交格式
    real: 原始文件格式
    output: 输出(信息)用于算法 debug
    http://tianchi.aliyun.com/competition/information.htm?spm=0.0.0.0.31CeDM&raceId=5
    """
    if output:
        output_file = open(output, 'w', encoding='utf-8')
    find = re.compile('\d+,\d+,\d+')
    t = re.compile('\t')
    split = re.compile(',')

    precision_up = 0
    precision_down = 0
    totalfp, totalcp, totallp = 0,0,0
    totalfr, totalcr, totallr = 0,0,0

    xf, xc, xl = 0,0,0
    yf, yc, yl = 0,0,0

    list_xf, list_xc, list_xl = [], [], []
    list_yf, list_yc, list_yl = [], [], []

    for pred_i in predict:
        real_i = real.readline()
        #print(re.search(find, pred_i).group())
        if not pred_i:
            print ('nothing')
            break
        #print(pred_i, real_i)

        fp, cp, lp = split.split(re.search(find, pred_i).group())
        fp, cp, lp = int(fp), int(cp), int(lp)
        totalfp += fp
        totalcp += cp
        totallp += lp
        # forward_predict, comment_predict, like_predict
        (uid, mid, time, fr,
        cr, lr, content) = t.split(real_i)
        #fr, cr, lr = split.split(re.search(find, real_i).group())
        fr, cr, lr = int(fr), int(cr), int(lr)
        totalfr += fr
        totalcr += cr
        totallr += lr
        # forward_real, comment_real, like_real
        dev_f = math.fabs(fp - fr) / (fr + 5)
        dev_c = math.fabs(cp - cr) / (cr + 3)
        dev_l = math.fabs(lp - lr) / (lr + 3)
        precision_i = 1 - 0.5 * dev_f - 0.25 * dev_c - 0.25 * dev_l
        #print(dev_f, dev_c, dev_l)
        count_i = fr + cr + lr
        if count_i > 100: # counti为第i篇博文的总的转发、评论、赞之和,当counti>100时，取值为100
            count_i = 100
        list_xf.append(fp)
        list_yf.append(fr)
        list_xc.append(cp)
        list_yc.append(cr) 
        list_xl.append(lp)
        list_yl.append(lr)
        if precision_i - 0.8 > 0:
            precision_up += count_i + 1
        else:

            if fp > fr:
                xf += 1
            elif fp < fr:
                yf += 1
            if cp > cr:
                xc += 1
            elif cp < cr:
                yc += 1
            if lp > lr:
                xl += 1
            elif lp < lr:
                yl += 1                
            '''
            if output: # 只输出不准的部分
                output_file.write(str(round(precision_i,2)) + '\t' +str(fp)+','+str(cp)+','+str(lp) +
                                  '\t'+ str(fr)+','+str(cr)+','+ str(lr) + '\t' +
                                  uid + '\t'+ mid + '\t'+ time+ '\t'+ content)'''
        precision_down += count_i + 1
    print(precision_up, precision_down, precision_up / precision_down)
    print('p', totalfp, totalcp, totallp)
    print('r', totalfr, totalcr, totallr)
    '''
    print(xf,yf)
    print(xc, yc)
    print(xl,yl)
    print(len(list_xc))
    '''
    plt.figure(1)
    plt.hist2d(list_xf,list_yf,range=[[0,20],[0,20]],bins=20, norm=LogNorm())
    plt.colorbar()
    plt.figure(2)
    plt.hist2d(list_xc,list_yc,range=[[0,20],[0,20]],bins=20, norm=LogNorm())
    plt.colorbar()
    plt.figure(3)
    plt.hist2d(list_xl,list_yl,range=[[0,20],[0,20]],bins=20, norm=LogNorm())  
    plt.colorbar()  
    plt.show()

precision(open(predict, encoding='utf-8'), 
    open(real, encoding='utf-8'), 
    output='precision_details.txt')