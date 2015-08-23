# -*- coding: utf-8 -*-
# Author: Frank-the-Obscure @ GitHub
# compare uids

import re

predict = open('predict_uid.txt')
train = open('train_uid.txt')

#output = open('predict.txt', 'w')

train_only = open('1uid_train_only.txt', 'w')
predict_only = open('1uid_predict_only.txt', 'w')
both = open('1uid_both.txt', 'w')
both2 = open('1uid_both2.txt', 'w')

t = re.compile('\t')

#print(t.split(f.readline()))

p = predict.readlines()
t = train.readlines()

to = 0
po = 0
b = 0

for uid in t:
    if uid in p:
        b += 1
        both.write(uid)
    else:
        to += 1
        train_only.write(uid)

print(to, po, b)
to = 0
po = 0
b = 0

for uid in p:
    if uid in t:
        b += 1
        both2.write(uid)
    else:
        po += 1
        predict_only.write(uid)

print(to, po, b)

'''
    data = t.split(line)
    del data[2:]
    data.append('0,0,0')
    output.write(' '.join(data))
    output.write('\n')'''