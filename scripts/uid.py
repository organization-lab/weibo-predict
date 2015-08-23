# -*- coding: utf-8 -*-
# Author: Frank-the-Obscure @ GitHub
# get uid

import re

f = open('weibo_predict_data.txt')

#output = open('predict.txt', 'w')

output = open('pre_uid.txt', 'w')

t = re.compile('\t')

#print(t.split(f.readline()))

users = 0
uid0 = ''

for line in f:
    uid = t.split(line)[0]
    if uid == uid0:
        continue
    else:
        uid0 = uid
        users += 1
        output.write(uid + '\n')

print(users)

'''
    data = t.split(line)
    del data[2:]
    data.append('0,0,0')
    output.write(' '.join(data))
    output.write('\n')'''