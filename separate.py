# -*- coding: utf-8 -*-
# Author: Frank-the-Obscure @ GitHub
# get uid

import re

train_data = open('train_a.txt')
train_only = open('train_b.txt')

#output = open('predict.txt', 'w')

set_a = open('train_a2.txt', 'w')
set_b = open('train_b2.txt', 'w')

t = re.compile('\t')

#print(t.split(f.readline()))

train_data = train_data.readlines()
train_only = train_only.readlines()

print(len(train_data))
print(len(train_only))

'''
for i in range(0, len(train_data)):
    line = train_data[i]
    if i % 10000 == 0:
        print(i)
    uid = t.split(line)[0]
    if uid + '\n' in a:
        set_a.write(line) # A 集合 是 train only user, 可作为训练集
    else:
        set_b.write(line) 
'''
'''
    data = t.split(line)
    del data[2:]
    data.append('0,0,0')
    output.write(' '.join(data))
    output.write('\n')'''