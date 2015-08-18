# -*- coding: utf-8 -*-
# Author: Frank-the-Obscure @ GitHub
# get uid

import re

train_data = open('weibo_train_data.txt')
#train_only = open('train_b.txt')

output = open('first1000user.txt', 'w')

#set_a = open('train_a2.txt', 'w')
#set_b = open('train_b2.txt', 'w')

t = re.compile('\t')

#print(t.split(f.readline()))

num_user = 1000
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

    output.write('\t'.join([uid, mid, time, forward_count, comment_count, like_count, str(len(content)), content]))
    


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