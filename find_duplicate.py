# -*- coding: utf-8 -*-
# Author: Frank-the-Obscure @ GitHub
# compare uids

import re

predict = open('predict_uid.txt')
train = open('train_uid.txt')
u1= open('1uid_predict_only.txt')

#print(t.split(f.readline()))

p = predict.readlines()
t = train.readlines()

num_dup = 0

s = []

for uid in t:
    if uid in s:
        num_dup += 1
    else:
        s.append(uid)

print(num_dup, len(s))