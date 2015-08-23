# -*- coding: utf-8 -*-
# Author: Frank-the-Obscure @ GitHub
# weibo-predict MVP

import re

f = open('first100user.txt')

#output = open('predict.txt', 'w')

output = open('predict_100.txt', 'w')

t = re.compile('\t')

#print(t.split(f.readline()))

for line in f:
    data = t.split(line)
    del data[2:]
    data.append('0,0,0')
    output.write('\t'.join(data))
    output.write('\n')