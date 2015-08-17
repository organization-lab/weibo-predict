# -*- coding: utf-8 -*-
# Author: Frank-the-Obscure @ GitHub
# weibo-predict MVP

import re

f = open('weibo_predict_data.txt')

output = open('predict.txt', 'w')

t = re.compile('\t')

#print(t.split(f.readline()))


for line in f:
    data = t.split(line)
    del data[2:]
    data.append('0,0,0')
    output.write(' '.join(data))
    output.write('\n')