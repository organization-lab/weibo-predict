# New feature - 201509

起因: 原先的模型效果不佳, 且发现 feature 似乎比算法更重要.

目标: 优化 feature, 提高预测效果

整体框架

- 重构 feature 词典
- 整理 已知信息



## 重构 feature 字典

重新分类:

0 / 1-5 / 6-10 / 11-20 / 21-50 / 51-100 / 101+

共7类:

post 数量:

forward0 1353263
comment0 1274256
like0 1298181
forward1 201375
comment1 286457
like1 286306
forward2 27361
comment2 39451
like2 20912
forward3 18451
comment3 17290
like3 9888
forward4 14086
comment4 6771
like4 7133
forward5 6613
comment5 1398
like5 2630
forward6 5601
comment6 1127
like6 1700

分词提取出现次数 >= 2 的词, 共计 406491 个 (包括了 标点 \n 等)
forward0 360567
forward1 120732
forward2 34264
forward3 26472
forward4 22492
forward5 12993
forward6 11486
comment0 359775
comment1 136057
comment2 39094
comment3 22694
comment4 12026
comment5 4046
comment6 3175
like0 361259
like1 137438
like2 27222
like3 16572
like4 12836
like5 5827
like6 3720

### programming note: list 遍历 in/not in 非常慢, 但 dict 则极快.

```python
if word[0] not in words_list:
	words_list[word[0]] = 0
```

如果 words_list 是列表(100k 量级), 代码运行速度相当慢...

但 先改写成 dict 之后, 查询速度则可忽略

猜测是 hash 的帮助.

## 提取 features

标准: mean * stdev 越大越好, 在每类中都按大小排序

[`numpy.mean()`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html) 

[`numpy.std()`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html)

选择结果: 很奇怪的, 最后一类(100+)一个都没有?! 按道理可能性非常小, 怀疑代码有问题..不过暂时不管

todo: 这里还可以考虑用其它手段降维: 如训练一个分类器, 然后找出分类器中的主要 feature 对应的单词

## 导出 X, y

导出 X 相对较慢: 30 万行要用 10 min

考虑到内存可能不足, 把 162 万行分为五个部分导出

## 回归

X 合并: 注意不同格式的坑, numpy.ndarray 和 scipy.sparse 混合会出问题

CV result

### linear regression

6004 features

R^2 = 0.1018

69s

4 features

R^2 = 0.1143

2.35s

### random forest

`n_estimators=10, max_features='sqrt'`时 速度尚可, 但用全体数据则很慢

2378s

先用前 50万数据 预测, 节约时间

### SVR

linear kernel sample = 5000 即超慢...

暂时 pass

## 预测

12.txt 有过拟合嫌疑

f 全部数据 c/l 40万

443076 988187 0.448372625828917
p 623745 249663 287834
r 585911 253255 302213

历史最好成绩!!!

c/l 用 80万 dataset 重新拟合

494432 988187 0.5003425465018261
p 623745 258803 380045
r 585911 253255 302213

唔, 到底是过拟合的贡献还是其它? 提交就知道大概了...

过拟合问题, 要用原来计划解决(7月作为CV, 8-12月训练)

发现一个 length 的 bug 修复后:

546201 988187 0.5527304042655894
p 564083 252520 280086
r 585911 253255 302213