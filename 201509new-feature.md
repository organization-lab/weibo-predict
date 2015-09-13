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

## 提取 features

标准: mean * stdev 越大越好, 在每类中都按大小排序

选择结果: 很奇怪的, 最后一类(100+)一个都没有?! 按道理可能性非常小, 怀疑代码有问题..不过暂时不管

