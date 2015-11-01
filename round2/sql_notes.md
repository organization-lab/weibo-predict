# weibo-predict round 2

## outline

- 基本信息
- 实现最基本的预测同一类别/平均数算法
  - 目标是熟悉系统(ODPS)
- 后续: 整体框架
  - feature 列表与实现思路
  - 基本分词与预测
- 201511 final version

## 基本信息

[detailed link](http://tianchi.aliyun.com/competition/information.htm?spm=5176.100071.5678.2.25q5Wa&raceId=5)

### 1. 博文数据: `weibo_blog_data_train`

|字段|类型|说明|
|uid|string|用户ID|
|mid|string|博文ID|
|blog_time|string|发微博时间|
|blog|string|博文内容|

共 110541018 条记录

### 2. 粉丝数据: `weibo_fans_data_train`

已整理为fans_count:

- uid: string
- fans_count: bigint

### 3. 用户行为数据: `weibo_action_data_train`  

已整理为 total forward/comment/count

forward\_count/comment\_count/like\_count/all\_count:

- mid: string
- ...(forward/comment/like)_count: bigint

### 4. 需要预测的博文数据 `weibo_blog_data_test`

uid, mid, blog_time, blog

共 9644805 条记录

### 5. 输出表 `weibo_rd_2_submit`

需要请按照以下数据结构来产出结果表，命名为 `weibo_rd_2_submit`

数据属性名 类型 定义
uid string 用户ID
mid string 博文ID
action_sum bigint 转评赞互动数之和

## MVP 实现纪要

基本操作: 复赛数据都在天池平台上操作; 需要使用阿里提供的 ODPS SQL 和 算法平台

数据都是 SQL 表, 初期遇到了相当大的上手困难.

最大的感受: **(可用于复制的)正确代码极其重要**, 在论坛找到了几段可用代码后, 模仿起到了很大作用.

另外, 文档其实写得不算很糟糕, 仔细看看还是能测试的. 但初期寻找文档和阅读效果并不甚好(不过逐渐深入可能正是常态, 只有明白了一点基础内容, 才能看懂更深入的东西).

## 后续整体框架

Angrew Ng 在课程中反复提到: engineering time 是最宝贵的财富, 一定要善加利用.

因此这里首先考虑整体框架与实现思路, 希望能通过系统化思维, 如定期的算法评估确定需要优化的主要问题, 优化整体思路.

### pipeline

- train set / CV set / test set: 复赛有足够数据, 应进行合理分组
- evaluate algorithm: 
  - know performance: 设法实现用标准文件的对照 (SQL任务)
  - learning curve: 确定算法是 high bias 还是 high variance
- final test set (初期可只用训练集的数据, 最后提交有必要用完整数据集)
- 对切换数据的准备

### feature 列表与实现思路

初赛版本的 feature: 

- uid level: average forward / comment / like
- post level: dict word cut / length

1011\_combine\_y:

uid, mid, blog, forward, comment, like, sum, y

### 基本分词与预测

## 20151101 final version

摘要: 由于前期学习和找工作太忙, 天池属于弃坑状态, (不会写 JAVA 是一个安慰自己的理由). 不过进入最后阶段, 在周日晚上闲来写几行娱乐一下.

发现了第二赛季切换数据前的问题: 预测的是 action_sum 而非档位, 所以当时用1-5预测档位必然都被归到0档了...[Reference](http://bbs.aliyun.com/read/259533.html?spm=5176.bbsl254.0.0.Ax4a32)

uid average: 

1. 用 action 计算 mid count; `total_count`
2. left join 到 uid 填充缺失值为0; `1101_left_join_filled`
3. cal average: `1101_uid_average`
4. 设计规则预测`weibo_rd_2_submit_1101`
5. 去掉辅助列(avg_uid), 输出正式文件`weibo_rd_2_submit`

### 统计各类微博与用户 count
`1101_stats`:
label, label_count
1,106921008
2,1446189
3,1499997
4,301801
5,372023

`1101_stats_uid_ave`:
三零 -1,1376142
\>0,464491
1,112062
5,7210
10,2797
20,1705
50,617
100,658

设计规则:

1. uid_ave <= 1, predict 0
2. uid_ave > 1, predict 6 (考虑到得分是十倍, 因此有10%的是下一档就值得预测为下一档)
3. uid_ave > 5, predict 11
4. uid_ave > 20, predict 51
5. uid_ave > 50, predict 101
(严格说来这个方法需要用一个测试集来验证, 进行 grid search 是正道...)

`weibo_rd_2_submit_1101`

mid, uid, avg_uid, action_sum

`weibo_rd_2_submit`

### todo

采用部分训练集(其实可以使用全部训练集, 这里暂时不用考虑过拟合问题)计算正确率, 进行 grid search 寻找最佳规则参数. (目标 top 50)
