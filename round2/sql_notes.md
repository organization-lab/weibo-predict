# weibo-predict round 2

## outline

- 基本信息
- 实现最基本的预测同一类别/平均数算法
  - 目标是熟悉系统(ODPS)
- 后续: 整体框架
  - feature 列表与实现思路
  - 基本分词与预测

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

forward_count/comment_count/like_count/all_count:

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

1011_combine_y:

uid, mid, blog, forward, comment, like, sum, y

### 基本分词与预测

