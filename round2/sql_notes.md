# weibo-predict round 2

## outline

- 0. 实现最基本的预测同一类别/平均数算法
- 目标是熟悉系统(ODPS)

## details

files

fans_count:

- uid: string
- fans_count: bigint

forward_count/comment_count/like_count/all_count:

- mid: string
- ...(forward/comment/like)_count: bigint