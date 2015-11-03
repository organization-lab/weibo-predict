--odps sql 
--********************************************************************--
--author:humour114
--create time:2015-11-03 22:51:43
--********************************************************************--


DROP TABLE IF EXISTS 1103_test_length ;
CREATE TABLE 1103_test_length as 
    SELECT *, length(blog) AS blog_length , weekday(blog_time) as weekday
    FROM weibo_blog_data_test;

select * from 1103_test_length;

-- join features for test set

DROP TABLE IF EXISTS 1103_test_all;

CREATE TABLE 1103_test_all
AS
SELECT t1.mid AS mid
	, t1.uid AS uid
	, t1.blog_length as blog_length
	, t2.avg_uid AS avg_uid
	, t2.count_post as count_post
	, t2.fans_count as fans_count
FROM 1103_test_length t1
LEFT OUTER JOIN 1103_uid_average t2
ON t1.uid = t2.uid;