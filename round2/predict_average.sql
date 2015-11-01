/* 

author: Frank-the-Obscure @ ChemDog
predict average 

method:
1. calculate uid average
2. left join to produce predict file

*/

/* 1.1 cal total count by mid, from weibo action data */
DROP TABLE IF EXISTS total_count ;
CREATE TABLE total_count as 
    SELECT mid, COUNT(*) AS count_all 
    FROM tianchi_weibo.weibo_action_data_train 
    group by mid;

/* 1.2 left join uid and mid and count */
drop table if exists 1101_left_join;
create table 1101_left_join as 
select t1.mid AS mid,
	t1.uid AS uid,
	t2.count_all AS action_sum 
from weibo_blog_data_train t1 LEFT OUTER JOIN total_count t2 
ON t1.mid=t2.mid

/* fill blank (用缺失值填充, 算法平台)*/

/* 1.4 calculate uid average and count by sql */
create table 1101_uid_average as 
select 
    uid as uid, 
    avg(action_sum) as avg_uid,
    count(*) as count_post 
from 1101_left_join_filled group by uid;

/* predict uid class */
create table 1011_y_uid_ave as 
select *,
	case when avg_uid > 100 then 5
		 when avg_uid > 50 then 4
		 when avg_uid > 10 then 3
		 when avg_uid > 5 then 2
		 else 1 end as y_pred_average
from 1011_uid_average ;

/* left outer join from 算法平台 */
drop table if exists 1101_left_join;
create table 1101_left_join as 
select t1.mid AS mid,
	t1.uid AS uid,
	t2.count_all AS action_sum 
from weibo_blog_data_train t1 LEFT OUTER JOIN total_count t2 
ON t1.mid=t2.mid


/* refresh final table column name */
ALTER TABLE weibo_rd_2_submit CHANGE COLUMN y_pred_average RENAME TO action_sum;

/* 设计规则 */

DROP TABLE IF EXISTS weibo_rd_2_submit_1101;

CREATE TABLE weibo_rd_2_submit_1101
AS
SELECT *
	, CASE 
		WHEN avg_uid > 50 THEN 101
		WHEN avg_uid > 20 THEN 51
		WHEN avg_uid > 5 THEN 11
		WHEN avg_uid > 1 THEN 6
		ELSE 0
	END AS action_sum
FROM 1101_predict;

select * from weibo_rd_2_submit_1101;


/* 去掉辅助列(avg_uid), 输出正式文件 weibo_rd_2_submit */
DROP TABLE IF EXISTS weibo_rd_2_submit;

CREATE TABLE weibo_rd_2_submit
AS
SELECT uid as uid,
mid as mid,
action_sum as action_sum
FROM weibo_rd_2_submit_1101;

select * from weibo_rd_2_submit;