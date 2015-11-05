--odps sql 
--********************************************************************--
--author:humour114
--create time:2015-11-04 22:21:24
--********************************************************************--
create table 1104_train_top100000 as 
SELECT * FROM 1104_tfidf_top order by ratio desc limit 100000;

select * from 1104_train_top100000


--odps sql 1104_action_ratio
--********************************************************************--
--author:humour114
--create time:2015-11-04 22:21:24
--********************************************************************--
create table 1104_train_action as 
SELECT *
	, y / (fans_count + 1) AS ratio
FROM 1103_train 

--odps sql see cv result
--********************************************************************--
--author:humour114
--create time:2015-11-03 19:39:07
--********************************************************************--

SELECT y_class
	, COUNT(*)
FROM 1103_rf_test_3
WHERE y_class = prediction_result
GROUP BY y_class;


--odps sql predict threshold
--********************************************************************--
--author:humour114
--create time:2015-11-01 21:42:02
--********************************************************************--
DROP TABLE IF EXISTS weibo_rd_2_submit_1104;

CREATE TABLE weibo_rd_2_submit_1104
AS
SELECT *
	, CASE 
		WHEN avg_uid > 55 THEN 101
		WHEN avg_uid > 30 THEN 51
		WHEN avg_uid > 3 THEN 11
		WHEN avg_uid > 2 THEN 6
		ELSE 0
	END AS y_sql
FROM 1103_test_rf;

select * from weibo_rd_2_submit_1104;



--odps sql cast
--********************************************************************--
--author:humour114
--create time:2015-11-02 20:21:07
--********************************************************************--

CREATE TABLE 1102_matrix_string
AS
SELECT uid AS uid
	, mid AS mid
	, CAST(y_real AS STRING) AS y_real
	, CAST(y_predict AS STRING) AS y_predict
FROM 1102_confusion_matrix;

--odps sql  action ratio
--********************************************************************--
--author:humour114
--create time:2015-11-04 22:21:24
--********************************************************************--
create table 1104_train_action as 
SELECT *
	, y / (fans_count + 1) AS ratio
FROM 1103_train 