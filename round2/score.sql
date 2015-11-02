--odps sql 
--********************************************************************--
--author:humour114
--create time:2015-11-02 23:04:02
--********************************************************************--

drop table if exists 1102_confusion_matrix_12;

CREATE TABLE 1102_confusion_matrix_12
AS
SELECT *
	, CASE 
		WHEN avg_uid > 55 THEN 5
		WHEN avg_uid > 25 THEN 4
		WHEN avg_uid > 3 THEN 3
		WHEN avg_uid > 2.5 THEN 2
		ELSE 1
	END AS y_predict
FROM 1102_y_predict;

select count(*) from 1102_confusion_matrix_12 where y_real = 1 and y_predict = 1;
select count(*) from 1102_confusion_matrix_12 where y_real = 2 and y_predict = 2;
select count(*) from 1102_confusion_matrix_12 where y_real = 3 and y_predict = 3;
select count(*) from 1102_confusion_matrix_12 where y_real = 4 and y_predict = 4;
select count(*) from 1102_confusion_matrix_12 where y_real = 5 and y_predict = 5;