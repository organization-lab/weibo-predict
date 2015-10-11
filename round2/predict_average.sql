/* 

author: Frank-the-Obscure @ ChemDog
predict average 

method:
1. calculate uid average
2. left join to produce predict file

*/

select avg(count_all) from 1011_combine_y_1000 group by uid

/* calculate uid average and count by sql */
create table 1011_uid_average as 
select 
    uid as uid, 
    avg(count_all) as avg_uid,
    count(*) as count_post 
from 1011_combine_y group by uid;

/* predict uid class */
create table 1011_y_uid_ave as 
select *,
	case when avg_uid > 100 then 5
		 when avg_uid > 50 then 4
		 when avg_uid > 10 then 3
		 when avg_uid > 5 then 2
		 else 1 end as y_pred_average
from 1011_uid_average ;

/* refresh final table column name */
ALTER TABLE weibo_rd_2_submit CHANGE COLUMN y_pred_average RENAME TO action_sum;