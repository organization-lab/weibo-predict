/* stats */

CREATE TABLE 1101_stats as 
select *,
    case when count_all > 100 then 5
         when count_all > 50 then 4
         when count_all > 10 then 3
         when count_all > 5 then 2
         else 1 end as label
from 1101_left_join_filled ;
select label, count(label) as label_count from 1101_stats group by label;


CREATE TABLE 1101_stats_uid_ave
AS
SELECT *
    , CASE 
        WHEN avg_uid > 100 THEN 100
        WHEN avg_uid > 50 THEN 50
        WHEN avg_uid > 20 THEN 20
        WHEN avg_uid > 10 THEN 10
        WHEN avg_uid > 5 THEN 5
        WHEN avg_uid > 1 THEN 1
        WHEN avg_uid > 0 THEN 0
        ELSE -1
    END AS label
FROM 1101_uid_average;

SELECT label
    , COUNT(label) AS label_count
FROM 1101_stats_uid_ave
GROUP BY label;