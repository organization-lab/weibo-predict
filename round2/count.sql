/*out put counts to three tables*/

DROP TABLE IF EXISTS forward_count ;
CREATE TABLE forward_count as 
    SELECT mid, COUNT(*) AS forward_count 
        FROM tianchi_weibo.weibo_action_data_train
        where action_type = 1 
        group by mid;
DROP TABLE IF EXISTS comment_count ;
CREATE TABLE comment_count as 
    SELECT mid, COUNT(*) AS comment_count 
        FROM tianchi_weibo.weibo_action_data_train
        where action_type = 2 
        group by mid;
DROP TABLE IF EXISTS like_count ;
CREATE TABLE like_count as 
    SELECT mid, COUNT(*) AS like_count 
        FROM tianchi_weibo.weibo_action_data_train
        where action_type = 3 
        group by mid;

/*count all*/
DROP TABLE IF EXISTS total_count ;
CREATE TABLE total_count as 
    SELECT mid, COUNT(*) AS count_all 
    FROM tianchi_weibo.weibo_action_data_train 
    group by mid;