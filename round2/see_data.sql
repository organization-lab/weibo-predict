show tables; /* show all tables */

select * from total_count;

select * from total_count limit 1000; /* max show 5000 */

select count(*) from weibo_rd_2_submit; /* calculate num of lines in table */

select * from weibo_rd_2_submit where uid = '00008a06c0c43e9097ee6316961dbed7'

select sum(action_sum) as sum_y from weibo_rd_2_submit;