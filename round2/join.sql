/* not work in ODPS */

select a.shop_name as ashop, b.shop_name as bshop from shop a
        left outer join sale_detail b on a.shop_name=b.shop_name;

create table 1010_combine 
	as select t1.mid AS mid,t1.uid AS uid,t2.count_all AS count_all 
	from weibo_action_data_train t1 
	LEFT OUTER JOIN total_count t2 ON t1.mid=t2.mid)
