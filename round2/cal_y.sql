/* calculate y */

create table 1011_combine_y as 
select *,
	case when count_all > 100 then 5
		 when count_all > 50 then 4
		 when count_all > 10 then 3
		 when count_all > 5 then 2
		 else 1 end as y
from 1011_combine ;