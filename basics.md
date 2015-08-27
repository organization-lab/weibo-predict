# Basics

total uid: 46885 (= 45671+1214 = 24818+22067)

--

unique uid in train data: 45671

lines(posts): 1626750

month distribution:
07 279319
08 254104
09 258090
10 294307
11 268228
7-11 1254048
12 272702

--

post: 
000: 1053604
111 at least one non-zero: 573146 
100 forward: 273487 
010 comment: 352494
001 like: 328569
222 all non-zero: 108779

--

uid need to predict: 24818

lines(posts): 275331

--

uid only in train data: 22067

in both: 23604

only in predict data: 1214

---

重复的 uid 2个:

- 07fc721342df1a4c1992560b582992f8 出现在首尾
- 83b76a297161308e937ab9a7d71e9309 出现在 22807, 27376

处理后的文件: weibo_train_data_sort.txt

## 20150818

first 100 user & first 1000 user: len(content) added.

---

first 100 user:

000: 0.469
average: 0.446
linear regression: 0.456

---

first 1000 user:

000: 0.396
average: 0.456
linear regression: 0.464

need more test (larger database)

---

分词基本数据:

7-10-000:
>2 248998; >3 166350; >5 112346 >10: 69910
7-10-100:
123967; 86454; 58422; 35262
7-10-010:
130765; 92160; 62744; 37707
7-10-001:
124270; 87339; 59129; 35528

001 like

m:272702, true_pos:9029, false_pos:6408, false_neg:46320, true_neg:210945

010 comment

m:272702, true_pos:11842, false_pos:10817, false_neg:39854, true_neg:210189

100 forward

m:272702, true_pos:12088, false_pos:8028, false_neg:30717, true_neg:221869


372195 988187 0.37664429910533126
p 13430264012492 22517387061210 330688390792202
r 585911 253255 302213
Franks-Mac-mini:weibo-predict FrankHu$ python cut.py ohno
Finished: runtime 172.57033705711365
Franks-Mac-mini:weibo-predict FrankHu$ python precision.py 0827-predict-12.txt copy/12.txt
372397 988187 0.37684871385679025
p 491922 280655 230124
r 585911 253255 302213

---

note:

http://bbs.aliyun.com/read/253860.html?spm=5176.7189909.0.0.ebONvd

这里提到的数据好像不太对, 猜测是那哥们算错了.