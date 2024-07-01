import os
import json
import numpy as np

with open("test.json",'r',encoding='utf-8') as load_f:
   datas = json.load(load_f)
for i in range(len(datas)):
   assert len(datas[i]['choosen'])==len(datas[i]['rejected']),'not equal'
print(len(datas))

