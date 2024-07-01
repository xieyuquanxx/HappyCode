import os
import json
import numpy as np

with open("test.json",'r',encoding='utf-8') as load_f:
   datas = json.load(load_f)

print(len(datas))

print(datas[0])