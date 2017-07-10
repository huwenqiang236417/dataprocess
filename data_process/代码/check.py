#-*- coding: utf-8 -*-
#数据预处理，删除缺失值
import pandas as pd
import numpy as np

print ("Scripts starts...")

inputfile = './data/data.xls'
data = pd.read_excel(inputfile) 
x = data.iloc[:,0:46]
x.head()
x.dtypes
print (x.dtypes)

for i in range (len(x.d101)) :
	try:
		np.int64(x.d101[i])
	except:
		print (i)