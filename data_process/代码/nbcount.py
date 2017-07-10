#-*- coding: utf-8 -*-
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体  
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题 
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print ("Scripts starts...")

inputfile='./data/data.xls' #数据
data = pd.read_excel(inputfile) #读入数据

i = 0
rate = []
while i < 16948 :
	
	i += 500
	y_train = data.iloc[0:i,62].as_matrix() #训练样本标签列
	x_train = data.iloc[0:i,0:46].as_matrix() #训练样本特征
	y_test = data.iloc[:,62].as_matrix() #测试样本标签列
	x_test = data.iloc[:,0:46].as_matrix() #测试样本特征

	clf = MultinomialNB().fit(x_train, y_train)  #训练多项式贝叶斯分类器模型 
	clf.predict(x_test)  #预测结果

	count = 0                                      #统计预测正确的结果个数
	for left , right in zip(clf.predict(x_test), y_test):
	      if left == right:
	            count += 1
	k = count/len(y_test)
	rate.append(k)

x=range(len(rate))
y=rate
plt.plot(x,y)
plt.xlabel(u'训练集数/*500个')
plt.ylabel(u'预测准确率/%')
plt.title(u'朴素贝叶斯算法模型')
plt.show()

print ("Script ends...")