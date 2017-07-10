#-*- coding: utf-8 -*-
#建立k最近邻算法分类模型
from sklearn import neighbors
import pandas as pd
import time

print ("Scripts starts...")
start = time.time()

inputfile='data.xls' #数据
outputfile = 'knn.xls' #模型输出文件
data = pd.read_excel(inputfile) #读入数据
y = data.iloc[:,62].as_matrix() #测试样本标签列
x = data.iloc[:,0:46].as_matrix() #测试样本特征

clf = neighbors.KNeighborsClassifier()
clf.fit(x,y)
clf.predict(x)
end1 = time.time()
print ("modeltime: %f s" %(end1-start))

count = 0                                      #统计预测正确的结果个数
for left , right in zip(clf.predict(x), y):
      if left == right:
            count += 1
print("预测准确度为：%f" %(float(count)/len(y)))

r = pd.DataFrame(clf.predict(x), columns = [u'预测结果'])
pd.concat([data.iloc[:,:63], r], axis = 1).to_excel(outputfile)

end2 = time.time()
print ("endtime: %f s" %(end2-start))
print ("Script ends...")
