#-*- coding: utf-8 -*-
#建立决策树分类模型
from sklearn import tree
import pandas as pd
import time

print ("Scripts starts...")
start = time.time()

inputfile='./data/test.xls' #数据
data = pd.read_excel(inputfile) #读入数据
#outputfile1 = './tmp/tree1.xls' #模型输出文件
#outputfile2 = './tmp/tree2.xls' #模型输出文件
#outputfile3 = './tmp/tree3.xls' #模型输出文件
#outputfile4 = './tmp/tree4.xls' #模型输出文件
#outputfile5 = './tmp/tree5.xls' #模型输出文件

y1 = data.iloc[:,36].as_matrix() #样本标签列
x1 = data.iloc[:,0:4].as_matrix() #样本特征
clf = tree.DecisionTreeClassifier(splitter='random')
clf.fit(x1,y1)
clf.predict(x1)
end1 = time.time()
print ("modeltime: %f s" %(end1-start))
count1 = 0                                      #统计预测正确的结果个数
for left , right in zip(clf.predict(x1), y1):
      if left == right:
            count1 += 1
print("饮食预测准确度为：%f" %(count1/len(y1)))

y2 = data.iloc[:,36].as_matrix() #样本标签列
x2 = data.iloc[:,5:8].as_matrix() #样本特征
clf = tree.DecisionTreeClassifier(splitter='random')
clf.fit(x2,y2)
clf.predict(x2)
end2 = time.time()
print ("modeltime: %f s" %(end2-end1))
count2 = 0                                      #统计预测正确的结果个数
for left , right in zip(clf.predict(x2), y2):
      if left == right:
            count2 += 1
print("烟酒预测准确度为：%f" %(count2/len(y2)))

y3 = data.iloc[:,36].as_matrix() #样本标签列
x3 = data.iloc[:,9:11].as_matrix() #样本特征
clf = tree.DecisionTreeClassifier(splitter='random')
clf.fit(x3,y3)
clf.predict(x3)
end3 = time.time()
print ("modeltime: %f s" %(end3-end2))
count3 = 0                                      #统计预测正确的结果个数
for left , right in zip(clf.predict(x3), y3):
      if left == right:
            count3 += 1
print("锻炼预测准确度为：%f" %(count3/len(y3)))

y4 = data.iloc[:,36].as_matrix() #样本标签列
x4 = data.iloc[:,12:20].as_matrix() #样本特征
clf = tree.DecisionTreeClassifier(splitter='random')
clf.fit(x4,y4)
clf.predict(x4)
end4 = time.time()
print ("modeltime: %f s" %(end4-end3))
count4 = 0                                      #统计预测正确的结果个数
for left , right in zip(clf.predict(x4), y4):
      if left == right:
            count4 += 1
print("休闲预测准确度为：%f" %(count4/len(y4)))

y5 = data.iloc[:,36].as_matrix() #样本标签列
x5 = data.iloc[:,22:25].as_matrix() #样本特征
clf = tree.DecisionTreeClassifier(splitter='random')
clf.fit(x5,y5)
clf.predict(x5)
end5 = time.time()
print ("modeltime: %f s" %(end5-end4))
count5 = 0                                      #统计预测正确的结果个数
for left , right in zip(clf.predict(x5), y5):
      if left == right:
            count5 += 1
print("家庭情况预测准确度为：%f" %(count5/len(y5)))


#r1 = pd.DataFrame(clf.predict(x), columns = [u'预测结果'])
#pd.concat([data.iloc[:,:37], r1], axis = 1).to_excel(outputfile1)

end = time.time()
print ("endtime: %f s" %(end-start))
print ("Script ends...")